import os
from pathlib import Path
from uuid import uuid4

from dotenv import load_dotenv

BACKEND_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BACKEND_DIR / ".env")


from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool
from starlette.responses import JSONResponse

from app.memory.database import init_database
from app.agent.llm import ModelServiceUnavailableError
from app.memory.service import ConversationNotFoundError, ConversationService
from app.observability import configure_logging
from app.rag.service import index_document, semantic_search
from app.rag.document_parser import DocumentParseError, parse_document
from app.rag.catalog import (
    append_knowledge_documents,
    delete_knowledge_document as delete_knowledge_document_metadata,
    delete_knowledge_base as delete_knowledge_base_metadata,
    get_knowledge_base,
    get_knowledge_document,
    knowledge_base_exists,
    list_knowledge_bases as list_knowledge_base_metadata,
    register_knowledge_base,
)
from app.rag.vector_store import (
    delete_chunks_by_source,
    delete_chunks_by_upload_batch,
    delete_collection,
)
from app.security import (
    InMemoryRateLimiter,
    PromptInjectionError,
    access_token_required,
    verify_bearer_token,
)

configure_logging()
app = FastAPI()
init_database()
conversation_service = ConversationService()
rate_limiter = InMemoryRateLimiter(
    int(os.getenv("API_RATE_LIMIT_PER_MINUTE", "60"))
)

configured_origins = [
    origin.strip()
    for origin in os.getenv("CORS_ALLOWED_ORIGINS", "").split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=configured_origins,
    # 未配置正式域名时，仅允许本地 Vite 开发端口。
    allow_origin_regex=(
        None
        if configured_origins
        else r"^http://(localhost|127\.0\.0\.1):\d{1,5}$"
    ),
    allow_credentials=False,
    allow_methods=["GET", "POST", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type"],
)


@app.middleware("http")
async def protect_api(request: Request, call_next):
    """为成本敏感接口提供可选访问令牌和单实例限流。"""
    if request.url.path.startswith("/api/") and request.url.path != "/api/health":
        if not verify_bearer_token(request.headers.get("authorization")):
            return JSONResponse(status_code=401, content={"detail": "访问令牌无效"})
        client_host = request.client.host if request.client else "unknown"
        if not rate_limiter.allow(client_host):
            return JSONResponse(status_code=429, content={"detail": "请求过于频繁"})
    return await call_next(request)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    knowledge_base_id: str | None = Field(
        default=None,
        min_length=1,
        max_length=64,
        pattern=r"^[A-Za-z0-9_-]+$",
    )
    conversation_id: str | None = Field(
        default=None,
        pattern=r"^[a-f0-9]{32}$",
    )


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=4000)
    knowledge_base_id: str | None = Field(
        default=None,
        min_length=1,
        max_length=64,
        pattern=r"^[A-Za-z0-9_-]+$",
    )


class ConversationCreateRequest(BaseModel):
    title: str = Field(default="新会话", min_length=1, max_length=60)


class ConversationUpdateRequest(BaseModel):
    title: str = Field(min_length=1, max_length=60)


MAX_FILE_SIZE = 10 * 1024 * 1024
MAX_UPLOAD_SIZE = 30 * 1024 * 1024
MAX_FILES_PER_UPLOAD = 10


def find_relevant_chunks(knowledge_base_id: str | None, query: str):
    if not knowledge_base_id:
        return []

    try:
        return semantic_search(knowledge_base_id, query)
    except RuntimeError as error:
        raise HTTPException(status_code=502, detail="知识库检索暂时不可用") from error


@app.post("/api/documents/search")
def search_document(request: SearchRequest):
    matches = find_relevant_chunks(request.knowledge_base_id, request.query)

    return {
        "query": request.query,
        "results": [
            {
                "text": match.document,
                "distance": round(match.distance, 4),
                "similarity": round(match.similarity, 4),
            }
            for match in matches
        ],
    }


@app.get("/api/health")
def health_check():
    return {
        "status": "ok",
        "message": "Hello from AI Agent Backend",
        "auth_required": access_token_required(),
    }


@app.post("/api/chat")
def chat(request: ChatRequest):
    """兼容旧客户端；所有聊天请求统一进入 Agent 执行链。"""
    return _chat(request)


def _chat(request: ChatRequest):
    message = request.message.strip()

    if not message:
        raise HTTPException(status_code=400, detail="消息不能为空")

    try:
        turn = conversation_service.chat(
            message=message,
            knowledge_base_id=request.knowledge_base_id,
            conversation_id=request.conversation_id,
        )
        result = turn.agent
        return {
            "conversation_id": turn.conversation_id,
            "history_messages": turn.history_messages,
            "trace": turn.trace.to_dict(),
            "answer": result.answer,
            "matched_chunks": result.matched_chunks,
            "llm_called": result.llm_called,
            "agent": {
                "tool_called": result.tool_called,
                "tool_name": result.tool_name,
                "steps": result.steps,
                "tools_used": result.tools_used,
                "active_skill": result.active_skill,
                "tool_source": result.tool_source,
                "mcp_server": result.mcp_server,
            },
        }

    except ConversationNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except ModelServiceUnavailableError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(
            status_code=502,
            detail="Agent 执行失败，请检查模型、工具参数和后端日志",
        ) from error


@app.post("/api/agent/chat")
def agent_chat(request: ChatRequest):
    """Agent 模式允许模型自主选择经过注册和校验的工具。"""
    return _chat(request)


@app.get("/api/conversations/{conversation_id}/messages")
def get_conversation_messages(
    conversation_id: str,
    limit: int = Query(default=100, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
):
    if len(conversation_id) != 32 or any(
        character not in "0123456789abcdef" for character in conversation_id
    ):
        raise HTTPException(status_code=404, detail="会话不存在")
    try:
        messages = conversation_service.get_history_with_traces(
            conversation_id,
            limit=limit,
            offset=offset,
        )
    except ConversationNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    return {
        "conversation_id": conversation_id,
        "messages": [
            {
                "role": item["role"],
                "content": item["content"],
                "created_at": item["created_at"],
                "trace": item.get("trace"),
            }
            for item in messages
        ],
    }


@app.get("/api/conversations")
def list_conversations(
    limit: int = Query(default=50, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
):
    return {
        "conversations": conversation_service.list_conversations(
            limit=limit,
            offset=offset,
        )
    }


@app.post("/api/conversations", status_code=201)
def create_conversation(request: ConversationCreateRequest):
    title = request.title.strip()
    if not title:
        raise HTTPException(status_code=400, detail="会话标题不能为空")
    return conversation_service.create_conversation(title)


@app.patch("/api/conversations/{conversation_id}")
def rename_conversation(conversation_id: str, request: ConversationUpdateRequest):
    title = request.title.strip()
    if not title:
        raise HTTPException(status_code=400, detail="会话标题不能为空")
    try:
        return conversation_service.rename_conversation(
            conversation_id, title
        )
    except ConversationNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error


@app.delete("/api/conversations/{conversation_id}", status_code=204)
def delete_conversation(conversation_id: str):
    try:
        conversation_service.delete_conversation(conversation_id)
    except ConversationNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error


@app.post("/api/documents/upload")
async def upload_document(
    files: list[UploadFile] = File(default=[]),
    file: UploadFile | None = File(default=None),
    knowledge_base_id: str | None = Form(default=None),
):
    """把一次选择的多份文档索引到同一个知识库，并兼容旧版单文件字段。"""
    uploaded_files = [*files, *([file] if file is not None else [])]
    if not uploaded_files:
        raise HTTPException(status_code=400, detail="请选择至少一个文档")
    if len(uploaded_files) > MAX_FILES_PER_UPLOAD:
        raise HTTPException(status_code=400, detail="一次最多上传 10 个文档")

    parsed_documents: list[tuple[str, str]] = []
    total_size = 0
    for uploaded_file in uploaded_files:
        # Path.name 去掉浏览器可能传入的目录信息，仅保留安全展示名称。
        filename = Path(uploaded_file.filename or "").name
        content = await uploaded_file.read(MAX_FILE_SIZE + 1)
        total_size += len(content)
        if len(content) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413,
                detail=f"{filename or '文档'} 不能超过 10 MB",
            )
        if total_size > MAX_UPLOAD_SIZE:
            raise HTTPException(status_code=413, detail="单次上传总大小不能超过 30 MB")
        if not content:
            raise HTTPException(status_code=400, detail=f"{filename or '文档'} 内容为空")
        try:
            text = await run_in_threadpool(parse_document, filename, content)
        except DocumentParseError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        if not text.strip():
            raise HTTPException(
                status_code=400,
                detail=f"无法从 {filename} 识别有效文字；请确认扫描清晰、方向正确且页数不超过限制",
            )
        parsed_documents.append((filename, text))

    is_new_knowledge_base = knowledge_base_id is None
    if knowledge_base_id is not None and not knowledge_base_exists(knowledge_base_id):
        raise HTTPException(status_code=404, detail="要追加的知识库不存在")
    knowledge_base_id = knowledge_base_id or uuid4().hex
    upload_batch = uuid4().hex
    document_metadata: list[dict[str, str | int]] = []

    def rollback_upload() -> None:
        if is_new_knowledge_base:
            delete_collection(knowledge_base_id)
        else:
            delete_chunks_by_upload_batch(knowledge_base_id, upload_batch)

    try:
        for filename, text in parsed_documents:
            chunk_count = await run_in_threadpool(
                index_document,
                knowledge_base_id,
                text,
                filename,
                upload_batch,
            )
            if chunk_count == 0:
                raise DocumentParseError(f"{filename} 中没有可用的文本内容")
            document_metadata.append(
                {
                    "filename": filename,
                    "chunk_count": chunk_count,
                    # 保存批次标识，删除同名文档时只删除用户选中的那一次上传。
                    "upload_batch": upload_batch,
                }
            )
    except PromptInjectionError as error:
        rollback_upload()
        raise HTTPException(status_code=400, detail=str(error)) from error
    except DocumentParseError as error:
        rollback_upload()
        raise HTTPException(status_code=400, detail=str(error)) from error
    except RuntimeError as error:
        rollback_upload()
        raise HTTPException(status_code=502, detail="文档处理服务暂时不可用") from error

    try:
        if is_new_knowledge_base:
            register_knowledge_base(knowledge_base_id, document_metadata)
        else:
            append_knowledge_documents(knowledge_base_id, document_metadata)
    except Exception:
        # 元数据保存失败时只回收本批次向量，已有知识库不受影响。
        rollback_upload()
        raise HTTPException(status_code=500, detail="知识库元数据保存失败")

    knowledge_base = get_knowledge_base(knowledge_base_id)
    return {
        "success": True,
        "documents": knowledge_base["documents"] if knowledge_base else document_metadata,
        "added_documents": document_metadata,
        "chunks": knowledge_base["chunk_count"] if knowledge_base else 0,
        "knowledge_base_id": knowledge_base_id,
    }


@app.get("/api/knowledge-bases")
def list_knowledge_bases(
    limit: int = Query(default=50, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
):
    return {
        "knowledge_bases": list_knowledge_base_metadata(
            limit=limit,
            offset=offset,
        )
    }


@app.delete("/api/knowledge-bases/{knowledge_base_id}", status_code=204)
def delete_knowledge_base(knowledge_base_id: str):
    if not knowledge_base_exists(knowledge_base_id):
        raise HTTPException(status_code=404, detail="知识库不存在")
    delete_collection(knowledge_base_id)
    delete_knowledge_base_metadata(knowledge_base_id)


@app.delete("/api/knowledge-bases/{knowledge_base_id}/documents/{document_id}")
def delete_knowledge_document(knowledge_base_id: str, document_id: int):
    """只删除选中的文档及其向量，不影响同一知识库中的其他文档。"""
    document = get_knowledge_document(knowledge_base_id, document_id)
    if document is None:
        raise HTTPException(status_code=404, detail="文档不存在")
    try:
        if document.get("upload_batch"):
            delete_chunks_by_upload_batch(
                knowledge_base_id,
                document["upload_batch"],
            )
        else:
            # 兼容升级前没有批次标识的旧记录。
            delete_chunks_by_source(knowledge_base_id, document["filename"])
    except Exception as error:
        raise HTTPException(status_code=500, detail="文档向量删除失败") from error
    if not delete_knowledge_document_metadata(knowledge_base_id, document_id):
        raise HTTPException(status_code=404, detail="文档不存在")
    return get_knowledge_base(knowledge_base_id)
