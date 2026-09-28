from datetime import datetime, timedelta, timezone
import os
import re
import threading
from pathlib import Path
from uuid import uuid4

from dotenv import load_dotenv

BACKEND_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BACKEND_DIR / ".env")

from app.configuration import validate_runtime_configuration

validate_runtime_configuration()


from fastapi import (
    Depends,
    FastAPI,
    File,
    Form,
    HTTPException,
    Query,
    Request,
    Response,
    UploadFile,
)
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool
from starlette.responses import JSONResponse, StreamingResponse

from app.auth import AuthenticationError, Principal, resolve_principal
from app.agent.checkpoint import serialize_agent_state
from app.agent.run_manager import RunManager
from app.agent.service import GUEST_ALLOWED_TOOLS
from app.agent.llm import ModelServiceUnavailableError
from app.memory.database import init_database
from app.memory import repository
from app.memory.service import ConversationNotFoundError, ConversationService
from app.observability import configure_logging
from app.quotas import (
    GuestQuotaExceeded,
    get_guest_quota_status,
    hash_client_ip,
    reserve_guest_ai_request,
)
from app.rag.service import index_document, semantic_search
from app.rag.document_parser import DocumentParseError, parse_document
from app.rag.catalog import (
    append_knowledge_documents,
    delete_knowledge_document as delete_knowledge_document_metadata,
    delete_knowledge_base as delete_knowledge_base_metadata,
    get_knowledge_base,
    get_knowledge_document,
    get_public_knowledge_base,
    get_public_knowledge_base_ids,
    get_owner_knowledge_base_count,
    knowledge_base_is_owned_by,
    knowledge_base_exists,
    list_knowledge_bases as list_knowledge_base_metadata,
    list_public_knowledge_bases,
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
)
from app.streaming import format_sse

configure_logging()
app = FastAPI()
init_database()
conversation_service = ConversationService()
agent_run_manager = RunManager()
rate_limiter = InMemoryRateLimiter(
    int(os.getenv("API_RATE_LIMIT_PER_MINUTE", "60"))
)
# 匿名文件上传不应继承普通查询额度，单独限制每个 IP 每分钟最多 3 次。
guest_upload_rate_limiter = InMemoryRateLimiter(3)

configured_origins = [
    origin.strip()
    for origin in os.getenv("CORS_ALLOWED_ORIGINS", "").split(",")
    if origin.strip()
]


async def get_current_principal(request: Request, response: Response) -> Principal:
    """供 API 路由复用统一身份解析，并将认证异常转换为安全 HTTP 响应。"""
    try:
        return await resolve_principal(request, response)
    except AuthenticationError as error:
        raise HTTPException(
            status_code=error.status_code,
            detail=error.detail,
        ) from error


def require_admin(principal: Principal) -> None:
    """所有者身份只由已验证的 Principal 决定，绝不接受请求参数覆盖。"""
    if principal.role != "admin":
        raise HTTPException(status_code=403, detail="此操作仅限管理员")


def _allowed_origin(origin: str) -> bool:
    if origin in configured_origins:
        return True
    if configured_origins:
        return False
    # 没有正式配置时仅放行本机开发页面，不能允许任意网页向本机服务写入。
    return bool(re.fullmatch(r"http://(localhost|127\.0\.0\.1):\d{1,5}", origin))

app.add_middleware(
    CORSMiddleware,
    allow_origins=configured_origins,
    # 未配置正式域名时，仅允许本地 Vite 开发端口。
    allow_origin_regex=(
        None
        if configured_origins
        else r"^http://(localhost|127\.0\.0\.1):\d{1,5}$"
    ),
    allow_credentials=True,
    allow_methods=["GET", "POST", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type"],
)


@app.middleware("http")
async def protect_api(request: Request, call_next):
    """保留基础限流，并阻止跨站网页伪造写请求（CSRF）。"""
    if request.url.path.startswith("/api/") and request.url.path != "/api/health":
        client_host = request.client.host if request.client else "unknown"
        if not rate_limiter.allow(client_host):
            return JSONResponse(status_code=429, content={"detail": "请求过于频繁"})
        if request.method in {"POST", "PATCH", "DELETE"}:
            origin = request.headers.get("origin")
            if not origin or not _allowed_origin(origin):
                return JSONResponse(status_code=403, content={"detail": "请求来源不受允许"})
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
MAX_GUEST_KNOWLEDGE_BYTES = 50 * 1024 * 1024


def find_relevant_chunks(knowledge_base_id: str | None, query: str):
    if not knowledge_base_id:
        return []

    try:
        return semantic_search(knowledge_base_id, query)
    except RuntimeError as error:
        raise HTTPException(status_code=502, detail="知识库检索暂时不可用") from error


@app.post("/api/documents/search")
def search_document(
    request: SearchRequest,
    principal: Principal = Depends(get_current_principal),
):
    if principal.role == "guest" and request.knowledge_base_id:
        public_knowledge_base = get_public_knowledge_base(
            request.knowledge_base_id,
            get_public_knowledge_base_ids(),
            owner_id=principal.owner_id,
        )
        if public_knowledge_base is None:
            raise HTTPException(status_code=404, detail="知识库不存在")
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
    }


@app.get("/api/session")
def get_session(principal: Principal = Depends(get_current_principal)):
    """返回服务端身份和不扣次的当前访客额度。"""
    quota = None
    if principal.role == "guest":
        if not principal.guest_session_hash:
            raise HTTPException(status_code=503, detail="访客额度服务暂时不可用")
        status = get_guest_quota_status(
            principal.guest_session_hash, datetime.now(timezone.utc)
        )
        quota = {
            "remaining": status.remaining,
            "limit": status.limit,
            "reset_at": status.reset_at.isoformat(),
        }
    return {"role": principal.role, "quota": quota}


@app.post("/api/chat")
def chat(
    request: ChatRequest,
    http_request: Request,
    principal: Principal = Depends(get_current_principal),
):
    """兼容旧客户端；所有聊天请求统一进入 Agent 执行链。"""
    return _chat(request, principal, http_request)


def _chat(
    request: ChatRequest,
    principal: Principal,
    http_request: Request,
    *,
    on_token=None,
):
    message = request.message.strip()

    if not message:
        raise HTTPException(status_code=400, detail="消息不能为空")

    _validate_knowledge_access(request, principal)
    _reserve_guest_quota(principal, http_request)

    try:
        turn = conversation_service.chat(
            message=message,
            knowledge_base_id=request.knowledge_base_id,
            conversation_id=request.conversation_id,
            owner_id=principal.owner_id,
            allowed_tools=(
                GUEST_ALLOWED_TOOLS if principal.role == "guest" else None
            ),
            on_token=on_token,
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


def _validate_knowledge_access(request: ChatRequest, principal: Principal) -> None:
    """校验访客只能使用公开或自己拥有的知识库。"""
    if principal.role == "guest" and request.knowledge_base_id:
        public_knowledge_base = get_public_knowledge_base(
            request.knowledge_base_id,
            get_public_knowledge_base_ids(),
            owner_id=principal.owner_id,
        )
        if public_knowledge_base is None:
            raise HTTPException(status_code=404, detail="知识库不存在")


def _reserve_guest_quota(principal: Principal, http_request: Request) -> None:
    """创建新 Run 时预占一次额度；恢复和断线重连不会调用此函数。"""
    if principal.role != "guest":
        return
    if not principal.guest_session_hash:
        raise HTTPException(status_code=503, detail="访客额度服务暂时不可用")
    try:
        reserve_guest_ai_request(
            principal.guest_session_hash,
            hash_client_ip(http_request),
            datetime.now(timezone.utc),
        )
    except GuestQuotaExceeded as error:
        retry_seconds = max(
            1,
            int(
                (
                    error.retry_at.astimezone(timezone.utc)
                    - datetime.now(timezone.utc)
                ).total_seconds()
            ),
        )
        raise HTTPException(
            status_code=429,
            detail="访客 AI 请求额度已达上限，请在额度恢复后重试",
            headers={
                "Retry-After": str(retry_seconds),
                "X-Quota-Reset": error.retry_at.isoformat(),
            },
        ) from error


@app.post("/api/agent/chat")
def agent_chat(
    request: ChatRequest,
    http_request: Request,
    principal: Principal = Depends(get_current_principal),
):
    """Agent 模式允许模型自主选择经过注册和校验的工具。"""
    return _chat(request, principal, http_request)


@app.post("/api/agent/chat/stream")
def agent_chat_stream(
    request: ChatRequest,
    http_request: Request,
    principal: Principal = Depends(get_current_principal),
):
    """创建可暂停的 Run，并以 SSE 返回状态、片段和工作轨迹。"""
    task, worker_factory = _prepare_agent_run(request, principal, http_request)
    events = agent_run_manager.start(
        task["run_id"], principal.owner_id, worker_factory
    )
    return _run_event_response(events)


def _prepare_agent_run(request: ChatRequest, principal: Principal, http_request: Request):
    """校验并创建 queued Run；这里只预占一次访客额度。"""
    message = request.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="消息不能为空")
    _validate_knowledge_access(request, principal)
    _reserve_guest_quota(principal, http_request)
    try:
        conversation_id = conversation_service.resolve_conversation(
            request.conversation_id, owner_id=principal.owner_id
        )
        task = repository.create_agent_task(
            conversation_id,
            principal.owner_id,
            None,
            quota_reserved=principal.role == "guest",
        )
    except ConversationNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error

    def worker_factory(control, emit):
        def checkpoint_callback(state):
            repository.update_agent_task(
                task["run_id"],
                principal.owner_id,
                expected_statuses={"running", "pause_requested"},
                checkpoint=serialize_agent_state(state),
            )
            retrieval_debug = state.get("retrieval_debug") or {}
            safe_rag = {
                key: retrieval_debug[key]
                for key in ("top_k", "max_distance", "returned", "matches")
                if key in retrieval_debug
            }
            emit(
                "trace",
                {
                    "trace": {
                        "steps": state.get("steps", 0),
                        "tools": state.get("tool_traces", []),
                        "rag": safe_rag,
                    }
                },
            )

        def status_callback(status: str):
            if status == "paused":
                repository.update_agent_task(
                    task["run_id"],
                    principal.owner_id,
                    expected_statuses={"running", "pause_requested"},
                    status="paused",
                )
                emit("paused", {"status": "paused"})
            else:
                emit("status", {"status": status})

        turn = conversation_service.run_resumable_task(
            run_id=task["run_id"],
            message=message,
            knowledge_base_id=request.knowledge_base_id,
            conversation_id=task["conversation_id"],
            owner_id=principal.owner_id,
            allowed_tools=(
                GUEST_ALLOWED_TOOLS if principal.role == "guest" else None
            ),
            control=control,
            checkpoint_callback=checkpoint_callback,
            status_callback=status_callback,
            on_token=lambda token: emit("token", {"content": token}),
        )
        result = turn.agent
        if result.status == "completed":
            repository.update_agent_task(
                task["run_id"],
                principal.owner_id,
                expected_statuses={"running"},
                status="completed",
            )
            emit("trace", {"trace": turn.trace.to_dict()})
            emit("done", _turn_payload(task["run_id"], turn))
        elif result.status == "paused":
            # paused 事件已由 status_callback 发出，等待 resume 继续订阅。
            pass
        elif result.status == "stopped":
            emit("cancelled", {"status": "stopped", "detail": "任务已停止"})
        elif result.status == "timed_out":
            emit("timeout", {"status": "timed_out", "detail": "任务执行超时，请重试"})
        else:
            emit("error", {"status": 500, "code": "agent_failed", "detail": "Agent 执行失败，请检查后端日志"})

    return task, worker_factory


def _turn_payload(run_id: str, turn) -> dict:
    result = turn.agent
    return {
        "run_id": run_id,
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
            "tool_source": result.tool_source,
            "mcp_server": result.mcp_server,
        },
    }


def _run_event_response(events):
    def event_stream():
        while True:
            item = events.get()
            if item is None:
                break
            event, data = item
            yield format_sse(event, data)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-store",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


def _public_task(task: dict) -> dict:
    """不向浏览器返回检查点正文，只返回可操作的生命周期信息。"""
    return {
        key: task.get(key)
        for key in (
            "run_id",
            "conversation_id",
            "owner_id",
            "status",
            "answer_prefix",
            "error_code",
            "error_message",
            "quota_reserved",
            "created_at",
            "updated_at",
            "started_at",
            "finished_at",
        )
    }


@app.post("/api/agent/runs/{run_id}/pause")
def pause_agent_run(
    run_id: str,
    principal: Principal = Depends(get_current_principal),
):
    try:
        return _public_task(agent_run_manager.request_pause(run_id, principal.owner_id))
    except LookupError as error:
        raise HTTPException(status_code=404, detail="任务不存在") from error
    except ValueError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error


@app.post("/api/agent/runs/{run_id}/resume")
def resume_agent_run(
    run_id: str,
    principal: Principal = Depends(get_current_principal),
):
    try:
        return _run_event_response(
            agent_run_manager.resume(run_id, principal.owner_id)
        )
    except LookupError as error:
        raise HTTPException(status_code=404, detail="任务不存在或恢复信息已过期") from error
    except ValueError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error


@app.post("/api/agent/runs/{run_id}/cancel")
def cancel_agent_run(
    run_id: str,
    principal: Principal = Depends(get_current_principal),
):
    try:
        return _public_task(agent_run_manager.cancel(run_id, principal.owner_id))
    except LookupError as error:
        raise HTTPException(status_code=404, detail="任务不存在") from error


@app.get("/api/agent/runs/{run_id}")
def get_agent_run(
    run_id: str,
    principal: Principal = Depends(get_current_principal),
):
    task = repository.get_agent_task(run_id, principal.owner_id)
    if task is None:
        raise HTTPException(status_code=404, detail="任务不存在")
    return _public_task(task)


@app.get("/api/agent/runs")
def list_agent_runs(
    conversation_id: str | None = Query(default=None),
    principal: Principal = Depends(get_current_principal),
):
    return {
        "runs": [
            _public_task(task)
            for task in repository.list_agent_tasks(
                conversation_id, owner_id=principal.owner_id
            )
        ]
    }


@app.get("/api/conversations/{conversation_id}/messages")
def get_conversation_messages(
    conversation_id: str,
    limit: int = Query(default=100, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    principal: Principal = Depends(get_current_principal),
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
            owner_id=principal.owner_id,
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
    principal: Principal = Depends(get_current_principal),
):
    return {
        "conversations": conversation_service.list_conversations(
            limit=limit,
            offset=offset,
            owner_id=principal.owner_id,
        )
    }


@app.post("/api/conversations", status_code=201)
def create_conversation(
    request: ConversationCreateRequest,
    principal: Principal = Depends(get_current_principal),
):
    title = request.title.strip()
    if not title:
        raise HTTPException(status_code=400, detail="会话标题不能为空")
    return conversation_service.create_conversation(title, owner_id=principal.owner_id)


@app.patch("/api/conversations/{conversation_id}")
def rename_conversation(
    conversation_id: str,
    request: ConversationUpdateRequest,
    principal: Principal = Depends(get_current_principal),
):
    title = request.title.strip()
    if not title:
        raise HTTPException(status_code=400, detail="会话标题不能为空")
    try:
        return conversation_service.rename_conversation(
            conversation_id, title, owner_id=principal.owner_id
        )
    except ConversationNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error


@app.delete("/api/conversations/{conversation_id}", status_code=204)
def delete_conversation(
    conversation_id: str,
    principal: Principal = Depends(get_current_principal),
):
    try:
        conversation_service.delete_conversation(
            conversation_id, owner_id=principal.owner_id
        )
    except ConversationNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error


@app.post("/api/documents/upload")
async def upload_document(
    http_request: Request,
    files: list[UploadFile] = File(default=[]),
    file: UploadFile | None = File(default=None),
    knowledge_base_id: str | None = Form(default=None),
    principal: Principal = Depends(get_current_principal),
):
    """把一次选择的多份文档索引到同一个知识库，并兼容旧版单文件字段。"""
    if principal.role == "guest" and not guest_upload_rate_limiter.allow(
        hash_client_ip(http_request)
    ):
        raise HTTPException(
            status_code=429,
            detail="上传过于频繁，请稍后再试",
        )
    uploaded_files = [*files, *([file] if file is not None else [])]
    if not uploaded_files:
        raise HTTPException(status_code=400, detail="请选择至少一个文档")
    if len(uploaded_files) > MAX_FILES_PER_UPLOAD:
        raise HTTPException(status_code=400, detail="一次最多上传 10 个文档")

    is_new_knowledge_base = knowledge_base_id is None
    if knowledge_base_id is not None:
        if not knowledge_base_exists(knowledge_base_id):
            raise HTTPException(status_code=404, detail="要追加的知识库不存在")
        if principal.role == "guest" and not knowledge_base_is_owned_by(
            knowledge_base_id, principal.owner_id
        ):
            # 共享知识库也不能追加；游客上传内容只能进入自己的库。
            raise HTTPException(status_code=404, detail="要追加的知识库不存在")
    elif principal.role == "guest":
        base_count, _, _ = get_owner_knowledge_base_count(principal.owner_id)
        if base_count >= 5:
            raise HTTPException(
                status_code=413,
                detail="访客最多创建 5 个个人知识库，请先删除不再使用的知识库",
            )

    if principal.role == "guest":
        _, document_count, _ = get_owner_knowledge_base_count(principal.owner_id)
        if document_count + len(uploaded_files) > 50:
            raise HTTPException(
                status_code=413,
                detail="访客个人知识库最多保存 50 份文档，请先删除部分文档",
            )

    uploaded_documents: list[tuple[str, bytes]] = []
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
        uploaded_documents.append((filename, content))

    if principal.role == "guest":
        _, _, existing_bytes = get_owner_knowledge_base_count(principal.owner_id)
        if existing_bytes + total_size > MAX_GUEST_KNOWLEDGE_BYTES:
            raise HTTPException(
                status_code=413,
                detail="访客个人知识库累计最多 50 MB，请先删除部分文档",
            )

    # 先检查整批文件的字节数，再做 PDF OCR 或向量化，避免超额文件消耗处理资源。
    parsed_documents: list[tuple[str, str, int]] = []
    for filename, content in uploaded_documents:
        try:
            text = await run_in_threadpool(parse_document, filename, content)
        except DocumentParseError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        if not text.strip():
            raise HTTPException(
                status_code=400,
                detail=f"无法从 {filename} 识别有效文字；请确认扫描清晰、方向正确且页数不超过限制",
            )
        parsed_documents.append((filename, text, len(content)))

    knowledge_base_id = knowledge_base_id or uuid4().hex
    upload_batch = uuid4().hex
    document_metadata: list[dict[str, str | int]] = []

    def rollback_upload() -> None:
        if is_new_knowledge_base:
            delete_collection(knowledge_base_id)
        else:
            delete_chunks_by_upload_batch(knowledge_base_id, upload_batch)

    try:
        for filename, text, size_bytes in parsed_documents:
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
                    "size_bytes": size_bytes,
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
            register_knowledge_base(
                knowledge_base_id,
                document_metadata,
                owner_id=principal.owner_id,
            )
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
    principal: Principal = Depends(get_current_principal),
):
    if principal.role == "guest":
        knowledge_bases = list_public_knowledge_bases(
            get_public_knowledge_base_ids(),
            owner_id=principal.owner_id,
            limit=limit,
            offset=offset,
        )
        for knowledge_base in knowledge_bases:
            knowledge_base["can_edit"] = (
                knowledge_base["owner_id"] == principal.owner_id
            )
            knowledge_base.pop("owner_id", None)
        return {
            "knowledge_bases": knowledge_bases
        }
    knowledge_bases = list_knowledge_base_metadata(limit=limit, offset=offset)
    for knowledge_base in knowledge_bases:
        knowledge_base["can_edit"] = True
        knowledge_base.pop("owner_id", None)
    return {
        "knowledge_bases": knowledge_bases
    }


@app.delete("/api/knowledge-bases/{knowledge_base_id}", status_code=204)
def delete_knowledge_base(
    knowledge_base_id: str,
    principal: Principal = Depends(get_current_principal),
):
    if not knowledge_base_exists(knowledge_base_id) or (
        principal.role == "guest"
        and not knowledge_base_is_owned_by(knowledge_base_id, principal.owner_id)
    ):
        raise HTTPException(status_code=404, detail="知识库不存在")
    delete_collection(knowledge_base_id)
    delete_knowledge_base_metadata(knowledge_base_id)


@app.delete("/api/knowledge-bases/{knowledge_base_id}/documents/{document_id}")
def delete_knowledge_document(
    knowledge_base_id: str,
    document_id: int,
    principal: Principal = Depends(get_current_principal),
):
    """只删除选中的文档及其向量，不影响同一知识库中的其他文档。"""
    if principal.role == "guest" and not knowledge_base_is_owned_by(
        knowledge_base_id, principal.owner_id
    ):
        raise HTTPException(status_code=404, detail="文档不存在")
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
    knowledge_base = get_knowledge_base(knowledge_base_id)
    if knowledge_base is None:
        raise HTTPException(status_code=404, detail="知识库不存在")
    return {
        "documents": knowledge_base["documents"],
        "chunks": knowledge_base["chunk_count"],
    }
