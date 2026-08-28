from pathlib import Path
from uuid import uuid4

from dotenv import load_dotenv

BACKEND_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BACKEND_DIR / ".env")


from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from app.memory.database import init_database
from app.memory.service import ConversationNotFoundError, ConversationService
from app.observability import configure_logging
from app.rag.service import index_document, semantic_search
from app.security import PromptInjectionError

configure_logging()
app = FastAPI()
init_database()
conversation_service = ConversationService()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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


MAX_FILE_SIZE = 1024 * 1024


def find_relevant_chunks(knowledge_base_id: str | None, query: str):
    if not knowledge_base_id:
        return []

    try:
        return semantic_search(knowledge_base_id, query)
    except RuntimeError as error:
        raise HTTPException(status_code=502, detail=str(error)) from error


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
    return {"status": "ok", "message": "Hello from AI Agent Backend"}


@app.post("/api/chat")
def chat(request: ChatRequest):
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
    except Exception as error:
        raise HTTPException(
            status_code=502,
            detail="Agent 执行失败，请检查模型、工具参数和后端日志",
        ) from error


@app.get("/api/conversations/{conversation_id}/messages")
def get_conversation_messages(conversation_id: str):
    if len(conversation_id) != 32 or any(
        character not in "0123456789abcdef" for character in conversation_id
    ):
        raise HTTPException(status_code=404, detail="会话不存在")
    try:
        messages = conversation_service.get_history(conversation_id)
    except ConversationNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    return {
        "conversation_id": conversation_id,
        "messages": [
            {"role": item["role"], "content": item["content"]}
            for item in messages
        ],
    }


@app.post("/api/documents/upload")
async def upload_document(file: UploadFile = File(...)):
    filename = file.filename or ""
    if not filename.lower().endswith(".txt"):
        raise HTTPException(status_code=400, detail="目前只支持 TXT 文件")

    content = await file.read(MAX_FILE_SIZE + 1)
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="TXT 文件不能超过 1 MB")

    if not content:
        raise HTTPException(status_code=400, detail="上传的文件为空")

    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError as error:
        raise HTTPException(
            status_code=400,
            detail="TXT 文件必须使用 UTF-8 编码",
        ) from error

    knowledge_base_id = uuid4().hex
    try:
        chunk_count = await run_in_threadpool(
            index_document,
            knowledge_base_id,
            text,
        )
    except PromptInjectionError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except RuntimeError as error:
        raise HTTPException(status_code=502, detail=str(error)) from error

    if chunk_count == 0:
        raise HTTPException(status_code=400, detail="文件中没有可用的文本内容")

    return {
        "success": True,
        "filename": filename,
        "chunks": chunk_count,
        "knowledge_base_id": knowledge_base_id,
    }
