from pathlib import Path
from uuid import uuid4

from dotenv import load_dotenv

BACKEND_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BACKEND_DIR / ".env")


from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from app.agent.service import run_agent
from app.rag.service import index_document, semantic_search

app = FastAPI()

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
        result = run_agent(
            message=message,
            knowledge_base_id=request.knowledge_base_id,
        )
        return {
            "answer": result.answer,
            "matched_chunks": result.matched_chunks,
            "llm_called": result.llm_called,
            "agent": {
                "tool_called": result.tool_called,
                "tool_name": result.tool_name,
                "steps": result.steps,
            },
        }

    except Exception as error:
        print(f"Agent 执行失败：{type(error).__name__}: {error}")
        raise HTTPException(
            status_code=502,
            detail="Agent 执行失败，请检查模型、工具参数和后端日志",
        ) from error


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
