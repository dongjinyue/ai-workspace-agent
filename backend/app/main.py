import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from pydantic import BaseModel, Field


BACKEND_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BACKEND_DIR / ".env")

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


@app.get("/")
def home():
    return {"message": "AI Agent Backend Running"}


@app.get("/api/health")
def health_check():
    return {"status": "ok", "message": "Hello from AI Agent Backend"}


@app.post("/api/chat")
def chat(request: ChatRequest):
    api_key = os.getenv("DASHSCOPE_API_KEY")
    model = os.getenv("QWEN_MODEL", "qwen-max")
    base_url = os.getenv(
        "QWEN_BASE_URL",
        "https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    message = request.message.strip()

    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="服务器没有配置 DASHSCOPE_API_KEY",
        )

    if not message:
        raise HTTPException(status_code=400, detail="消息不能为空")

    try:
        client = OpenAI(api_key=api_key, base_url=base_url)
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "你是 AI Workspace Agent，"
                        "请使用清晰、简洁的中文回答用户。"
                    ),
                },
                {"role": "user", "content": message},
            ],
        )

        answer = response.choices[0].message.content
        if not answer:
            raise ValueError("模型返回了空内容")

        return {"answer": answer}

    except Exception as error:
        print(f"调用模型失败：{type(error).__name__}: {error}")
        raise HTTPException(
            status_code=502,
            detail="调用大模型失败，请稍后重试",
        ) from error
