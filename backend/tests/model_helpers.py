"""迁移既有决策测试的脚本响应，生产代码不携带旧 SDK 兼容分支。"""
from langchain_core.messages import AIMessageChunk, convert_to_openai_messages
from app.agent.messages import normalize_messages


class ScriptedChatModel:
    def __init__(self, client):
        self.create = client.chat.completions.create
        self.options = {"tools": [], "tool_choice": "none"}

    def bind_tools(self, tools, **kwargs):
        self.options = {"tools": tools, **kwargs}
        return self

    def invoke(self, messages, config=None):
        response = self.create(messages=convert_to_openai_messages(messages), **self.options)
        value = response.choices[0].message
        data = vars(value).copy()
        data.setdefault("role", "assistant")
        return normalize_messages([data])[0]

    def stream(self, messages, config=None):
        chunks = self.create(messages=convert_to_openai_messages(messages), stream=True, **self.options)
        try:
            for chunk in chunks:
                choices = getattr(chunk, "choices", [])
                if choices:
                    yield AIMessageChunk(content=getattr(choices[0].delta, "content", None) or "")
        finally:
            close = getattr(chunks, "close", None)
            if close:
                close()


def as_chat_model(client):
    return ScriptedChatModel(client)
