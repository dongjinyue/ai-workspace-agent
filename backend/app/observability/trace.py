from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class RequestTrace:
    """可公开给开发页面的执行元数据，不包含模型思维过程或敏感正文。"""

    # request_id 标识一次请求；conversation_id 属于会话，因此不在这里混用。
    request_id: str
    started_at: str
    duration_ms: float
    steps: int
    skill: str | None
    tools: list[dict]
    rag: dict[str, bool | int]
    llm_calls: int
    llm_duration_ms: float

    def to_dict(self) -> dict:
        """转换为 FastAPI 可以直接序列化的普通字典。"""
        return asdict(self)
