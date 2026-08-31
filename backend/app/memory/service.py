import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from time import perf_counter
from uuid import uuid4

from app.agent.service import AgentResult, run_agent
from app.memory import repository
from app.observability.trace import RequestTrace


HISTORY_WINDOW = 20
logger = logging.getLogger(__name__)


class ConversationNotFoundError(LookupError):
    """客户端提交了不存在的 Conversation ID（会话标识符）。"""


@dataclass(frozen=True)
class ConversationTurnResult:
    conversation_id: str
    history_messages: int
    agent: AgentResult
    trace: RequestTrace


class ConversationService:
    def list_conversations(self) -> list[dict[str, str | int]]:
        return repository.list_conversations()

    def create_conversation(self, title: str = "新会话") -> dict[str, str]:
        conversation_id = repository.create_conversation(title)
        return {"id": conversation_id, "title": title}

    def rename_conversation(self, conversation_id: str, title: str) -> dict[str, str]:
        if not repository.rename_conversation(conversation_id, title):
            raise ConversationNotFoundError("会话不存在")
        return {"id": conversation_id, "title": title}

    def delete_conversation(self, conversation_id: str) -> None:
        if not repository.delete_conversation(conversation_id):
            raise ConversationNotFoundError("会话不存在")

    def resolve_conversation(self, conversation_id: str | None) -> str:
        if conversation_id is None:
            return repository.create_conversation()
        if not repository.conversation_exists(conversation_id):
            raise ConversationNotFoundError("会话不存在")
        return conversation_id

    def chat(
        self,
        *,
        message: str,
        knowledge_base_id: str | None,
        conversation_id: str | None,
    ) -> ConversationTurnResult:
        # 同一 Conversation（会话）可有多次请求，每次必须有独立 request_id。
        request_id = uuid4().hex
        started_at = datetime.now(timezone.utc).isoformat()
        request_started = perf_counter()
        resolved_id: str | None = None
        try:
            resolved_id = self.resolve_conversation(conversation_id)

            # 先保存，再读取。当前用户消息因此只会进入 Agent 上下文一次。
            repository.save_message(resolved_id, "user", message)
            repository.use_first_message_as_title(resolved_id, message)
            history = repository.get_messages(resolved_id, limit=HISTORY_WINDOW)
            agent_result = run_agent(
                message=message,
                knowledge_base_id=knowledge_base_id,
                conversation_id=resolved_id,
                history_messages=[
                    {"role": item["role"], "content": item["content"]}
                    for item in history
                ],
            )
            repository.save_message(resolved_id, "assistant", agent_result.answer)
        except Exception as error:
            # 日志只记录不可逆推出正文的标识和错误类型，不记录用户消息或密钥。
            logger.error(
                "Agent request failed request_id=%s conversation_id=%s error_type=%s",
                request_id,
                resolved_id,
                type(error).__name__,
            )
            raise

        duration_ms = round((perf_counter() - request_started) * 1000, 3)
        # 只组装可安全公开的执行元数据，不包含提示词和模型内部思维过程。
        trace = RequestTrace(
            request_id=request_id,
            started_at=started_at,
            duration_ms=duration_ms,
            steps=agent_result.steps,
            skill=agent_result.active_skill,
            tools=agent_result.tool_traces or [],
            rag={
                "hit": agent_result.matched_chunks > 0,
                "results": agent_result.matched_chunks,
            },
            llm_calls=agent_result.llm_calls,
            llm_duration_ms=agent_result.llm_duration_ms,
        )
        logger.info(
            "Agent request completed request_id=%s conversation_id=%s duration_ms=%.3f",
            request_id,
            resolved_id,
            duration_ms,
        )
        return ConversationTurnResult(
            conversation_id=resolved_id,
            history_messages=len(history),
            agent=agent_result,
            trace=trace,
        )

    def get_history(self, conversation_id: str) -> list[dict[str, str]]:
        if not repository.conversation_exists(conversation_id):
            raise ConversationNotFoundError("会话不存在")
        return repository.get_messages(conversation_id)
