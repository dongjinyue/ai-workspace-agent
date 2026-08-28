import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Callable
from unittest.mock import patch
from uuid import uuid4

os.environ.setdefault(
    "CHROMA_PATH",
    str(Path(tempfile.gettempdir()) / "ai-workspace-agent-eval-chroma"),
)

from app.security import PromptInjectionError


CASES_PATH = Path(__file__).with_name("cases.json")


@dataclass(frozen=True)
class EvaluationResult:
    cases: int
    tool_correct: int
    tool_total: int
    skill_correct: int
    skill_total: int
    no_hit_correct: int
    no_hit_total: int
    retrieval_correct: int
    retrieval_total: int
    security_correct: int
    security_total: int

    @staticmethod
    def _percent(correct: int, total: int) -> float:
        return round(correct / total * 100, 2) if total else 100.0

    def summary(self) -> str:
        return "\n".join(
            [
                "Evaluation Summary",
                "Execution: production LangGraph + production tools + Mock LLM",
                f"Cases: {self.cases}",
                f"Tool Routing: {self.tool_correct} / {self.tool_total} "
                f"({self._percent(self.tool_correct, self.tool_total)}%)",
                f"Skill Routing: {self.skill_correct} / {self.skill_total} "
                f"({self._percent(self.skill_correct, self.skill_total)}%)",
                f"RAG No-hit: {self.no_hit_correct} / {self.no_hit_total} "
                f"({self._percent(self.no_hit_correct, self.no_hit_total)}%)",
                f"Retrieval Facts: {self.retrieval_correct} / {self.retrieval_total} "
                f"({self._percent(self.retrieval_correct, self.retrieval_total)}%)",
                f"Injection Safety: {self.security_correct} / {self.security_total} "
                f"({self._percent(self.security_correct, self.security_total)}%)",
            ]
        )


def load_cases() -> list[dict]:
    return json.loads(CASES_PATH.read_text(encoding="utf-8"))


def evaluate(
    predictor: Callable[[dict], dict], cases: list[dict] | None = None
) -> EvaluationResult:
    selected_cases = cases or load_cases()
    tool_correct = skill_correct = no_hit_correct = no_hit_total = 0
    retrieval_correct = retrieval_total = security_correct = security_total = 0
    for case in selected_cases:
        observed = predictor(case)
        tool_correct += observed.get("tool") == case.get("expected_tool")
        skill_correct += observed.get("skill") == case.get("expected_skill")
        if "expected_no_hit" in case:
            no_hit_total += 1
            no_hit_correct += (
                observed.get("no_hit") == case["expected_no_hit"]
                and observed.get("llm_calls_after_retrieval", 0) == 0
            )
        if "expected_contains" in case:
            retrieval_total += 1
            retrieval_correct += all(
                text in observed.get("retrieved_text", "")
                for text in case["expected_contains"]
            )
        if "expected_secure" in case:
            security_total += 1
            security_correct += observed.get("secure", False) and not any(
                text in observed.get("answer", "")
                for text in case.get("forbidden_contains", [])
            )
    return EvaluationResult(
        cases=len(selected_cases),
        tool_correct=tool_correct,
        tool_total=len(selected_cases),
        skill_correct=skill_correct,
        skill_total=len(selected_cases),
        no_hit_correct=no_hit_correct,
        no_hit_total=no_hit_total,
        retrieval_correct=retrieval_correct,
        retrieval_total=retrieval_total,
        security_correct=security_correct,
        security_total=security_total,
    )


class MockLLMCompletions:
    """模拟模型选择工具，生产 LangGraph 和 Tool 层仍真实执行。"""

    def __init__(self, question: str) -> None:
        self.question = question

    def create(self, **kwargs):
        last_message = kwargs["messages"][-1]
        role = last_message.get("role") if isinstance(last_message, dict) else None
        if role == "system":
            tool_message = kwargs["messages"][-2]
            result = json.loads(tool_message["content"])
            if "chunks" in result:
                content = result["chunks"][0] if result["chunks"] else ""
            else:
                content = str(result.get("result", result.get("untrusted_data", "")))
            message = SimpleNamespace(content=content, tool_calls=None)
            return SimpleNamespace(choices=[SimpleNamespace(message=message)])

        if any(word in self.question for word in ("乘", "除", "加", "减")):
            name = "calculator"
            arguments = '{"a": 137, "b": 29, "operation": "multiply"}'
        elif any(
            word in self.question for word in ("公司", "政策", "补贴", "退款")
        ):
            name = "search_knowledge_base"
            arguments = json.dumps({"query": self.question}, ensure_ascii=False)
        else:
            message = SimpleNamespace(content="你好", tool_calls=None)
            return SimpleNamespace(choices=[SimpleNamespace(message=message)])

        tool_call = SimpleNamespace(
            id="mock-tool-call",
            function=SimpleNamespace(name=name, arguments=arguments),
        )
        message = SimpleNamespace(content=None, tool_calls=[tool_call])
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def _deterministic_embeddings(
    texts: list[str], *, text_type: str
) -> list[list[float]]:
    """提供稳定向量；相似度筛选仍由真实 Chroma 完成。"""
    return [
        [0.0, 1.0] if "火星旅游" in text else [1.0, 0.0]
        for text in texts
    ]


def production_path_predictor(case: dict) -> dict:
    """仅 Mock 外部模型与向量，执行生产 Agent、工具和检索路径。"""
    from app.agent.service import run_agent
    from app.rag.service import index_document

    knowledge_base_id = f"eval_{uuid4().hex}"
    secure = True
    document = case.get("document")
    if document:
        try:
            with patch("app.rag.service.embed_texts", _deterministic_embeddings):
                index_document(knowledge_base_id, document)
            secure = False
        except PromptInjectionError:
            secure = True
    elif case.get("expected_tool") == "search_knowledge_base" and not case.get(
        "expected_no_hit"
    ):
        with patch("app.rag.service.embed_texts", _deterministic_embeddings):
            index_document(knowledge_base_id, "退款需要订单号和购买凭证。")

    fake_client = SimpleNamespace(
        chat=SimpleNamespace(completions=MockLLMCompletions(case["question"]))
    )
    with (
        patch("app.agent.nodes._client", return_value=fake_client),
        patch("app.rag.service.embed_texts", _deterministic_embeddings),
    ):
        result = run_agent(case["question"], knowledge_base_id)

    no_hit = result.tool_name == "search_knowledge_base" and result.matched_chunks == 0
    return {
        "tool": result.tool_name,
        "skill": result.active_skill,
        "no_hit": no_hit,
        "llm_calls_after_retrieval": 0 if no_hit and result.llm_calls == 1 else 1,
        "retrieved_text": result.answer,
        "secure": secure,
        "answer": result.answer,
    }


if __name__ == "__main__":
    print("Production Path Mock Evaluation（生产路径模拟评测）")
    print(evaluate(production_path_predictor).summary())
