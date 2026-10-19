"""LangSmith 只接收允许的执行元数据，自动正文追踪始终关闭。"""
import logging
import math
import os
import re
from contextlib import contextmanager
from datetime import datetime, timezone
from functools import wraps
from time import perf_counter
from typing import Any
from uuid import uuid4
from urllib3.util import Retry

from langchain_core.callbacks import BaseCallbackHandler
from langsmith import Client, tracing_context

logger = logging.getLogger(__name__)
_NAMES = frozenset({"LangGraph", "agent", "tools", "ChatOpenAI", "calculator", "search_knowledge_base",
                    "get_knowledge_base_info", "get_current_time", "calculate_text_stats"})
_COUNTS = frozenset({"steps", "llm_calls", "matched_chunks", "duration_ms", "input_tokens", "output_tokens", "total_tokens"})


def sanitize_trace_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """只重建已知字段，绝不把 kwargs、异常或模型配置整体透传。"""
    result = {key: payload[key] for key in ("id", "run_id", "parent_run_id", "trace_id", "dotted_order", "start_time", "end_time") if key in payload}
    if "name" in payload:
        result["name"] = payload["name"] if payload["name"] in _NAMES else "operation"
    if payload.get("run_type") in {"chain", "llm", "tool", "retriever"}:
        result["run_type"] = payload["run_type"]
    if "inputs" in payload:
        result["inputs"] = {}
    if "outputs" in payload:
        result["outputs"] = {}
    if payload.get("error"):
        result["error"] = "execution_failed"
    metadata = payload.get("extra", {}).get("metadata", {})
    safe = {key: value for key, value in metadata.items() if key in _COUNTS
            and type(value) in (int, float) and math.isfinite(value) and value >= 0}
    correlation = metadata.get("correlation_id", "")
    if isinstance(correlation, str) and re.fullmatch(r"[a-f0-9]{32}", correlation):
        safe["correlation_id"] = correlation
    result["extra"] = {"metadata": safe}
    return result


class MetadataTraceHandler(BaseCallbackHandler):
    """使用标准回调建立父子追踪，但在 SDK 边界之前丢弃所有正文。"""
    run_inline = True
    raise_error = False

    def __init__(self, client: Client, correlation_id: str, project: str):
        self.client, self.correlation_id, self.project = client, correlation_id, project
        self.runs = {}
        self.unavailable = False

    def _send(self, method, payload):
        if self.unavailable:
            return
        try:
            safe = sanitize_trace_payload(payload)
            if method == "create_run":
                safe["project_name"] = self.project
            getattr(self.client, method)(**safe)
        except Exception:
            self.unavailable = True
            # 追踪失败不能覆盖模型异常，也不打印含正文的异常消息。
            logger.warning("LangSmith metadata transport unavailable")

    def _start(self, run_type, *, run_id, parent_run_id=None, name=None, **kwargs):
        now = datetime.now(timezone.utc)
        parent = self.runs.get(parent_run_id)
        order = now.strftime("%Y%m%dT%H%M%S%fZ") + str(run_id)
        trace_id = parent[0] if parent else run_id
        dotted = parent[1] + "." + order if parent else order
        self.runs[run_id] = (trace_id, dotted, perf_counter())
        self._send("create_run", {"id": run_id, "name": name or "operation", "run_type": run_type,
                   "inputs": {}, "start_time": now, "parent_run_id": parent_run_id,
                   "trace_id": trace_id, "dotted_order": dotted,
                   "extra": {"metadata": {"correlation_id": self.correlation_id}}})

    def _end(self, *, run_id, error=None, counts=None, **kwargs):
        info = self.runs.get(run_id)
        metadata = {"correlation_id": self.correlation_id, **(counts or {})}
        if info:
            metadata["duration_ms"] = (perf_counter() - info[2]) * 1000
        self._send("update_run", {"run_id": run_id, "outputs": {}, "error": error,
                   "end_time": datetime.now(timezone.utc), "extra": {"metadata": metadata}})

    def on_chain_start(self, serialized, inputs, **kwargs):
        self._start("chain", **kwargs)

    def on_chain_end(self, outputs, **kwargs):
        counts = {k: outputs[k] for k in _COUNTS if isinstance(outputs, dict) and k in outputs}
        self._end(counts=counts, **kwargs)

    def on_chain_error(self, error, **kwargs):
        self._end(error=True, **kwargs)

    def on_chat_model_start(self, serialized, messages, **kwargs):
        self._start("llm", name="ChatOpenAI", **{k: v for k, v in kwargs.items() if k != "name"})

    def on_llm_end(self, response, **kwargs):
        counts = {}
        for group in response.generations:
            for generation in group:
                counts.update(getattr(getattr(generation, "message", None), "usage_metadata", None) or {})
        self._end(counts=counts, **kwargs)

    def on_llm_error(self, error, **kwargs):
        self._end(error=True, **kwargs)

    def on_tool_start(self, serialized, input_str, **kwargs):
        self._start("tool", name=serialized.get("name"), **{k: v for k, v in kwargs.items() if k != "name"})

    def on_tool_end(self, output, **kwargs):
        self._end(**kwargs)

    def on_tool_error(self, error, **kwargs):
        self._end(error=True, **kwargs)

    def on_retriever_start(self, serialized, query, **kwargs):
        self._start("retriever", name="search_knowledge_base", **{k: v for k, v in kwargs.items() if k != "name"})

    def on_retriever_end(self, documents, **kwargs):
        self._end(counts={"matched_chunks": len(documents)}, **kwargs)

    def on_retriever_error(self, error, **kwargs):
        self._end(error=True, **kwargs)


@contextmanager
def agent_trace_config(correlation_id: str):
    """在线程内显式建立上下文，禁止环境变量另行启用未脱敏追踪。"""
    with tracing_context(enabled=False):
        client = None
        config = {}
        if os.getenv("LANGSMITH_TRACING", "false").lower() == "true" and os.getenv("LANGSMITH_API_KEY"):
            try:
                client = Client(api_key=os.environ["LANGSMITH_API_KEY"],
                                api_url=os.getenv("LANGSMITH_ENDPOINT") or "https://api.smith.langchain.com",
                                auto_batch_tracing=False, hide_inputs=True, hide_outputs=True,
                                hide_metadata=lambda value: sanitize_trace_payload({"extra": {"metadata": value}})["extra"]["metadata"],
                                omit_traced_runtime_info=True, tracing_mode="langsmith",
                                retry_config=Retry(total=0), timeout_ms=1000)
                correlation = correlation_id if re.fullmatch(r"[a-f0-9]{32}", correlation_id) else uuid4().hex
                config = {"callbacks": [MetadataTraceHandler(client, correlation, os.getenv("LANGSMITH_PROJECT", "ai-workspace-agent"))]}
            except Exception:
                logger.warning("LangSmith metadata tracing unavailable")
        try:
            yield config
        finally:
            if client:
                try:
                    client.close()
                except Exception:
                    logger.warning("LangSmith metadata tracing shutdown unavailable")


def traced_agent(function):
    """直接入口与后台执行器共用追踪边界；配置只存在运行态。"""
    @wraps(function)
    def wrapped(*args, **kwargs):
        correlation_id = kwargs.pop("trace_correlation_id", None) or uuid4().hex
        with agent_trace_config(correlation_id) as config:
            kwargs["config"] = config
            return function(*args, **kwargs)
    return wrapped
