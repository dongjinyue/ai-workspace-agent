"""注册表到 LangChain 工具的适配；可信资源上下文不作为模型参数。"""
from typing import Any

from jsonschema import validate, ValidationError
from langchain_core.tools import BaseTool, StructuredTool
from langchain_core.runnables import RunnableConfig
from langchain_core.runnables.config import get_callback_manager_for_config
from langsmith import tracing_context

from app.agent.registry import TOOLS, ALLOWED_MCP_TOOLS
from app.mcp.client import MCPClient, MCPClientError


def build_tool(tool_name: str, *, knowledge_base_id: str | None,
               allowed_tools: frozenset[str] | tuple[str, ...] | None) -> BaseTool:
    registration = TOOLS.get(tool_name)
    if registration is None or (allowed_tools is not None and tool_name not in allowed_tools):
        raise ValueError(f"当前技能不允许调用工具：{tool_name}")
    if registration.source == "mcp" and (tool_name not in ALLOWED_MCP_TOOLS or registration.input_schema is None):
        raise ValueError("不允许调用 MCP 工具")

    def execute(config: RunnableConfig, **arguments: Any) -> dict:
        # 调用时再次校验，避免构造后权限或动态注册发生变化。
        if TOOLS.get(tool_name) is not registration or (allowed_tools is not None and tool_name not in allowed_tools):
            raise ValueError("工具注册或权限已变化")
        if registration.source == "mcp":
            try:
                validate(instance=arguments, schema=registration.input_schema)
            except ValidationError as error:
                raise ValueError("MCP 工具参数不符合 JSON Schema") from error
            try:
                value = MCPClient().call_tool_sync(tool_name, arguments)
            except MCPClientError as error:
                raise RuntimeError("MCP 工具当前不可用") from error
            return {"source": "mcp", "server": registration.server, "untrusted_data": value}
        if registration.arguments_model is None or registration.handler is None:
            raise RuntimeError("本地工具注册不完整")
        validated = registration.arguments_model.model_validate(arguments).model_dump()
        if registration.needs_knowledge_base:
            validated["knowledge_base_id"] = knowledge_base_id
        if tool_name != "search_knowledge_base":
            return registration.handler(**validated)
        # 检索作为工具的子运行；追踪回调只上传命中数量，不上传查询或文档。
        callbacks = get_callback_manager_for_config(config)
        retrieval = callbacks.on_retriever_start({"name": tool_name}, validated["query"])
        try:
            result = registration.handler(**validated)
        except Exception as error:
            retrieval.on_retriever_error(error)
            raise
        retrieval.on_retriever_end(result.get("chunks", []))
        return result

    schema = registration.input_schema if registration.source == "mcp" else registration.arguments_model
    return StructuredTool(name=tool_name, description=registration.schema["function"]["description"],
                          args_schema=schema, func=execute)


def invoke_registered_tool(tool_name: str, arguments: dict[str, Any], knowledge_base_id: str | None, *,
                           allowed_tools: frozenset[str] | tuple[str, ...] | None = None,
                           config: RunnableConfig | None = None) -> dict:
    if not isinstance(arguments, dict):
        raise ValueError("工具参数必须是 JSON 对象")
    # 旧的直接调用入口也必须禁用环境触发的默认正文追踪。
    # 显式传入的元数据回调仍通过 config 正常传播。
    with tracing_context(enabled=False):
        return build_tool(tool_name, knowledge_base_id=knowledge_base_id,
                          allowed_tools=allowed_tools).invoke(arguments, config=config)
