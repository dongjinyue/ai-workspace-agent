# LangChain、LangGraph 与 LangSmith 迁移实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** 将现有生产调用路径迁移到 LangChain（大模型应用框架）的模型、消息和工具接口，保留 LangGraph（流程编排框架），接入可关闭且只上传安全元数据的 LangSmith（执行追踪平台）。

**Architecture:** 保留两个节点的状态图、自建 SQLite（轻量数据库）检查点和现有会话服务。新增消息转换、工具适配、流式响应处理、统一结果整理与追踪模块，普通和可恢复入口共用这些模块。MCP（模型上下文协议）继续真实跨进程调用。

**Tech Stack:** Python（编程语言）、LangChain、langchain-openai 模型适配包、LangGraph、LangSmith、Pydantic（参数校验库）、Pytest（测试框架）、HTTPX（网络请求库）。

**Spec:** `docs/superpowers/specs/2026-10-19-langchain-langgraph-langsmith-design.md`，用户已回复“设计通过”。

## Global Constraints

- “继续使用 MAX_STEPS=5，模型节点与工具节点各计一步；每个工具节点只执行一个工具。”
- “不为流式分类额外发起未计数的模型请求。”
- “旧检查点须能够继续读取；未知版本安全报错，不猜测恢复格式。不修改已有数据库表结构。”
- “工具权限取当前请求允许列表与任务技能允许列表的交集，在模型可见列表和实际执行入口各检查一次。”
- “MCP 保留宿主允许列表和真实客户端调用，不能直接导入服务端函数。”
- “本次只提供元数据追踪模式”；`LANGSMITH_TRACING` 默认 false，默认关闭时不得创建外部追踪请求。
- “前端接口、事件名称和结果字段维持兼容”；不迁移数据库、不重建索引、不替换嵌入模型、不部署外部服务。
- 所有数据库和向量存储测试使用临时路径，不读取真实 `.env`、不修改正式数据；生产代码关键逻辑添加中文注释。
- 实施前使用 `superpowers:using-git-worktrees` 检查隔离方案；按用户选择的执行方法使用对应执行技能。

## Review Focus

1. 模型先输出文本再输出工具分片：不得把中间文本或参数当最终答案展示；任务 4 测试固定行为。
2. 工具参数尚未完整时暂停，随后恢复：不执行半成品调用、不重置预算；任务 4、6 验证。
3. 新旧检查点混用以及未知版本：旧格式可读、未知格式明确拒绝、不重复执行已完成工具；任务 2、6 验证。
4. 自动追踪的元数据、异常、后台线程上下文：即使 inputs/outputs（输入输出）隐藏，也不能通过其他字段泄露正文；任务 7 检查完整出站载荷。
5. 政策摘要包含无原文依据字段且经过暂停：完成事件、数据库和同步结果使用相同校验答案；任务 5、6、8 验证。

## 执行约定与文件边界

所有测试命令在 `backend` 目录执行。`.\venv\Scripts\python.exe -m pytest` 使用项目虚拟环境运行测试，`-q` 简化输出；特定测试文件用于先失败、实现、再通过。命令成功须有真实测试通过输出，失败必须由预期行为缺失造成，不能以导入或环境错误冒充失败测试。

依赖安装只影响实施工作区对应的 Python 环境；在安装前核对解释器路径、现有版本和包解析结果。不得直接更新所有依赖。Git（版本管理）提交只暂存对应任务文件，不包含 `.env`、运行数据库、索引或缓存。

新增模块：

| 文件 | 唯一职责 |
| --- | --- |
| `backend/app/agent/messages.py` | 新旧消息转标准消息、提取纯文本、检查点消息编码解码。 |
| `backend/app/agent/tool_adapter.py` | 从注册定义构造 LangChain 工具，安全验证后执行。 |
| `backend/app/agent/model_response.py` | 累积模型流、识别完整工具调用、处理流内控制信号。 |
| `backend/app/agent/result.py` | 公共结果类型、最终回答和政策摘要整理。 |
| `backend/app/agent/tracing.py` | 追踪配置、关联上下文、载荷允许列表与安全传输。 |

修改既有模块：`llm.py`、`nodes.py`、`state.py`、`registry.py`、`service.py`、`runner.py`、`checkpoint.py`、`memory/service.py`；`main.py` 和 `run_manager.py` 仅在上下文传递需要时修改。`graph.py` 保留已有节点和路由结构。

## Task 1: 接入 LangChain 模型并固定兼容边界

**Files:** 修改 `backend/app/agent/llm.py`、`backend/requirements.txt`、`backend/tests/unit/test_llm_config.py`；新增 `backend/tests/unit/test_langchain_model.py`。

**Interfaces:** 产出 `create_chat_model(*, http_client: httpx.Client | None = None) -> ChatOpenAI`；保留 `model_name() -> str` 与 `translate_model_error(error: Exception) -> None`。`http_client` 仅用于注入测试传输；模型从现有通义千问环境配置读取参数。

- [x] **Step 1:** 检查 Python 版本和现有包元数据，以依赖解析预演确定 `langchain`、`langchain-openai` 与现有核心包的兼容版本；记录需要改变的直接和间接依赖。
- [x] **Step 2:** 安装已解析的兼容组合，更新精确版本；运行 `.\venv\Scripts\python.exe -m pip check`，预期报告没有依赖冲突。若解析不可行，先修正组合，不写替代模型分支。
- [x] **Step 3:** 编写测试 `test_chat_model_uses_original_qwen_config`，断言配置模型名、接口地址、密钥、45 秒默认超时、2 次重试；缺失密钥和无效超时仍返回既有安全错误。
- [x] **Step 4:** 用 `httpx.MockTransport`（模拟网络传输）编写 `test_bind_tools_uses_chat_completions_and_disables_thinking`：请求路径为 `/chat/completions`，有 calculator 工具，`parallel_tool_calls` 为 false，扩展参数 `enable_thinking` 为 false；响应解析为 AIMessage（助手消息）。覆盖无用量字段和 401、429、超时异常。
- [x] **Step 5:** 运行 `.\venv\Scripts\python.exe -m pytest tests/unit/test_llm_config.py tests/unit/test_langchain_model.py -q`，确认新接口行为尚未实现导致失败。
- [x] **Step 6:** 实现模型工厂，显式设置 `use_responses_api=False`，扩展参数放 `extra_body`；不记录密钥，不替换真实配置。旧工厂只暂时供尚未迁移的节点使用，并在任务 4 删除。
- [x] **Step 7:** 重跑 Step 5，预期全部通过；提交上述四个文件，提交说明 `feat: add LangChain Qwen model adapter`。

## Task 2: 标准消息与有版本的兼容检查点

**Files:** 新增 `backend/app/agent/messages.py`、`backend/tests/unit/test_agent_messages.py`；修改 `state.py`、`checkpoint.py` 和 `tests/unit/test_agent_checkpoint.py`。

**Interfaces:** `normalize_messages(messages: list[Any]) -> list[BaseMessage]`；`message_text(message: Any) -> str`；`dump_messages(messages: list[BaseMessage]) -> list[dict]`；`load_messages(raw: list[dict], *, version: int) -> list[BaseMessage]`。检查点顶层 `checkpoint_version=1`；缺失版本按旧版 0 读取；其他版本拒绝。既有序列化函数签名保持不变。

- [x] **Step 1:** 编写 `test_normalize_legacy_tool_calls`，断言旧 OpenAI 结构转换后 `message.tool_calls[0] == {"name": "calculator", "args": {"a": 1, "b": 2, "operation": "add"}, "id": "call-1", "type": "tool_call"}`，工具结果转换为 ToolMessage（工具结果消息）。
- [x] **Step 2:** 编写检查点往返测试，断言保留工具标识和 next_node（恢复节点），排除正文、回调与秘密；缺失版本的旧检查点可读，版本 999 抛安全 ValueError（参数异常）；反序列化不修改仓库提供的原字典。
- [x] **Step 3:** 运行 `.\venv\Scripts\python.exe -m pytest tests/unit/test_agent_messages.py tests/unit/test_agent_checkpoint.py -q`，确认因缺少转换行为失败。
- [x] **Step 4:** 实现接口并使用标准消息的公开序列化方法；在消息转字典前执行工具结果省略，避免 LangChain 的 `type/data` 格式绕过旧 `role` 判断。运行态状态声明改为标准消息，旧格式只在输入边界接受。
- [x] **Step 5:** 重跑 Step 3，预期全部通过；提交任务文件，提交说明 `feat: normalize messages and version agent checkpoints`。

## Task 3: 结构化工具适配与执行权限

**Files:** 新增 `backend/app/agent/tool_adapter.py`、`backend/tests/unit/test_langchain_tools.py`；修改 `service.py`、`registry.py`，按需要修改 `skill.py`；回归已有工具、技能与 MCP 测试。

**Interfaces:** `build_tool(tool_name: str, *, knowledge_base_id: str | None, allowed_tools: frozenset[str] | tuple[str, ...] | None) -> BaseTool`；`invoke_registered_tool(tool_name: str, arguments: dict[str, Any], knowledge_base_id: str | None, *, allowed_tools: frozenset[str] | tuple[str, ...] | None = None, config: RunnableConfig | None = None) -> dict`。保留 `execute_tool` 旧签名，解析 JSON 后委托新入口，禁止新旧函数递归互调。

- [x] **Step 1:** 编写 `test_calculator_invokes_structured_tool`，断言实际 LangChain 工具 `.invoke()` 参与且 `1 + 2` 返回 3；`test_tool_rejects_unallowed_name_before_handler` 断言越权工具处理器调用次数为 0。
- [x] **Step 2:** 编写 `test_knowledge_context_cannot_be_overridden`，断言模型传入知识库标识被拒绝，合法查询只向处理器注入后端标识；MCP 数字 text 参数必须被 JSON Schema 拒绝，未知动态工具不覆盖静态工具。
- [x] **Step 3:** 运行 `.\venv\Scripts\python.exe -m pytest tests/unit/test_langchain_tools.py -q`，确认新适配路径尚未实现导致失败。
- [x] **Step 4:** 从现有注册定义构造 StructuredTool（结构化工具），模型参数结构复用现有 Pydantic 或 JSON Schema。权限和 MCP 宿主允许列表检查在工具构造与执行入口都保留，实际处理函数只有一份；MCP 仍调用客户端而非服务端函数。
- [x] **Step 5:** 运行 `.\venv\Scripts\python.exe -m pytest tests/unit/test_langchain_tools.py tests/unit/test_agent_registry.py tests/unit/test_agent_public_tools.py tests/unit/test_calculator.py tests/unit/test_skills.py tests/integration/test_mcp.py -q`，预期全部通过，包含真实 MCP 子进程；提交任务文件，提交说明 `feat: execute registered tools through LangChain`。

## Task 4: 统一模型节点和支持工具的流式响应

**Files:** 新增 `backend/app/agent/model_response.py`；修改 `nodes.py`、`service.py`、模型调用相关单元与集成测试及 `tests/evals/runner.py` 中的模拟模型接口；主要新增 `backend/tests/unit/test_langchain_streaming.py`。

**Interfaces:** `ModelTurn` 数据类字段为 `message: AIMessage | None`、`control_status: str | None`、`answer_prefix: str`；`consume_model_stream(chunks: Iterator[AIMessageChunk], *, on_token: Callable[[str], None] | None, control_poll: Callable[[], str | None] | None, allow_tools: bool, buffer_text: bool) -> ModelTurn`。图节点采用 LangGraph 支持的 `config: RunnableConfig` 参数并传播至模型和工具调用。

- [x] **Step 1:** 将现有模拟模型改为 LangChain `invoke`、`stream`、`bind_tools` 接口，保留每个原测试的业务断言；不再以返回 OpenAI SDK 对象作为生产接口测试。
- [x] **Step 2:** 新增同步和流式参数化测试 `test_two_tools_finish_in_five_steps`，断言工具顺序为 calculator、get_current_time，`steps == 5`、`llm_calls == 3`；第五步再请求工具时执行次数不增加。
- [x] **Step 3:** 编写 `test_stream_without_knowledge_base_can_use_tools`，分别覆盖计算、时间和字符统计；`test_stream_buffers_text_followed_by_tool_chunks` 断言中间文本及 JSON 参数未进入正文回调；多个调用不部分执行，畸形 JSON 转安全工具错误。
- [x] **Step 4:** 编写 `test_pause_during_partial_tool_call_executes_nothing`，断言关闭上游流、无工具执行、不保存半成品调用，同时增加真实模型调用计数和耗时。空流、空消息块、工具分片跨多个块和没有用量元数据均有断言。
- [x] **Step 5:** 运行 `.\venv\Scripts\python.exe -m pytest tests/unit/test_agent_loop.py tests/unit/test_agent_streaming.py tests/unit/test_langchain_streaming.py -q`，确认新增流式与标准消息行为失败。
- [x] **Step 6:** 节点通过 `create_chat_model().bind_tools(...)` 调用模型；普通与流式采用相同工具过滤和路由，移除旧 `create_llm_client` 及直接 `chat.completions.create` 正常业务路径。首轮专用工具意图不依赖知识库；保留知识库默认检索规则。
- [x] **Step 7:** 实现流累加和控制轮询；工具决策阶段缓冲文本，纯回答阶段允许分片输出。可用工具不为空时不能仅凭先出现文本就判定最终答案；暂停时不执行尚未完成的调用。不增加用于分类的隐藏模型请求。
- [x] **Step 8:** 重跑 Step 5 和 `.\venv\Scripts\python.exe -m pytest tests/evals tests/integration/test_mcp.py tests/unit/test_prompt_injection.py tests/unit/test_agent_public_tools.py -q`；预期通过，提交任务文件，提交说明 `feat: unify LangChain agent invocation and streaming`。

## Task 5: 同步与可恢复任务共用结果校验

**Files:** 新增 `backend/app/agent/result.py`、`backend/tests/unit/test_agent_results.py`；修改 `service.py`、`runner.py`、`nodes.py` 和相关导入位置。

**Interfaces:** 将 `AgentResult` 移入 result 模块，旧 service 模块继续导出以兼容调用方；`result_from_state(state: AgentState, *, status: str = "completed") -> AgentResult` 为唯一结果构建入口。政策摘要函数从 service 移入 result 模块，规则不放宽。

- [x] **Step 1:** 新增 `test_both_entrypoints_ground_policy_fields_identically`，使用相同检索片段与含编造字段的模型回答，断言两个入口都将无依据字段改为“知识库未说明”，有依据字段相同。
- [x] **Step 2:** 新增 `test_policy_summary_never_emits_unvalidated_text`，断言正文回调没有编造字段；全空或纯标点完成答案安全报错，暂停与停止不因没有完整答案误报失败。
- [x] **Step 3:** 运行 `.\venv\Scripts\python.exe -m pytest tests/unit/test_agent_results.py -q`，确认统一校验尚缺失导致失败。
- [x] **Step 4:** 实现统一结果构造，两个入口都调用；政策摘要只在完成校验后发送最终正文。保留公开字段、工具元数据和现有前端完成事件结构。
- [x] **Step 5:** 运行 `.\venv\Scripts\python.exe -m pytest tests/unit/test_agent_results.py tests/unit/test_skills.py tests/integration/test_streaming.py tests/integration/test_memory.py -q`，预期全部通过；提交任务文件，提交说明 `fix: share grounded results across agent execution paths`。

## Task 6: 恢复完整调用、预算与依据

**Files:** 修改 `runner.py`、`checkpoint.py`、`state.py`、`memory/service.py`；扩展 `tests/unit/test_agent_runner.py`、`test_agent_checkpoint.py` 与 `tests/integration/test_agent_run_lifecycle.py`。

**Interfaces:** 保留 `run_resumable_agent` 的公开调用参数，新增可选 `config: RunnableConfig | None = None`。检查点记录 `omitted_tool_result_ids: list[str]`；运行态新增同名可选字段，明确哪些结果已省略。`prepare_resumed_state(state: AgentState) -> AgentState` 在 runner 模块处理省略后的安全恢复，不伪造结果或重置计数。

- [x] **Step 1:** 编写 `test_resume_completed_tool_does_not_repeat_execution`：暂停在工具完成后，恢复不再次调用计算器或 MCP；不得用“工具结果已省略”生成事实答案。无需工具依据的最终前缀续写不能重复发送前缀。
- [x] **Step 2:** 编写 `test_resume_pending_call_preserves_call_id_and_budget`：完整待执行调用从旧版或新版检查点恢复，工具仅执行一次，原 steps 和 llm_calls 累加；半成品调用恢复重做决策且计数不归零。
- [x] **Step 3:** 编写 `test_resume_omitted_retrieval_reacquires_within_budget`：剩余预算允许时通过正常工具节点重新检索当前有权限的知识库，计入工具步数；预算不足返回明确无法恢复依据的安全提示。
- [x] **Step 4:** 运行 `.\venv\Scripts\python.exe -m pytest tests/unit/test_agent_runner.py tests/unit/test_agent_checkpoint.py tests/integration/test_agent_run_lifecycle.py -q`，确认新恢复语义的失败来自实际行为。
- [x] **Step 5:** 实现恢复准备：不重复已完成工具；省略的只读检索允许重新获取并计步；省略计算或时间结果且仍需其事实时安全终止，不伪造结果。恢复前重新施加当前权限。清理未完成流的临时片段，恢复已发布前缀时仅追加新文本。
- [x] **Step 6:** 重跑 Step 4，验证停止、超时、后台线程终态、前端完成事件和数据库回复一致；提交任务文件，提交说明 `fix: preserve agent budgets and evidence on resume`。

## Task 7: 安全接入 LangSmith 并传播执行上下文

**Files:** 新增 `backend/app/agent/tracing.py`、`backend/tests/unit/test_langsmith_tracing.py`、`backend/tests/integration/test_langsmith_flow.py`；修改 `service.py`、`runner.py`、`memory/service.py`、`backend/.env.example`，必要时仅为上下文传递修改 `main.py`、`run_manager.py`。

**Interfaces:** `agent_trace_config(correlation_id: str) -> ContextManager[RunnableConfig]` 管理请求追踪；`sanitize_trace_payload(payload: dict[str, Any]) -> dict[str, Any]` 对最终发送载荷执行字段允许列表。可选配置读取与安全追踪客户端均封装在 tracing 模块；不返回外部平台凭据。

- [x] **Step 1:** 编写 `test_disabled_tracing_never_sends_requests` 与 `test_enabled_without_key_preserves_chat`，断言默认 false、未配置密钥时追踪传输为零且业务返回成功。
- [x] **Step 2:** 编写完整载荷测试 `test_trace_payload_excludes_content_metadata_and_exception_secrets`：在消息、工具参数、结果、文档、metadata（元数据）、异常、模型配置中植入不同秘密；遍历最终出站载荷，断言均不存在，仅保留设计允许的元数据及必要追踪关联字段。
- [x] **Step 3:** 编写 `test_worker_resume_links_distinct_segments`，断言实际图、模型、工具和检索子运行具备父子关系；后台线程建立上下文，恢复片段使用新追踪运行标识、相同业务关联标识。测试 `create_run`、`update_run` 与批量发送路径的脱敏。
- [x] **Step 4:** 编写 `test_tracing_transport_failure_preserves_business_outcome`，追踪初始化、发送与结束分别故障时聊天仍成功，真实模型错误仍按原规则失败；日志和追踪错误字段均不泄露异常原文。
- [x] **Step 5:** 运行 `.\venv\Scripts\python.exe -m pytest tests/unit/test_langsmith_tracing.py tests/integration/test_langsmith_flow.py -q`，确认因尚无追踪策略失败。
- [x] **Step 6:** 实现只上传元数据的 LangSmith 客户端与回调，图输入输出、模型消息、工具参数及结果都不出站；阻止环境变量触发第二套未脱敏自动上传。默认关闭时在显式禁用上下文中运行，开启时传播受控回调和配置，传输故障只记录安全诊断。
- [x] **Step 7:** 普通请求使用已有 request_id；可恢复任务在线程内使用 run_id 建立上下文。配置随图调用传入模型与工具，不能把回调写入检查点；不把 LangSmith 密钥加入 MCP 子进程环境。
- [x] **Step 8:** 重跑 Step 5，并运行 `.\venv\Scripts\python.exe -m pytest tests/integration/test_observability.py tests/integration/test_memory.py tests/integration/test_agent_run_lifecycle.py tests/integration/test_mcp.py -q`，预期全部通过；提交任务文件，提交说明 `feat: add metadata-only LangSmith tracing`。

## Task 8: 全流程回归与教学文档

**Files:** 修改 `README.md`、必要的测试夹具和 `backend/tests/conftest.py`；新增 `backend/tests/integration/test_framework_migration.py`，更新本计划的完成记录。

**Interfaces:** 沿用公共聊天接口、事件和结果字段；集成测试使用实际 LangChain 模型适配器加模拟 HTTP 传输，实际 LangGraph 与 MCP，不用替换整张图来证明集成成功。

- [x] **Step 1:** 编写 `test_framework_flow_preserves_public_contract`，通过真实会话服务和受控模型响应验证同步与流式“计算再查时间”；断言两个工具、五步、协议通信发生、公开字段不变、工具结果不进入 messages 表。
- [x] **Step 2:** 编写 `test_policy_resume_persists_only_validated_answer`，通过临时数据库执行暂停恢复政策摘要，断言最终完成事件、助手数据库消息和统一结果相同，当前用户消息只出现一次；保留会话隔离、不存在资源与最近 20 条顺序测试。
- [x] **Step 3:** 运行 `.\venv\Scripts\python.exe -m pytest tests/integration/test_framework_migration.py -q`，修复跨模块连接失败，不能通过减少断言或改成整层模拟掩盖问题。
- [x] **Step 4:** 运行 `.\venv\Scripts\python.exe -m pip check`、`.\venv\Scripts\python.exe -m pytest -q`、`.\venv\Scripts\python.exe -m compileall -q app mcp_servers`。预期依赖无冲突、后端全部通过、编译无错误；记录数量与警告。
- [x] **Step 5:** 若改变公共响应或前端代码，完成前端规范检查和生产构建。按用户 Windows 约定核对 `C:\Program Files\nodejs\npm.cmd` 与 PATH（命令搜索路径），使用完整路径并申请必要权限；不重装 Node.js（运行环境）或擅改 PATH。
- [x] **Step 6:** 更新 README：三框架职责、迁移后的文件索引、LangSmith 默认关闭及启用配置、元数据上传范围、五步口径、流式首字延迟和暂停恢复限制。提供不包含真实密钥的配置示例。
- [x] **Step 7:** 若有可用凭据和授权网络，使用非敏感示例分别验证真实通义千问和真实 LangSmith；没有条件则明确记为未执行，不能代替完整本地验收，也不能宣称外部联通。不得为此上传用户文档或读取并展示真实密钥。
- [x] **Step 8:** 用设计第 9 节验收矩阵逐项核对实际证据，检查 `git diff --check`，提交本任务文件，提交说明 `test: verify framework migration and document architecture`。按对应执行技能完成代码审核与交付，未验证项目明确列为待办。

## 自查与执行交接

覆盖对应：设计第 4 节由任务 1、2 实现；第 5 节由任务 3、5 实现；第 6 节由任务 4、6 实现；第 7 节由任务 7 实现；第 8、9 节由任务 1、8 及各任务的回归检查实现。全部新增接口已在所属任务定义；依赖解析结果只在实际实施后写入锁定文件，不预填未经验证的版本。

本计划已获用户批准，采用当前会话逐项实施；实现和本地验收已完成，详细证据见 `../verification/2026-10-19-langchain-langgraph-langsmith.md`。条件性外部平台验证明确未执行，不计为已联通。推荐 Native（由当前智能代理在本会话逐项实施），因为消息、节点、恢复和追踪接口关联紧密，顺序实施便于及时做整体验证；也可选择 Subagent-driven（分任务智能代理实施并逐项审核），成本较高但增加独立检查。
