# Agent Goal 生命周期与可调试流式问答实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在现有 SSE 聊天基础上加入可持久化的 Goal（目标）暂停/恢复/停止/超时生命周期、并发提问和可调试的 RAG 检索轨迹。

**Architecture:** 用 SQLite `agent_tasks` 表保存 Run（执行任务）状态和最近安全检查点；用现有 LangGraph（语言模型工作流）节点流式执行，并在节点边界保存状态。SSE（服务器推送事件）负责传递任务状态、文本片段和安全轨迹，前端用 `run_id` 管理多个独立回答。保留现有普通 JSON 接口和 `/api/agent/chat/stream` 创建接口的兼容性。

**Tech Stack:** FastAPI、SQLite、LangGraph、OpenAI-compatible Chat Completions、React、浏览器 Fetch Streams、SSE。

**Spec:** `docs/superpowers/specs/2026-09-28-agent-goal-lifecycle-design.md`

## Global Constraints

- 所有回答、注释和用户可见错误使用中文；首次出现的英文术语必须附中文解释。
- `knowledge_base_id` 与 `conversation_id` 不得混用；Run 控制接口必须校验 `owner_id`。
- Tool Call（工具调用）、Tool Result（工具结果）和隐藏思维链不进入用户聊天记录或前端轨迹。
- RAG 文档、MCP 返回值和工具输出都是不可信数据；检索调试只公开数值元数据。
- 不提交 `backend/.env`、正式 SQLite/Chroma 数据、`node_modules` 或 `dist`。
- 保留现有 `/api/chat`、`/api/agent/chat` 和普通 JSON 响应兼容性。
- 使用现有会话锁保证同一会话的持久化顺序；前端可以提前提交下一题，但不能让答案通过数组最后一项串线。

## Review Focus

- 暂停与完成同时到达：必须只有一个终态获胜，并保留最后安全检查点；由任务仓储并发测试覆盖。
- 同一会话的两个 Run 同时提交：两个前端流不能串 token，数据库消息仍按会话锁顺序保存；由流式集成测试覆盖。
- SSE 断开与主动取消的区别：断开只停止订阅，取消才终止 Run；由控制接口测试覆盖。
- 访客恢复与重试的额度差异：恢复不重复扣次，重试创建新 Run 并重新扣次；由额度集成测试覆盖。
- RAG 无命中、`top_k` 边界和注入文档：轨迹安全、答案继续遵守无命中规则；由 RAG 单元测试覆盖。

---

### Task 1: 持久化 Agent Task（任务）与安全检查点

**Files:**
- Modify: `backend/app/memory/database.py`
- Modify: `backend/app/memory/repository.py`
- Create: `backend/app/agent/checkpoint.py`
- Create: `backend/tests/unit/test_agent_tasks.py`
- Create: `backend/tests/unit/test_agent_checkpoint.py`

**Interfaces:**
- Produces `create_agent_task(...) -> dict`、`get_agent_task(run_id, owner_id) -> dict | None`、`list_agent_tasks(...) -> list[dict]` 和带期望状态条件的 `update_agent_task(...) -> bool`。
- Produces `serialize_agent_state(state) -> str`、`deserialize_agent_state(raw) -> AgentState` 和 `merge_agent_update(state, update) -> AgentState`。

- [ ] **Step 1: Write the failing tests**

  测试任务创建、按 `owner_id` 隔离读取、合法状态转换、并发终态条件更新，以及包含模型消息/工具调用的状态可以序列化后恢复；断言密钥、完整 chunk 和回调不会进入 checkpoint。

- [ ] **Step 2: Run tests to verify they fail**

  Run: `.\venv\Scripts\python.exe -m pytest tests/unit/test_agent_tasks.py tests/unit/test_agent_checkpoint.py -q`
  Expected: FAIL，因为表、仓储函数和检查点模块尚未存在。

- [ ] **Step 3: Implement the persistence boundary**

  在 `init_database()` 幂等创建 `agent_tasks` 表和索引；仓储层只使用参数化 SQL，并让状态更新带 `WHERE run_id AND owner_id AND status IN (...)`。检查点只保存可恢复的公开 Agent 状态，Pydantic（数据模型）消息转换为普通字典。

- [ ] **Step 4: Run tests to verify they pass**

  Run the same targeted pytest command. Expected: PASS。

- [ ] **Step 5: Commit**

  `git add backend/app/memory/database.py backend/app/memory/repository.py backend/app/agent/checkpoint.py backend/tests/unit/test_agent_tasks.py backend/tests/unit/test_agent_checkpoint.py`
  `git commit -m "feat: persist agent task checkpoints"`

### Task 2: 可暂停、可恢复的 LangGraph 执行器

**Files:**
- Modify: `backend/app/agent/state.py`
- Modify: `backend/app/agent/graph.py`
- Modify: `backend/app/agent/nodes.py`
- Modify: `backend/app/agent/service.py`
- Create: `backend/app/agent/runner.py`
- Create: `backend/tests/unit/test_agent_runner.py`

**Interfaces:**
- Produces `RunControl.request_pause()`、`RunControl.cancel()`、`RunControl.check_boundary()`。
- Produces `run_resumable_agent(initial_state, *, control, checkpoint_callback, status_callback, on_token) -> AgentResult`；恢复时接受反序列化的 checkpoint 状态。

- [ ] **Step 1: Write the failing tests**

  测试执行器在模型节点和工具节点之间保存 checkpoint；收到暂停请求后返回 `paused` 而不进入下一节点；取消和超时不会继续调用工具；从 checkpoint 恢复时不重复追加用户消息，并最终只生成一个完整回答。

- [ ] **Step 2: Run tests to verify they fail**

  Run: `.\venv\Scripts\python.exe -m pytest tests/unit/test_agent_runner.py -q`
  Expected: FAIL，因为现有 Agent 只有一次性 `agent_graph.invoke()`，没有 RunControl 或安全边界。

- [ ] **Step 3: Implement the resumable runner**

  让图从 `START` 根据 `next_node` 进入 `agent` 或 `tools`，使用现有节点和路由，不重写工具注册。执行器消费图的节点更新并合并状态，在每个节点结束后调用 checkpoint 回调；节点开始前调用 `check_boundary()`。暂停和取消使用线程事件，模型 token 回调只负责发送文本，不保存隐藏响应。

- [ ] **Step 4: Run tests to verify they pass**

  Run targeted runner tests and existing Agent tests: `.\venv\Scripts\python.exe -m pytest tests/unit/test_agent_runner.py tests/unit/test_agent_loop.py tests/unit/test_agent_public_tools.py -q`
  Expected: PASS。

- [ ] **Step 5: Commit**

  `git add backend/app/agent/state.py backend/app/agent/graph.py backend/app/agent/nodes.py backend/app/agent/service.py backend/app/agent/runner.py backend/tests/unit/test_agent_runner.py`
  `git commit -m "feat: add resumable agent runner"`

### Task 3: RAG 检索调试元数据与安全轨迹

**Files:**
- Modify: `backend/app/rag/service.py`
- Modify: `backend/app/agent/skills/knowledge_skill.py`
- Modify: `backend/app/agent/nodes.py`
- Modify: `backend/app/agent/service.py`
- Modify: `backend/app/observability/trace.py`
- Modify: `backend/app/memory/service.py`
- Modify: `backend/app/configuration.py`
- Modify: `backend/.env.example`
- Create: `backend/tests/unit/test_rag_debug_trace.py`

**Interfaces:**
- Produces `RAG_TOP_K` 配置，默认 `5`，有效范围 `1..20`。
- `knowledge_search()` 返回 `retrieval_debug`：`top_k`、`max_distance`、`returned`、`matches`，其中每个 match 只有 rank、similarity、distance。
- `RequestTrace.rag` 保持已有 `hit/results` 字段，并扩展安全调试字段。

- [ ] **Step 1: Write the failing tests**

  测试默认和边界 `top_k`、无命中结果、相似度/距离排序、轨迹不包含 chunk 正文和工具参数，并确认现有 `RAG_MAX_COSINE_DISTANCE` 行为不变。

- [ ] **Step 2: Run tests to verify they fail**

  Run: `.\venv\Scripts\python.exe -m pytest tests/unit/test_rag_debug_trace.py -q`
  Expected: FAIL，因为当前检索结果只有 chunks/similarities，轨迹只有命中数量。

- [ ] **Step 3: Implement bounded retrieval metadata**

  从环境变量读取并校验 `RAG_TOP_K`，把实际阈值和候选结果数转换为安全元数据；保留完整 chunks 仅供 Agent 内部回答和已有安全过滤逻辑使用，不写入 `RequestTrace`。

- [ ] **Step 4: Run tests to verify they pass**

  Run targeted RAG tests plus existing RAG/eval tests: `.\venv\Scripts\python.exe -m pytest tests/unit/test_rag_debug_trace.py tests/unit/test_rag_chunking.py tests/evals/test_evals.py -q`
  Expected: PASS。

- [ ] **Step 5: Commit**

  `git add backend/app/rag/service.py backend/app/agent/skills/knowledge_skill.py backend/app/agent/nodes.py backend/app/agent/service.py backend/app/observability/trace.py backend/app/memory/service.py backend/app/configuration.py backend/.env.example backend/tests/unit/test_rag_debug_trace.py`
  `git commit -m "feat: expose safe rag debug metadata"`

### Task 4: Run 生命周期 API、SSE 控制和错误处理

**Files:**
- Modify: `backend/app/main.py`
- Modify: `backend/app/streaming.py`
- Modify: `backend/app/memory/service.py`
- Create: `backend/app/agent/run_manager.py`
- Create: `backend/tests/integration/test_agent_run_lifecycle.py`

**Interfaces:**
- `POST /api/agent/chat/stream` 首帧发送 `run`，并在后台通过 RunManager（任务管理器）执行。
- 新增 `POST /api/agent/runs/{run_id}/pause`、`resume`、`cancel`，以及 `GET /api/agent/runs/{run_id}` 和按会话查询接口。
- SSE 事件统一为 `run/status/token/trace/paused/done/cancelled/timeout/error`。

- [ ] **Step 1: Write the failing tests**

  测试创建 Run、暂停后不自动继续、同一 Run 恢复、主动取消、模型异常、总超时、SSE 断开不等同取消、Run 所有权校验和访客额度只预占一次。

- [ ] **Step 2: Run tests to verify they fail**

  Run: `.\venv\Scripts\python.exe -m pytest tests/integration/test_agent_run_lifecycle.py -q`
  Expected: FAIL，因为控制接口和持久化 Run 管理器不存在。

- [ ] **Step 3: Implement the API boundary**

  把现有 `_chat` 的额度、会话和结果组装逻辑拆成可被 RunManager 调用的步骤；控制接口只改变任务状态/事件，不直接执行模型。所有错误转换为安全错误码，SSE 响应设置禁缓存和反向代理不缓冲头；恢复沿用原 `run_id`，重试仍走创建新 Run。每个 SSE（服务器推送事件）事件都携带 `run_id`，使多个并发流可以安全路由到对应助手卡片。

- [ ] **Step 4: Run tests to verify they pass**

  Run integration lifecycle tests plus existing streaming/observability/security tests: `.\venv\Scripts\python.exe -m pytest tests/integration/test_agent_run_lifecycle.py tests/integration/test_streaming.py tests/integration/test_observability.py tests/unit/test_api_security.py -q`
  Expected: PASS。

- [ ] **Step 5: Commit**

  `git add backend/app/main.py backend/app/streaming.py backend/app/memory/service.py backend/app/agent/run_manager.py backend/tests/integration/test_agent_run_lifecycle.py`
  `git commit -m "feat: add agent run lifecycle api"`

### Task 5: 前端并发 Run 管理与控制操作

**Files:**
- Modify: `frontend/src/App.jsx`
- Modify: `frontend/src/streaming.js`
- Modify: `frontend/src/App.css`
- Modify: `frontend/package.json`
- Create: `frontend/src/run-manager.js`
- Create: `frontend/src/run-manager.test.js`

**Interfaces:**
- `createRunManager({ streamRequest, request })` 返回 `start`, `pause`, `resume`, `cancel`, `retry` 和 `activeRuns` 管理接口。
- 每条 UI 消息使用稳定 `run_id` 更新，不允许通过数组最后一项匹配。

- [ ] **Step 1: Write the failing tests**

  测试两个 Run 同时接收 token 时分别更新；暂停/恢复/取消调用正确接口；超时产生可重试错误；重试创建新 Run；发送第二个问题时输入框仍可用。

- [ ] **Step 2: Run tests to verify they fail**

  Run: `node --test src/run-manager.test.js`
  Expected: FAIL，因为当前 App 只有全局 `loading` 和单个流请求。

- [ ] **Step 3: Implement the frontend run manager**

  用 `Map` 保存每个 Run 的 AbortController（请求取消控制器）、状态、文本缓冲和 `conversation_id`；SSE 事件按 `run_id` 分发。网络断开只标记需要重连，主动取消才调用后端 cancel；为每个 Run 设置总超时并清理计时器。输入框只在会话未加载或额度耗尽时禁用。

- [ ] **Step 4: Run tests to verify they pass**

  Run: `node --test src/run-manager.test.js src/streaming.test.js src/usage.test.js src/deployment-config.test.js`
  Expected: PASS。

- [ ] **Step 5: Commit**

  `git add frontend/src/App.jsx frontend/src/streaming.js frontend/src/App.css frontend/package.json frontend/src/run-manager.js frontend/src/run-manager.test.js`
  `git commit -m "feat: control concurrent agent runs in chat"`

### Task 6: 工作过程、状态按钮和可操作错误界面

**Files:**
- Modify: `frontend/src/App.jsx`
- Modify: `frontend/src/ExecutionTrace.css`
- Create: `frontend/src/RunControls.css`
- Modify: `frontend/src/run-manager.test.js`

**Interfaces:**
- `ExecutionTrace` 接收扩展后的 `trace.rag` 和状态转换数据。
- 每个活动助手卡片显示暂停/继续/停止；失败、超时和停止显示重试或关闭结果。

- [ ] **Step 1: Write the failing tests**

  测试状态文案、按钮禁用条件、RAG `topK/阈值/排名/相似度/距离` 的安全显示，以及错误详情中不出现内部堆栈或密钥。

- [ ] **Step 2: Run tests to verify they fail**

  Run: `node --test src/run-manager.test.js`
  Expected: FAIL，因为现有轨迹只显示工具名称和命中数量。

- [ ] **Step 3: Implement the UI**

  将控制按钮放在对应助手卡片内；状态变化只更新对应 Run；轨迹面板增加检索详情和耗时分解，使用固定中文安全文案，避免把服务端原始异常渲染到页面。

- [ ] **Step 4: Run tests to verify they pass**

  Run: `node --test src/run-manager.test.js src/streaming.test.js src/usage.test.js src/deployment-config.test.js`
  Expected: PASS。

- [ ] **Step 5: Commit**

  `git add frontend/src/App.jsx frontend/src/ExecutionTrace.css frontend/src/RunControls.css frontend/src/run-manager.test.js`
  `git commit -m "feat: add run controls and rag trace view"`

### Task 7: 全量验证、文档和交付检查

**Files:**
- Modify: `README.md` or `docs/` only if the final API/configuration needs user-facing documentation.
- Test: existing backend and frontend suites.

- [ ] **Step 1: Run backend verification**

  `.\venv\Scripts\python.exe -m pytest -q`
  `.\venv\Scripts\python.exe -m compileall -q app mcp_servers`

- [ ] **Step 2: Run frontend verification**

  `node --test src/usage.test.js src/deployment-config.test.js src/streaming.test.js src/run-manager.test.js`
  `node node_modules/eslint/bin/eslint.js .`
  `node node_modules/vite/bin/vite.js build`

- [ ] **Step 3: Review security and diff**

  检查 `git diff --check`、`git status --short`、敏感配置未被加入提交、所有 Run 控制接口都有所有权测试、RAG 轨迹不含正文。

- [ ] **Step 4: Commit documentation changes if any**

  文档只在接口或配置确实变化时更新，并单独提交，避免把运行数据和构建产物带入 Git。

- [ ] **Step 5: Prepare delivery**

  汇总变更文件、测试数量、非阻塞警告和云端同步前置条件；只有所有验证命令新鲜通过后，才推送 GitHub 并在用户确认云端未提交文件处理方式后部署。
