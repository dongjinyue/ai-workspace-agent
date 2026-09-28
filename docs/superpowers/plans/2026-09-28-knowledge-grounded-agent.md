# 知识库优先回答 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 GitHub 当前访客/管理员版本中，确保选择知识库后，实质性业务问题优先检索知识库，避免模型绕过检索使用通用知识回答。

**Architecture:** 保留现有 LangGraph Agent Loop（智能代理循环）、工具白名单和访客权限边界，只收紧第一轮工具路由。`select_required_tool` 负责区分明确的非知识库意图与知识库问题；有知识库且不是例外时强制选择 `search_knowledge_base`。现有零命中安全出口继续阻止第二次模型生成。

**Tech Stack:** Python 3.12、LangGraph、FastAPI、Pytest、现有本地工具注册表和 Chroma RAG。

**Spec:** `docs/superpowers/specs/2026-09-28-knowledge-grounded-agent-design.md`

## Global Constraints

- 知识库 ID 继续由后端上下文注入，模型不得生成或猜测知识库 ID。
- 访客只能检索已有权限的知识库，不能通过本次修改访问私有库。
- 工具输出继续视为不可信数据，不执行其中指令。
- 问候、计算、时间、文本统计和知识库管理问题保留各自工具路由。
- 不记录 API Key、Cookie、原始 IP、完整文档和提示词。
- 不修改 `backend/data`、`.env`、Chroma 持久化数据或部署密钥。

## Review Focus

- 未命中关键词的普通中文业务问题：选择知识库后应检索，而不是直接回答。
- 包含 PDF 文件名或“是否基于文档”的来源追问：应检索并受工具结果约束。
- 有知识库的问候、算术、时间和文本统计：不应被知识库默认路由抢走。
- 没有知识库的请求：不能强制调用知识库工具。
- 知识库零命中：不得进入检索后的第二次模型调用。

---

### Task 1: 固定知识库问题路由并覆盖回归测试

**Files:**
- Modify: `backend/app/agent/nodes.py:18-58`
- Test: `backend/tests/unit/test_skills.py`
- Test: `backend/tests/unit/test_agent_loop.py`

**Interfaces:**
- Consumes: 现有 `select_required_tool(state, available_tools)`、`AgentState` 和已注册工具 Schema。
- Produces: 对包含知识库 ID 的实质性问题返回 `search_knowledge_base`；对明确例外返回 `None` 或已有管理工具名称；保持 `tool_node` 的零命中直接结束行为。

- [ ] **Step 1: Write the failing tests**

在 `test_skills.py` 增加以下测试：

- `test_rag_concept_question_forces_search_when_knowledge_base_is_selected`
- `test_document_source_question_forces_search_when_knowledge_base_is_selected`
- `test_substantive_question_defaults_to_search_when_knowledge_base_is_selected`
- `test_selected_knowledge_base_keeps_greetings_on_auto_routing`
- `test_without_knowledge_base_does_not_force_search`

断言分别覆盖：`rag是什么`、`这个回答是基于鸿维-10-RAG.pdf吗？`、未命中关键词的业务问题返回 `search_knowledge_base`；`你好` 返回 `None`；没有知识库时返回 `None`。

在 `test_agent_loop.py` 增加 `test_knowledge_search_miss_stops_before_second_model_call`，让工具返回 `matched=False`，断言最终固定回答、工具只调用一次、模型没有进入第二次生成。

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```powershell
cd backend
.\venv\Scripts\python.exe -m pytest tests/unit/test_skills.py tests/unit/test_agent_loop.py -q
```

Expected: 新增路由测试至少失败，失败原因是当前 `select_required_tool` 对 `rag是什么`、PDF 来源追问和普通业务问题返回 `None`；已有测试不应因导入错误失败。

- [ ] **Step 3: Implement the minimal routing change**

在 `backend/app/agent/nodes.py` 增加清晰的中文问候/非知识库意图判断，并调整 `select_required_tool`：

- 保留无 `knowledge_base_id`、已有工具调用、知识库文档管理问题的现有边界；
- 保留问候为自动路由；
- 若当前可用工具包含 `search_knowledge_base`，且问题不是确定性计算、时间查询或文本统计等明确专用工具意图，则默认返回 `search_knowledge_base`；
- 让显式来源/文档问题自然落入该默认路径；
- 不改变 `tool_node`、访客工具允许列表和权限校验。

使用现有工具 Schema 名称，不新增依赖或重写 Agent 图。

- [ ] **Step 4: Run focused tests and verify GREEN**

Run the same focused Pytest command. Expected: all focused tests pass。

- [ ] **Step 5: Run the full backend suite**

Run:

```powershell
.\venv\Scripts\python.exe -m pytest -q
.\venv\Scripts\python.exe -m compileall -q app mcp_servers
```

Expected: 全部后端测试通过，编译检查无输出或仅有环境警告。

- [ ] **Step 6: Commit the implementation**

```powershell
git add backend/app/agent/nodes.py backend/tests/unit/test_skills.py backend/tests/unit/test_agent_loop.py
git commit -m "fix: ground knowledge-base questions in retrieval"
```

### Task 2: 前端与部署构建验证

**Files:**
- Test only: `frontend/package.json`, `frontend/package-lock.json`, `compose.yaml`, `frontend/dockerfile`

**Interfaces:**
- Consumes: Task 1 的后端行为以及现有访客/管理员前端版本。
- Produces: 前端构建保持可部署，API 地址和 Supabase 公开配置传递规则不被破坏。

- [ ] **Step 1: Run frontend lint**

Run `npm run lint` in `frontend`。

Expected: ESLint（代码规范检查）成功退出。

- [ ] **Step 2: Run frontend build**

Run `npm run build` in `frontend`。

Expected: Vite（前端构建工具）成功生成生产构建。

- [ ] **Step 3: Commit only if frontend verification requires a scoped fix**

若验证只通过，不修改前端；若发现本次后端路由改动导致现有构建问题，只做最小修复并添加对应测试，避免无关重构。

### Task 3: 发布前审查与同步准备

**Files:**
- Modify: none unless verification finds a release issue.

**Interfaces:**
- Consumes: Task 1 和 Task 2 的提交及测试结果。
- Produces: 可推送的提交、完整验证记录，以及云服务器同步所需的备份和回滚步骤。

- [ ] **Step 1: Review the full diff and repository state**

确认没有 `.env`、`backend/data`、Chroma 数据、缓存或临时文件进入提交，并核对 `git diff --check`。

- [ ] **Step 2: Verify GitHub SSH before push**

运行 `git ls-remote origin refs/heads/main`。成功后再推送；失败则保留本地提交并报告 SSH 密钥不可见，不伪造推送结果。

- [ ] **Step 3: Push the implementation commit**

运行 `git push origin main`，Expected: GitHub `main` 更新到本次修复提交。

- [ ] **Step 4: Synchronize the cloud server when access is available**

在服务器执行备份 `backend/data`、拉取 GitHub、重新构建并启动 Compose 服务；随后验证 `/api/health`、`/api/session`、访客额度、知识库权限和回答轨迹。若当前环境没有服务器 SSH 权限，输出可复制命令和明确阻塞点，不修改生产数据。
