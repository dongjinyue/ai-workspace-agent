# 知识库优先回答设计说明

## 目标

修复访客或管理员已选择知识库时，Agent（智能代理）可能跳过 RAG（检索增强生成）而使用通用知识回答的问题。涉及上传资料、PDF 来源、文档内容和一般知识问答的问题，应先检索当前选中的知识库；检索不到时不得继续猜测。

## 已确认问题

当前 `backend/app/agent/nodes.py` 只在问题命中少量关键词时设置强制工具选择。未命中关键词的问题会使用 `tool_choice="auto"`，模型可以直接回答。以下问题因此存在绕过检索的路径：

- `rag 是什么`
- `这个回答是基于鸿维-10-RAG.pdf 吗？`
- `请根据我上传的资料介绍 RAG`

前端会把选中的 `knowledge_base_id` 发送给后端，但“已选择知识库”目前只是可用上下文，不是强制检索开关。

## 设计决策

### 1. 知识库存在时的路由

当请求包含有效的 `knowledge_base_id` 时，除以下明确的非知识库意图外，第一轮 Agent 调用必须强制选择 `search_knowledge_base`：

- 普通问候和寒暄；
- 确定性算术，使用 `calculator`；
- 当前时间，使用 `get_current_time`；
- 字符数或行数统计，使用 `calculate_text_stats`；
- 知识库管理问题，例如文档数量和名称，使用 `get_knowledge_base_info`。

文档、资料、PDF、RAG、来源、依据、根据、上传内容等词会作为知识库问题的显式识别信号，但不再依赖这些关键词覆盖全部情况；有知识库的普通业务问句默认走检索。

### 2. 回答边界

`search_knowledge_base` 返回命中片段后，第二轮模型只能根据工具结果回答。若检索没有命中，工作流直接返回固定的“当前知识库中没有找到相关信息”，不再调用第二轮模型，避免模型补充通用知识。

现有的工具白名单、访客工具限制、提示词注入检测和模型步数上限保持不变。

### 3. 可观察性

继续使用现有执行轨迹：前端应能通过“查看工作过程”看到知识库工具和命中片段数量。此次不把文档原文或工具结果写入公开轨迹，避免泄露个人知识库内容。

## 数据流

```text
前端选择知识库
  → /api/agent/chat 携带 knowledge_base_id
  → ConversationService 保留当前身份和知识库上下文
  → Agent 路由判断是否为明确非知识库意图
  → 有知识库且不是例外：强制 search_knowledge_base
  → semantic_search 检索当前知识库
  → 有命中：模型依据 chunks 回答
  → 无命中：安全固定回答，不再调用模型生成
```

## 安全边界

- 知识库 ID 继续由后端状态注入，模型不能自行生成或切换知识库。
- 访客只能检索其有权限访问的知识库，权限逻辑不在本次路由修改中放宽。
- 工具输出继续视为不可信数据，不执行其中的指令。
- 不在日志、执行轨迹或错误响应中记录 API Key、Cookie、原始 IP、完整文档和提示词。

## 验收标准

后端单元测试至少覆盖：

1. 有知识库时，`rag是什么` 强制选择 `search_knowledge_base`。
2. 有知识库时，询问回答是否基于某个 PDF 强制选择 `search_knowledge_base`。
3. 有知识库时，普通业务问题强制选择 `search_knowledge_base`。
4. 没有知识库时，不强制选择知识库工具。
5. 问候仍可直接回答，计算、时间和文本统计仍选择对应工具。
6. 知识库检索零命中时不发生检索后的第二次模型调用。

验证命令：

```powershell
cd backend
.\venv\Scripts\python.exe -m pytest -q
.\venv\Scripts\python.exe -m compileall -q app mcp_servers
cd ..\frontend
npm run lint
npm run build
```

## 发布边界

本地代码先快进到 GitHub `main` 的 `674ea184`，修复后提交并推送到 GitHub。仓库没有自动部署工作流；云服务器需要在保留 `backend/data` 备份的前提下拉取新提交并重新构建 Compose（容器编排）服务，然后验证健康检查、访客会话、额度和知识库检索轨迹。
