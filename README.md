# ai-workspace-agent

一个正在开发中的企业知识助手项目。

## 项目目标

- 使用 Agent（智能代理）组织任务处理流程
- 使用 RAG（检索增强生成）实现企业知识库问答
- 使用 MCP（模型上下文协议）连接外部工具
- 使用 Skill（技能模块）扩展应用能力
- 使用 SQLite（轻量关系型数据库）持久化会话历史
- 后端采用 FastAPI（Python 后端框架）
- 前端计划采用 React（前端框架）

## 当前进度

- [x] 初始化项目目录和 Git（版本管理）仓库
- [x] 创建 FastAPI 后端应用
- [x] 实现后端健康检查接口 `GET /`
- [x] 初始化 React 前端
- [x] 实现 RAG、MCP、Skill 和 Agent 功能
- [x] 实现 Conversation Memory（会话记忆）与最近 20 条滑动窗口

## 会话接口

- `POST /api/chat`：首次请求省略 `conversation_id`，后端创建会话并返回 ID；后续请求携带该 ID。
- `GET /api/conversations/{conversation_id}/messages`：读取用户可见的历史消息。

当前项目尚未实现 Authentication（身份认证）和 User（用户）系统。因此，随机
`conversation_id` 只是资源标识符，不是权限控制；生产环境必须验证当前用户是否有权
访问对应会话，防止 IDOR（不安全的直接对象引用）。

## 测试与评测

在 `backend` 目录运行全部自动化测试：

```powershell
.\venv\Scripts\python.exe -m pytest -q
```

运行可重复的 Evaluation（评测）基线汇总：

```powershell
.\venv\Scripts\python.exe -m tests.evals.runner
```

该基线使用 Fake LLM（模拟大模型），用于验证评测管线、Tool Routing（工具路由）、
Skill Routing（技能路由）和 RAG No-hit（检索零命中）安全回归，不代表真实千问模型的
线上准确率。真实模型评测应单独运行并记录模型版本、提示词版本、数据集版本和成本。

聊天接口会返回不含用户正文和 Chain of Thought（模型内部思维过程）的 `trace`，包括
请求 ID、总耗时、Agent 步骤、工具耗时、RAG 命中情况和 LLM 调用耗时。前端开发面板
默认折叠显示这些信息。

## 启动后端

在 `backend` 目录创建并激活 Python 虚拟环境，安装依赖后运行：
激活：

```powershell
.\venv\Scripts\Activate.ps1
```

```powershell
python -m uvicorn app.main:app --reload
```

启动成功后可访问：

- 接口地址：http://127.0.0.1:8000/
- API（接口）文档：http://127.0.0.1:8000/docs
