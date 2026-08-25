# ai-workspace-agent

一个正在开发中的企业知识助手项目。

## 项目目标

- 使用 Agent（智能代理）组织任务处理流程
- 使用 RAG（检索增强生成）实现企业知识库问答
- 使用 MCP（模型上下文协议）连接外部工具
- 使用 Skill（技能模块）扩展应用能力
- 后端采用 FastAPI（Python 后端框架）
- 前端计划采用 React（前端框架）

## 当前进度

- [x] 初始化项目目录和 Git（版本管理）仓库
- [x] 创建 FastAPI 后端应用
- [x] 实现后端健康检查接口 `GET /`
- [ ] 初始化 React 前端
- [ ] 实现 RAG、MCP 和 Agent 功能

## 启动后端

在 `backend` 目录创建并激活 Python 虚拟环境，安装依赖后运行：

```powershell
python -m uvicorn app.main:app --reload
```

启动成功后可访问：

- 接口地址：http://127.0.0.1:8000/
- API（接口）文档：http://127.0.0.1:8000/docs
