import { useEffect, useRef, useState } from "react";
import "./App.css";
import "./AgentStatus.css";
import "./ExecutionTrace.css";
import "./Auth.css";
import "./KnowledgeBase.css";

const API = import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000";
async function request(path, options) {
  const token = sessionStorage.getItem("access_token");
  const headers = new Headers(options?.headers || {});
  if (token) headers.set("Authorization", `Bearer ${token}`);
  const response = await fetch(`${API}${path}`, { ...options, headers });
  if (response.status === 204) return null;
  const data = await response.json();
  if (!response.ok) throw new Error(data.detail || "请求失败，请稍后重试");
  return data;
}

function formatTime(value) {
  if (!value) return "";
  return new Intl.DateTimeFormat("zh-CN", {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  }).format(new Date(value));
}

function formatDuration(milliseconds) {
  if (milliseconds == null) return "";
  return milliseconds < 1000
    ? `${Math.round(milliseconds)} 毫秒`
    : `${(milliseconds / 1000).toFixed(1)} 秒`;
}

function ExecutionTrace({ trace }) {
  if (!trace) return null;
  const tools = trace.tools || [];
  return <details className="execution-trace">
    <summary><span>⌁</span> 查看工作过程 <b>⌄</b></summary>
    <div className="trace-panel">
      <p className="trace-note">这里展示执行轨迹，不包含模型的隐藏思维链。</p>
      <ol>
        <li><i>1</i><div><strong>理解并规划任务</strong><small>Agent 共执行 {trace.steps} 个步骤</small></div></li>
        {tools.map((tool, index) => <li key={`${tool.name}-${index}`}><i>{index + 2}</i><div><strong>调用工具 · {tool.name}</strong><small>{tool.source === "mcp" ? `MCP 服务${tool.server ? ` · ${tool.server}` : ""}` : "本地安全工具"} · {formatDuration(tool.duration_ms)}</small></div></li>)}
        {trace.rag?.hit && <li><i>{tools.length + 2}</i><div><strong>检索企业知识库</strong><small>命中 {trace.rag.results} 个相关片段</small></div></li>}
        <li><i>✓</i><div><strong>生成并检查回答</strong><small>模型调用 {trace.llm_calls} 次 · 模型耗时 {formatDuration(trace.llm_duration_ms)}</small></div></li>
      </ol>
      <div className="trace-total"><span>总耗时</span><strong>{formatDuration(trace.duration_ms)}</strong></div>
    </div>
  </details>;
}

function App() {
  const [conversations, setConversations] = useState([]);
  const [conversationId, setConversationId] = useState("");
  const [messages, setMessages] = useState([]);
  const [question, setQuestion] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [editing, setEditing] = useState(null);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [authRequired, setAuthRequired] = useState(null);
  const [accessToken, setAccessToken] = useState(() => sessionStorage.getItem("access_token") || "");
  const [tokenInput, setTokenInput] = useState("");
  const [backendOnline, setBackendOnline] = useState(false);
  const [knowledgeBaseId, setKnowledgeBaseId] = useState(() => localStorage.getItem("knowledge_base_id") || "");
  const bottomRef = useRef(null);

  async function refresh(preferredId) {
    const data = await request("/api/conversations");
    setConversations(data.conversations);
    const saved = preferredId || localStorage.getItem("conversation_id");
    return data.conversations.some((item) => item.id === saved) ? saved : data.conversations[0]?.id || "";
  }

  async function restoreKnowledgeBase() {
    if (localStorage.getItem("knowledge_base_id")) return;
    const data = await request("/api/knowledge-bases?limit=1");
    const latest = data.knowledge_bases?.[0];
    if (latest) {
      setKnowledgeBaseId(latest.id);
      localStorage.setItem("knowledge_base_id", latest.id);
    }
  }

  async function select(id) {
    setConversationId(id);
    localStorage.setItem("conversation_id", id);
    setSidebarOpen(false);
    const data = await request(`/api/conversations/${id}/messages`);
    setMessages(data.messages || []);
  }

  useEffect(() => {
    let active = true;
    async function restore() {
      try {
        const health = await request("/api/health");
        setBackendOnline(health.status === "ok");
        setAuthRequired(Boolean(health.auth_required));
        if (health.auth_required && !sessionStorage.getItem("access_token")) return;
        await restoreKnowledgeBase();
        const id = await refresh();
        if (active && id) await select(id);
      } catch (loadError) { if (active) { setBackendOnline(false); setError(loadError.message); } }
    }
    restore();
    return () => { active = false; };
  }, []);
  useEffect(() => { bottomRef.current?.scrollIntoView({ behavior: "smooth" }); }, [messages, loading]);

  async function createConversation() {
    try {
      const item = await request("/api/conversations", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ title: "新会话" }) });
      await refresh(item.id);
      await select(item.id);
      setQuestion("");
    } catch (e) { setError(e.message); }
  }

  async function unlockWorkspace(event) {
    event.preventDefault();
    const token = tokenInput.trim();
    if (!token) return;
    sessionStorage.setItem("access_token", token);
    try {
      const id = await refresh();
      setAccessToken(token);
      setTokenInput("");
      setError("");
      await restoreKnowledgeBase();
      if (id) await select(id);
    } catch (loginError) {
      sessionStorage.removeItem("access_token");
      setError(loginError.message);
    }
  }

  async function renameConversation(id) {
    const title = editing?.title.trim();
    if (!title) return setEditing(null);
    try {
      await request(`/api/conversations/${id}`, { method: "PATCH", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ title }) });
      setEditing(null);
      await refresh(id);
    } catch (e) { setError(e.message); }
  }

  async function deleteConversation(id) {
    if (!window.confirm("确定删除这个会话及其全部消息吗？此操作无法撤销。")) return;
    try {
      await request(`/api/conversations/${id}`, { method: "DELETE" });
      const nextId = await refresh();
      if (nextId) await select(nextId);
      else { setConversationId(""); setMessages([]); localStorage.removeItem("conversation_id"); }
    } catch (e) { setError(e.message); }
  }

  async function sendMessage() {
    const content = question.trim();
    if (!content || loading) return;
    setQuestion(""); setError(""); setLoading(true);
    setMessages((items) => [...items, { role: "user", content, created_at: new Date().toISOString() }]);
    try {
      const data = await request("/api/agent/chat", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ message: content, conversation_id: conversationId || null, knowledge_base_id: knowledgeBaseId || null }) });
      setConversationId(data.conversation_id);
      localStorage.setItem("conversation_id", data.conversation_id);
      setMessages((items) => [...items, { role: "assistant", content: data.answer, created_at: data.trace?.completed_at, trace: data.trace }]);
      await refresh(data.conversation_id);
    } catch (e) {
      setMessages((items) => items.slice(0, -1)); setQuestion(content); setError(e.message);
    } finally { setLoading(false); }
  }

  async function uploadDocument(event) {
    const file = event.target.files?.[0];
    if (!file) return;
    setUploading(true); setError("");
    try {
      const form = new FormData(); form.append("file", file);
      const data = await request("/api/documents/upload", { method: "POST", body: form });
      setKnowledgeBaseId(data.knowledge_base_id);
      localStorage.setItem("knowledge_base_id", data.knowledge_base_id);
    } catch (e) { setError(e.message); }
    finally { setUploading(false); event.target.value = ""; }
  }

  async function removeKnowledgeBase() {
    if (!knowledgeBaseId || !window.confirm("确定删除当前知识库及全部向量数据吗？")) return;
    try {
      await request(`/api/knowledge-bases/${knowledgeBaseId}`, { method: "DELETE" });
      setKnowledgeBaseId("");
      localStorage.removeItem("knowledge_base_id");
    } catch (deleteError) { setError(deleteError.message); }
  }

  const activeConversation = conversations.find((item) => item.id === conversationId);
  if (authRequired && !accessToken) {
    return <main className="auth-screen"><form onSubmit={unlockWorkspace}><div className="auth-mark">AI</div><h1>AI Workspace Agent</h1><p>此工作区已启用访问保护，请输入部署者提供的访问令牌。</p><input type="password" value={tokenInput} onChange={(event) => setTokenInput(event.target.value)} placeholder="访问令牌" autoFocus /><button type="submit" disabled={!tokenInput.trim()}>进入工作区</button>{error && <small>{error}</small>}</form></main>;
  }
  return <main className="app-shell">
    <button className="mobile-menu" onClick={() => setSidebarOpen(true)} aria-label="打开会话列表">☰</button>
    {sidebarOpen && <button className="sidebar-mask" onClick={() => setSidebarOpen(false)} aria-label="关闭会话列表" />}
    <aside className={`sidebar ${sidebarOpen ? "open" : ""}`}>
      <div className="brand"><span>AI</span><div><strong>Workspace</strong><small>智能工作台</small></div></div>
      <button className="new-chat" onClick={createConversation}>＋ 新建会话</button>
      <p className="section-label">最近会话</p>
      <div className="conversation-list">
        {conversations.map((item) => <div className={`conversation-item ${item.id === conversationId ? "active" : ""}`} key={item.id}>
          {editing?.id === item.id ? <input autoFocus value={editing.title} onChange={(e) => setEditing({ id: item.id, title: e.target.value })} onBlur={() => renameConversation(item.id)} onKeyDown={(e) => { if (e.key === "Enter") renameConversation(item.id); if (e.key === "Escape") setEditing(null); }} /> :
            <button className="conversation-main" onClick={() => select(item.id)}><b>◇</b><span><strong>{item.title}</strong><small>{item.message_count} 条消息</small></span></button>}
          <div className="actions"><button title="重命名" onClick={() => setEditing({ id: item.id, title: item.title })}>✎</button><button title="删除" onClick={() => deleteConversation(item.id)}>×</button></div>
        </div>)}
        {!conversations.length && <p className="empty">还没有会话，点击上方按钮开始吧</p>}
      </div>
      <div className="knowledge"><span>企业知识库 <i>{knowledgeBaseId ? "已连接" : "未连接"}</i></span><small>{uploading ? "正在上传…" : knowledgeBaseId ? `ID · ${knowledgeBaseId.slice(0, 8)}…` : "尚未上传知识文档"}</small><div className="knowledge-actions"><label>{uploading ? "处理中" : "上传 TXT"}<input type="file" accept=".txt,text/plain" onChange={uploadDocument} disabled={uploading} /></label>{knowledgeBaseId && <button onClick={removeKnowledgeBase}>删除</button>}</div></div>
    </aside>
    <section className="chat-panel">
      <header><div><h1>{activeConversation?.title || "AI Workspace Agent"}</h1><p>{conversationId ? "会话已持久保存" : "创建会话，开始探索"}</p></div><div className="header-actions"><span className="agent-status">Agent 模式</span><span className={backendOnline ? "online" : "offline"}>● {backendOnline ? "在线" : "离线"}</span></div></header>
      <div className="messages">
        {!messages.length && <div className="welcome"><span>✦</span><h2>有什么可以帮你？</h2><p>你可以询问工作问题，也可以上传企业文档，让我结合知识库回答。</p></div>}
        {messages.map((message, index) => <article className={`message ${message.role}`} key={`${message.role}-${index}`}>
          {message.role === "assistant" && <div className="avatar ai">AI</div>}
          <div className="message-content"><div className="message-meta"><span>{message.role === "assistant" ? "AI 助手" : "你"}</span><time>{formatTime(message.trace?.completed_at || message.created_at)}{message.trace?.duration_ms != null ? ` · 用时 ${formatDuration(message.trace.duration_ms)}` : ""}</time></div>{message.role === "assistant" && <ExecutionTrace trace={message.trace} />}<div className="bubble">{message.content}</div></div>
          {message.role === "user" && <div className="avatar user">你</div>}
        </article>)}
        {loading && <article className="message assistant"><div className="avatar ai">AI</div><div className="typing"><i /><i /><i /></div></article>}
        <div ref={bottomRef} />
      </div>
      <footer>{error && <div className="error">{error}</div>}<div className="composer"><textarea rows="1" maxLength="4000" value={question} onChange={(e) => setQuestion(e.target.value)} onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); sendMessage(); } }} placeholder="输入你的问题…" /><button onClick={sendMessage} disabled={!question.trim() || loading}>↑</button></div><small>Enter 发送 · Shift + Enter 换行 · 对话会自动保存</small></footer>
    </section>
  </main>;
}
export default App;
