import { useEffect, useRef, useState } from "react";
import { getAccessToken, isSupabaseConfigured, signInAdmin, signOutAdmin } from "./auth.js";
import { formatQuotaStatus } from "./usage.js";
import "./App.css";
import "./AgentStatus.css";
import "./ExecutionTrace.css";
import "./Auth.css";
import "./KnowledgeBase.css";

const API = import.meta.env.VITE_API_BASE_URL || (import.meta.env.PROD ? "/backend" : "http://127.0.0.1:8000");

async function request(path, options = {}) {
  const headers = new Headers(options.headers || {});
  const token = await getAccessToken();
  if (token) headers.set("Authorization", `Bearer ${token}`);
  const response = await fetch(`${API}${path}`, { ...options, headers, credentials: "include" });
  if (response.status === 204) return null;
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    const error = new Error(data.detail || (response.status === 429 ? "今日访客额度已用完，请稍后再试。" : "请求失败，请稍后重试。"));
    error.status = response.status;
    error.retryAfter = response.headers.get("Retry-After");
    throw error;
  }
  return data;
}

function formatTime(value) {
  if (!value) return "";
  return new Intl.DateTimeFormat("zh-CN", { hour: "2-digit", minute: "2-digit", second: "2-digit" }).format(new Date(value));
}

function formatDuration(milliseconds) {
  if (milliseconds == null) return "";
  return milliseconds < 1000 ? `${Math.round(milliseconds)} 毫秒` : `${(milliseconds / 1000).toFixed(1)} 秒`;
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
        {trace.rag?.hit && <li><i>{tools.length + 2}</i><div><strong>检索知识库</strong><small>命中 {trace.rag.results} 个相关片段</small></div></li>}
        <li><i>✓</i><div><strong>生成并检查回答</strong><small>模型调用 {trace.llm_calls} 次 · 模型耗时 {formatDuration(trace.llm_duration_ms)}</small></div></li>
      </ol>
      <div className="trace-total"><span>总耗时</span><strong>{formatDuration(trace.duration_ms)}</strong></div>
    </div>
  </details>;
}

function AdminLoginDialog({ open, onClose, onAuthenticated }) {
  const dialogRef = useRef(null);
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    const dialog = dialogRef.current;
    if (!dialog) return;
    if (open && !dialog.open) dialog.showModal();
    if (!open && dialog.open) dialog.close();
  }, [open]);

  async function submit(event) {
    event.preventDefault();
    if (submitting) return;
    setSubmitting(true);
    setError("");
    try {
      await signInAdmin(email.trim(), password);
      const session = await request("/api/session");
      if (session.role !== "admin") {
        await signOutAdmin();
        throw new Error("此账号没有管理员权限，已安全返回访客模式。");
      }
      setPassword("");
      onAuthenticated(session);
    } catch (loginError) {
      setError(loginError.status === 503 ? loginError.message : loginError.message || "登录失败，请检查账号信息后重试。");
    } finally {
      setSubmitting(false);
    }
  }

  return <dialog className="admin-dialog" ref={dialogRef} onCancel={(event) => { event.preventDefault(); onClose(); }} onClose={onClose} aria-labelledby="admin-login-title">
    <button className="dialog-close" type="button" onClick={onClose} aria-label="关闭管理员登录">×</button>
    <div className="auth-mark" aria-hidden="true">AI</div>
    <p className="dialog-eyebrow">ADMIN ACCESS</p>
    <h2 id="admin-login-title">管理员登录</h2>
    <p className="dialog-intro">访客可直接体验工作区。管理员使用网站已有的 Supabase 账号登录。</p>
    {!isSupabaseConfigured && <p className="config-notice" role="status">管理员登录尚未配置；访客体验不受影响。</p>}
    <form onSubmit={submit} noValidate>
      <label htmlFor="admin-email">邮箱</label>
      <input id="admin-email" type="email" autoComplete="username" value={email} onChange={(event) => setEmail(event.target.value)} required autoFocus />
      <label htmlFor="admin-password">密码</label>
      <input id="admin-password" type="password" autoComplete="current-password" value={password} onChange={(event) => setPassword(event.target.value)} required />
      {error && <p className="dialog-error" role="alert">{error}</p>}
      <button className="dialog-submit" type="submit" disabled={!isSupabaseConfigured || !email.trim() || !password || submitting} aria-busy={submitting}>
        {submitting ? "正在验证…" : "登录管理工作区"}
      </button>
    </form>
    <small className="dialog-footnote">不提供访客注册。管理员身份由服务器验证，浏览器无法自行提升权限。</small>
  </dialog>;
}

function ConfirmDialog({ confirmation, onClose }) {
  const dialogRef = useRef(null);
  const [submitting, setSubmitting] = useState(false);
  useEffect(() => {
    const dialog = dialogRef.current;
    if (!dialog) return;
    if (confirmation && !dialog.open) dialog.showModal();
    if (!confirmation && dialog.open) dialog.close();
  }, [confirmation]);

  async function confirm() {
    if (submitting) return;
    setSubmitting(true);
    try {
      const completed = await confirmation?.onConfirm();
      if (completed !== false) onClose();
    } finally {
      setSubmitting(false);
    }
  }

  return <dialog className="admin-dialog confirm-dialog" ref={dialogRef} onCancel={(event) => { event.preventDefault(); onClose(); }} onClose={onClose} aria-labelledby="confirm-title">
    <button className="dialog-close" type="button" onClick={onClose} aria-label="取消操作">×</button>
    <p className="dialog-eyebrow">请确认操作</p>
    <h2 id="confirm-title">{confirmation?.title}</h2>
    <p className="dialog-intro">{confirmation?.description}</p>
    <div className="confirm-actions"><button type="button" autoFocus onClick={onClose}>暂不删除</button><button className="confirm-danger" type="button" onClick={confirm} disabled={submitting} aria-busy={submitting}>{submitting ? "正在处理…" : confirmation?.actionLabel}</button></div>
  </dialog>;
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
  const [session, setSession] = useState(undefined);
  const [loginOpen, setLoginOpen] = useState(false);
  const [confirmation, setConfirmation] = useState(null);
  const [backendOnline, setBackendOnline] = useState(false);
  const [knowledgeBases, setKnowledgeBases] = useState([]);
  const [knowledgeBaseId, setKnowledgeBaseId] = useState("");
  const [knowledgeDocuments, setKnowledgeDocuments] = useState([]);
  const bottomRef = useRef(null);
  const booted = useRef(false);

  const role = session?.role;
  const isAdmin = role === "admin";
  const quotaCopy = formatQuotaStatus(session?.quota);
  const conversationStorageKey = isAdmin ? "conversation_id" : "guest_conversation_id";
  const knowledgeStorageKey = isAdmin ? "knowledge_base_id" : "guest_knowledge_base_id";

  async function refresh(preferredId, activeRole = role) {
    const data = await request("/api/conversations");
    setConversations(data.conversations || []);
    const key = activeRole === "admin" ? "conversation_id" : "guest_conversation_id";
    const saved = preferredId || localStorage.getItem(key);
    return data.conversations?.some((item) => item.id === saved) ? saved : data.conversations?.[0]?.id || "";
  }

  async function restoreKnowledgeBase(activeRole) {
    const data = await request("/api/knowledge-bases?limit=100");
    const bases = data.knowledge_bases || [];
    setKnowledgeBases(bases);
    const key = activeRole === "admin" ? "knowledge_base_id" : "guest_knowledge_base_id";
    const savedId = localStorage.getItem(key);
    const selected = bases.find((item) => item.id === savedId) || bases[0];
    if (selected) {
      setKnowledgeBaseId(selected.id);
      setKnowledgeDocuments(selected.documents || []);
      localStorage.setItem(key, selected.id);
    } else {
      setKnowledgeBaseId("");
      setKnowledgeDocuments([]);
      localStorage.removeItem(key);
    }
  }

  async function select(id, activeRole = role) {
    if (!id) return;
    const key = activeRole === "admin" ? "conversation_id" : "guest_conversation_id";
    setConversationId(id);
    localStorage.setItem(key, id);
    setSidebarOpen(false);
    const data = await request(`/api/conversations/${id}/messages`);
    setMessages(data.messages || []);
  }

  async function loadWorkspace(currentSession) {
    setSession(currentSession);
    const activeRole = currentSession.role;
    await restoreKnowledgeBase(activeRole);
    const id = await refresh(undefined, activeRole);
    if (id) await select(id, activeRole);
    else {
      setConversationId("");
      setMessages([]);
    }
  }

  // 一次性启动流程会复用当前最新的恢复函数，同时避免依赖变化重复引导会话。
  const loadWorkspaceRef = useRef(loadWorkspace);
  useEffect(() => {
    loadWorkspaceRef.current = loadWorkspace;
  });

  useEffect(() => {
    if (booted.current) return;
    booted.current = true;
    async function restore() {
      try {
        const health = await request("/api/health");
        setBackendOnline(health.status === "ok");
        const currentSession = await request("/api/session");
        setSession(currentSession);
        const [knowledgeData, conversationData] = await Promise.all([
          request("/api/knowledge-bases?limit=100"),
          request("/api/conversations"),
        ]);
        const bases = knowledgeData.knowledge_bases || [];
        const conversationsForRole = conversationData.conversations || [];
        const baseKey = currentSession.role === "admin" ? "knowledge_base_id" : "guest_knowledge_base_id";
        const conversationKey = currentSession.role === "admin" ? "conversation_id" : "guest_conversation_id";
        const savedBase = localStorage.getItem(baseKey);
        const selectedBase = bases.find((item) => item.id === savedBase) || bases[0];
        const savedConversation = localStorage.getItem(conversationKey);
        const selectedConversation = conversationsForRole.find((item) => item.id === savedConversation) || conversationsForRole[0];
        setKnowledgeBases(bases);
        setKnowledgeBaseId(selectedBase?.id || "");
        setKnowledgeDocuments(selectedBase?.documents || []);
        if (selectedBase) localStorage.setItem(baseKey, selectedBase.id);
        else localStorage.removeItem(baseKey);
        setConversations(conversationsForRole);
        if (selectedConversation) {
          setConversationId(selectedConversation.id);
          localStorage.setItem(conversationKey, selectedConversation.id);
          const history = await request(`/api/conversations/${selectedConversation.id}/messages`);
          setMessages(history.messages || []);
        }
        else { setConversationId(""); setMessages([]); }
      } catch (loadError) {
        // 旧的或已过期的管理员会话不能卡住访客入口；清除本地认证后重新取得访客身份。
        if (loadError.status === 401 && await getAccessToken()) {
          try {
            await signOutAdmin();
            const guestSession = await request("/api/session");
            await loadWorkspaceRef.current(guestSession);
            setError("管理员登录状态已过期，已切回访客模式。");
            return;
          } catch { /* 访客会话也不可用时，使用下面的通用错误状态。 */ }
        }
        setBackendOnline(false);
        setSession(null);
        setError(loadError.message);
      }
    }
    restore();
  }, []);

  useEffect(() => { bottomRef.current?.scrollIntoView({ behavior: "smooth", block: "end" }); }, [messages, loading]);

  async function createConversation() {
    try {
      const item = await request("/api/conversations", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ title: "新会话" }) });
      await refresh(item.id);
      await select(item.id);
      setQuestion("");
      setError("");
    } catch (requestError) { setError(requestError.message); }
  }

  async function onAdminAuthenticated(newSession) {
    setLoginOpen(false);
    setError("");
    try {
      await loadWorkspace(newSession);
    } catch (loadError) {
      setError(loadError.message);
    }
  }

  async function logout() {
    try {
      await signOutAdmin();
      setLoginOpen(false);
      setKnowledgeBaseId("");
      setKnowledgeBases([]);
      setKnowledgeDocuments([]);
      setConversationId("");
      setMessages([]);
      const guestSession = await request("/api/session");
      await loadWorkspace(guestSession);
      setError("");
    } catch (logoutError) { setError(logoutError.message); }
  }

  async function renameConversation(id) {
    const title = editing?.title.trim();
    if (!title) return setEditing(null);
    try {
      await request(`/api/conversations/${id}`, { method: "PATCH", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ title }) });
      setEditing(null);
      await refresh(id);
    } catch (requestError) { setError(requestError.message); }
  }

  async function deleteConversation(id) {
    const item = conversations.find((conversation) => conversation.id === id);
    setConfirmation({
      title: "删除这段对话？",
      description: `“${item?.title || "这段对话"}”及其全部消息将被永久删除，无法恢复。`,
      actionLabel: "永久删除对话",
      onConfirm: async () => {
        try {
      await request(`/api/conversations/${id}`, { method: "DELETE" });
      const nextId = await refresh();
      if (nextId) await select(nextId);
      else { setConversationId(""); setMessages([]); localStorage.removeItem(conversationStorageKey); }
      setError("");
          return true;
        } catch (requestError) { setError(requestError.message); return false; }
      },
    });
  }

  async function sendMessage() {
    const content = question.trim();
    if (!content || loading || (session?.quota && session.quota.remaining <= 0)) return;
    setQuestion("");
    setError("");
    setLoading(true);
    setMessages((items) => [...items, { role: "user", content, created_at: new Date().toISOString() }]);
    try {
      const data = await request("/api/agent/chat", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ message: content, conversation_id: conversationId || null, knowledge_base_id: knowledgeBaseId || null }) });
      setConversationId(data.conversation_id);
      localStorage.setItem(conversationStorageKey, data.conversation_id);
      setMessages((items) => [...items, { role: "assistant", content: data.answer, created_at: data.trace?.completed_at, trace: data.trace }]);
      await refresh(data.conversation_id);
      const updatedSession = await request("/api/session");
      setSession(updatedSession);
    } catch (requestError) {
      setMessages((items) => items.slice(0, -1));
      setQuestion(content);
      if (requestError.status === 429) {
        try { setSession(await request("/api/session")); } catch { /* 保留原本的限额反馈 */ }
      }
      setError(requestError.message);
    } finally { setLoading(false); }
  }

  async function uploadDocument(event) {
    const selectedFiles = Array.from(event.target.files || []);
    if (!selectedFiles.length) return;
    setUploading(true);
    setError("");
    try {
      const form = new FormData();
      selectedFiles.forEach((selectedFile) => form.append("files", selectedFile));
      const selectedKnowledgeBase = knowledgeBases.find((item) => item.id === knowledgeBaseId);
      if (knowledgeBaseId && (isAdmin || selectedKnowledgeBase?.can_edit)) form.append("knowledge_base_id", knowledgeBaseId);
      const data = await request("/api/documents/upload", { method: "POST", body: form });
      setKnowledgeBaseId(data.knowledge_base_id);
      setKnowledgeDocuments(data.documents || []);
      localStorage.setItem(isAdmin ? "knowledge_base_id" : "guest_knowledge_base_id", data.knowledge_base_id);
      await restoreKnowledgeBase(isAdmin ? "admin" : "guest");
    } catch (requestError) { setError(requestError.message); }
    finally { setUploading(false); event.target.value = ""; }
  }

  async function removeKnowledgeBase() {
    if (!knowledgeBaseId) return;
    setConfirmation({
      title: "删除整个知识库？",
      description: "知识库中的全部文档和向量数据都会被永久删除，无法恢复。",
      actionLabel: "永久删除知识库",
      onConfirm: async () => {
        try {
      await request(`/api/knowledge-bases/${knowledgeBaseId}`, { method: "DELETE" });
      setKnowledgeBaseId("");
      setKnowledgeDocuments([]);
      setKnowledgeBases((items) => items.filter((item) => item.id !== knowledgeBaseId));
      localStorage.removeItem(knowledgeStorageKey);
          return true;
        } catch (requestError) { setError(requestError.message); return false; }
      },
    });
  }

  async function removeKnowledgeDocument(document) {
    if (!document.id) return;
    setConfirmation({
      title: "删除这份文档？",
      description: `“${document.filename}”会从知识库移除；其他文档会保留。`,
      actionLabel: "删除文档",
      onConfirm: async () => {
        try {
      const data = await request(`/api/knowledge-bases/${knowledgeBaseId}/documents/${document.id}`, { method: "DELETE" });
      setKnowledgeDocuments(data?.documents || []);
          return true;
        } catch (requestError) { setError(requestError.message); return false; }
      },
    });
  }

  const activeConversation = conversations.find((item) => item.id === conversationId);
  const selectedKnowledgeBase = knowledgeBases.find((item) => item.id === knowledgeBaseId);
  const canEditKnowledgeBase = isAdmin || selectedKnowledgeBase?.can_edit === true;
  const exhausted = !isAdmin && session?.quota?.remaining <= 0;

  return <main className="app-shell">
    <button className="mobile-menu" onClick={() => setSidebarOpen(true)} aria-label="打开会话列表" aria-expanded={sidebarOpen}>☰</button>
    {sidebarOpen && <button className="sidebar-mask" onClick={() => setSidebarOpen(false)} aria-label="关闭会话列表" />}
    <aside className={`sidebar ${sidebarOpen ? "open" : ""}`} aria-label="工作区侧栏">
      <div className="brand"><span className="brand-mark" aria-label="人工智能"><b>人工</b><b>智能</b></span><div><strong>Workspace</strong><small>智能工作台</small></div></div>
      <button className="new-chat" onClick={createConversation} disabled={!session}>＋ 新建会话</button>
      <p className="section-label">最近会话</p>
      <div className="conversation-list">
        {conversations.map((item) => <div className={`conversation-item ${item.id === conversationId ? "active" : ""}`} key={item.id}>
          {editing?.id === item.id ? <input aria-label="会话名称" autoFocus value={editing.title} onChange={(event) => setEditing({ id: item.id, title: event.target.value })} onBlur={() => renameConversation(item.id)} onKeyDown={(event) => { if (event.key === "Enter") renameConversation(item.id); if (event.key === "Escape") setEditing(null); }} /> :
            <button className="conversation-main" onClick={() => select(item.id)}><b aria-hidden="true">◇</b><span><strong>{item.title}</strong><small>{item.message_count} 条消息</small></span></button>}
          {isAdmin && <div className="actions"><button title="重命名" aria-label={`重命名 ${item.title}`} onClick={() => setEditing({ id: item.id, title: item.title })}>✎</button><button title="删除" aria-label={`删除 ${item.title}`} onClick={() => deleteConversation(item.id)}>×</button></div>}
        </div>)}
        {session === undefined && <p className="empty">正在连接工作区…</p>}
        {session === null && <p className="empty">服务暂时不可用，请稍后重试。</p>}
        {session && !conversations.length && <p className="empty">还没有会话，点击上方按钮开始吧</p>}
      </div>
      <section className="knowledge" aria-labelledby="knowledge-title">
        <span id="knowledge-title">{isAdmin ? "我的知识库" : "个人与共享库"}<i>{knowledgeBases.length ? `${knowledgeBases.length} 个` : "暂无"}</i></span>
        {knowledgeBases.length > 0 && <label className="knowledge-select-label" htmlFor="knowledge-base-select">选择本次问答使用的知识库</label>}
        {knowledgeBases.length > 0 && <select id="knowledge-base-select" value={knowledgeBaseId} onChange={(event) => {
          const selected = knowledgeBases.find((item) => item.id === event.target.value);
          setKnowledgeBaseId(event.target.value);
          setKnowledgeDocuments(selected?.documents || []);
          if (event.target.value) localStorage.setItem(knowledgeStorageKey, event.target.value);
          else localStorage.removeItem(knowledgeStorageKey);
        }}>
          <option value="">＋ 新建知识库</option>
          {knowledgeBases.map((item) => <option key={item.id} value={item.id}>{!isAdmin && !item.can_edit ? "共享 · " : ""}{item.title || item.name || item.documents?.[0]?.filename || item.id}</option>)}
        </select>}
        {uploading ? <small role="status">正在解析并建立索引…</small> : knowledgeDocuments.length ? <ul className="knowledge-documents">{knowledgeDocuments.map((document, index) => <li key={document.id || `${document.filename}-${index}`} title={document.filename}><span>{document.filename}</span>{canEditKnowledgeBase && document.id && <button type="button" title={`删除 ${document.filename}`} aria-label={`删除 ${document.filename}`} onClick={() => removeKnowledgeDocument(document)}>×</button>}</li>)}</ul> : <small>{isAdmin ? "尚未上传知识文档" : "上传文件，创建只属于当前访客会话的个人知识库"}</small>}
        <div className="knowledge-actions"><label>{uploading ? "处理中" : knowledgeBaseId && canEditKnowledgeBase ? "继续添加" : "创建个人库"}<input type="file" multiple accept=".txt,.md,.docx,.pdf,text/plain,text/markdown,application/pdf,application/vnd.openxmlformats-officedocument.wordprocessingml.document" onChange={uploadDocument} disabled={uploading} /></label>{knowledgeBaseId && canEditKnowledgeBase && <button onClick={removeKnowledgeBase}>{isAdmin ? "清空" : "删除我的库"}</button>}</div>
        <small className="knowledge-hint">{isAdmin ? "管理员可管理全部知识库；游客只能使用自己的文件" : canEditKnowledgeBase ? "个人库只对当前访客可见；清除此网站 Cookie 或更换浏览器后将无法找回" : "共享库仅可检索；上传会另建私人库"}</small>
      </section>
    </aside>
    <section className="chat-panel">
      <header>
        <div className="chat-heading"><h1>{activeConversation?.title || "AI Workspace Agent"}</h1><p>{conversationId ? "会话已保存至当前身份" : "开始一个新的 AI 工作会话"}</p></div>
        <div className="header-actions"><span className="agent-status">Agent 模式</span><span className={backendOnline ? "online" : "offline"} aria-live="polite">● {backendOnline ? "在线" : "离线"}</span>
          {isAdmin ? <><span className="role-badge">管理员</span><button className="account-button" onClick={logout}>退出</button></> : <><span className="role-badge guest-badge">访客 · {session ? quotaCopy : "正在读取额度…"}</span><button className="account-button" onClick={() => setLoginOpen(true)}>管理员登录</button></>}
        </div>
      </header>
      <div className="messages" aria-live="polite" aria-busy={loading}>
        {!messages.length && <div className="welcome"><span aria-hidden="true">✦</span><p className="welcome-eyebrow">AI WORKSPACE</p><h2>{isAdmin ? "欢迎回来" : "今天想解决什么问题？"}</h2><p>{isAdmin ? "管理知识库并开始新的工作对话。" : "直接开始体验 AI 工作区。访客每天可免费提问 10 次。"}</p>
          {!isAdmin && <div className="guest-limit-note"><strong>访客体验说明</strong><span>对话仅对当前浏览器会话可见，每日次数于北京时间午夜重置。</span></div>}
        </div>}
        {messages.map((message, index) => <article className={`message ${message.role}`} key={`${message.role}-${index}`}>
          {message.role === "assistant" && <div className="avatar ai" aria-hidden="true">AI</div>}
          <div className="message-content"><div className="message-meta"><span>{message.role === "assistant" ? "AI 助手" : "你"}</span><time>{formatTime(message.trace?.completed_at || message.created_at)}{message.trace?.duration_ms != null ? ` · 用时 ${formatDuration(message.trace.duration_ms)}` : ""}</time></div>{message.role === "assistant" && <ExecutionTrace trace={message.trace} />}<div className="bubble">{message.content}</div></div>
          {message.role === "user" && <div className="avatar user" aria-hidden="true">你</div>}
        </article>)}
        {loading && <article className="message assistant" aria-label="AI 正在生成回答"><div className="avatar ai" aria-hidden="true">AI</div><div className="typing" role="status"><i /><i /><i /><span className="sr-only">正在生成回答</span></div></article>}
        <div ref={bottomRef} />
      </div>
      <footer>
        <div className="quota-banner" role="status"><span>{session ? quotaCopy : "正在读取使用额度…"}</span>{!isAdmin && session?.quota && <span>IP 保护：每分钟 5 次，全天 30 次</span>}</div>
        {error && <div className="error" role="alert">{error}</div>}
        <div className="composer"><textarea className="resize-none" aria-label="输入你的问题" rows="1" maxLength="4000" value={question} onChange={(event) => setQuestion(event.target.value)} onKeyDown={(event) => { if (event.key === "Enter" && !event.shiftKey && !event.nativeEvent.isComposing) { event.preventDefault(); sendMessage(); } }} placeholder={exhausted ? "今日访客提问次数已用完" : "输入你的问题…"} disabled={!session || loading || exhausted} /><button aria-label="发送问题" onClick={sendMessage} disabled={!question.trim() || loading || !session || exhausted} aria-busy={loading}>↑</button></div>
        <small>Enter 发送 · Shift + Enter 换行 · {isAdmin ? "管理员模式不受访客次数限制" : "每日 10 次 · 北京时间午夜重置"}</small>
      </footer>
    </section>
    <AdminLoginDialog open={loginOpen} onClose={() => setLoginOpen(false)} onAuthenticated={onAdminAuthenticated} />
    <ConfirmDialog confirmation={confirmation} onClose={() => setConfirmation(null)} />
  </main>;
}

export default App;
