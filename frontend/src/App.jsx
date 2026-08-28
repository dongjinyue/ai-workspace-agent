import { useEffect, useState } from "react";

const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000";

function App() {
  const [question, setQuestion] = useState("");
  const [messages, setMessages] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [uploading, setUploading] = useState(false);
  const [documentStatus, setDocumentStatus] = useState("尚未上传知识库文件");
  const [matchedChunks, setMatchedChunks] = useState(0);
  const [llmCalled, setLlmCalled] = useState(false);
  const [agentInfo, setAgentInfo] = useState(null);
  const [trace, setTrace] = useState(null);
  const [knowledgeBaseId, setKnowledgeBaseId] = useState(
    () => localStorage.getItem("knowledge_base_id") || "",
  );
  const [conversationId, setConversationId] = useState(
    () => localStorage.getItem("conversation_id") || "",
  );

  useEffect(() => {
    // 页面刷新后只用随机会话 ID 拉取历史；Local Storage 不保存密钥。
    const storedId = localStorage.getItem("conversation_id");
    if (!storedId) return;

    let cancelled = false;
    async function restoreConversation() {
      try {
        const response = await fetch(
          `${API_BASE_URL}/api/conversations/${storedId}/messages`,
        );
        const data = await response.json();
        if (!response.ok) throw new Error(data.detail || "读取会话历史失败");
        if (!cancelled) setMessages(data.messages || []);
      } catch (restoreError) {
        if (!cancelled) {
          localStorage.removeItem("conversation_id");
          setConversationId("");
          setError(restoreError.message);
        }
      }
    }
    restoreConversation();
    return () => {
      cancelled = true;
    };
  }, []);

  function newConversation() {
    // 清空当前 ID 后，下一条消息会由后端创建一个完全隔离的新会话。
    localStorage.removeItem("conversation_id");
    setConversationId("");
    setMessages([]);
    setQuestion("");
    setAgentInfo(null);
    setTrace(null);
    setError("");
  }

  function updateKnowledgeBaseId(value) {
    const normalizedValue = value.trim();
    setKnowledgeBaseId(normalizedValue);

    if (normalizedValue) {
      localStorage.setItem("knowledge_base_id", normalizedValue);
      setDocumentStatus("已连接到持久化知识库");
    } else {
      localStorage.removeItem("knowledge_base_id");
      setDocumentStatus("尚未上传知识库文件");
    }
  }

  async function uploadDocument(event) {
    const file = event.target.files?.[0];
    if (!file) return;

    if (file.size > 1024 * 1024) {
      setError("TXT 文件不能超过 1 MB");
      event.target.value = "";
      return;
    }

    setUploading(true);
    setError("");

    try {
      const formData = new FormData();
      formData.append("file", file);

      const response = await fetch(`${API_BASE_URL}/api/documents/upload`, {
        method: "POST",
        body: formData,
      });
      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || "文档上传失败");
      }

      updateKnowledgeBaseId(data.knowledge_base_id);
      setDocumentStatus(`已加载 ${data.filename}，共 ${data.chunks} 个文本块`);
    } catch (uploadError) {
      setDocumentStatus("知识库文件上传失败");
      setError(uploadError.message);
    } finally {
      setUploading(false);
      event.target.value = "";
    }
  }

  async function sendMessage() {
    if (!question.trim()) {
      setError("请输入问题");
      return;
    }

    setLoading(true);
    setMatchedChunks(0);
    setLlmCalled(false);
    setAgentInfo(null);
    setTrace(null);
    setError("");

    try {
      const response = await fetch(
        `${API_BASE_URL}/api/chat`,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            message: question,
            knowledge_base_id: knowledgeBaseId || null,
            conversation_id: conversationId || null,
          }),
        }
      );

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || "请求失败");
      }

      // 首次响应返回后端生成的 ID，后续请求继续携带它以恢复上下文。
      if (data.conversation_id !== conversationId) {
        setConversationId(data.conversation_id);
        localStorage.setItem("conversation_id", data.conversation_id);
      }
      setMessages((current) => [
        ...current,
        { role: "user", content: question.trim() },
        { role: "assistant", content: data.answer },
      ]);
      setQuestion("");
      setMatchedChunks(data.matched_chunks || 0);
      setLlmCalled(Boolean(data.llm_called));
      setAgentInfo(data.agent || null);
      setTrace(data.trace || null);
    } catch (requestError) {
      setError(requestError.message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <main>
      <h1>AI Workspace Agent</h1>

      <section>
        <h2>Conversation（会话）</h2>
        <p>Conversation ID：{conversationId || "将在首次发送消息时创建"}</p>
        <button type="button" onClick={newConversation} disabled={loading}>
          新建会话
        </button>
      </section>

      <section>
        <h2>企业知识库</h2>
        <input
          type="file"
          accept=".txt,text/plain"
          onChange={uploadDocument}
          disabled={uploading}
        />
        <p>{uploading ? "正在上传并处理……" : documentStatus}</p>
        <label>
          知识库 ID
          <input
            type="text"
            value={knowledgeBaseId}
            onChange={(event) => updateKnowledgeBaseId(event.target.value)}
            placeholder="上传后自动生成，也可以粘贴已有 ID"
            maxLength={64}
          />
        </label>
      </section>

      <textarea
        value={question}
        onChange={(event) => setQuestion(event.target.value)}
        placeholder="请输入你的问题"
        rows="5"
        maxLength={4000}
      />

      <div>
        <button
          onClick={sendMessage}
          disabled={loading}
        >
          {loading ? "正在思考……" : "发送"}
        </button>
      </div>

      {error && (
        <p style={{ color: "red" }}>
          {error}
        </p>
      )}

      {messages.length > 0 && (
        <section>
          <h2>聊天记录</h2>
          {messages.map((message, index) => (
            <article key={`${message.role}-${index}`}>
              <strong>{message.role === "user" ? "你" : "助手"}</strong>
              <p style={{ whiteSpace: "pre-wrap" }}>{message.content}</p>
            </article>
          ))}
        </section>
      )}

      {agentInfo && (
        <section>
          <h2>最近一次执行</h2>
          <small>本次回答匹配到 {matchedChunks} 个知识片段</small>
          <small> · {llmCalled ? "已调用 LLM" : "未调用 LLM"}</small>
          <p>Skill：{agentInfo.active_skill || "无"}</p>
          <p>
            Tools：
            {agentInfo.tools_used?.length
              ? agentInfo.tools_used.join(", ")
              : "无"}
          </p>
          <p>Tool Source：{agentInfo.tool_source || "无"}</p>
          <p>MCP Server：{agentInfo.mcp_server || "无"}</p>
          <p>执行步骤：{agentInfo.steps}</p>
        </section>
      )}

      {trace && (
        <details>
          {/* 调试轨迹默认折叠，只展示执行元数据，不展示模型思维过程。 */}
          <summary>Agent Trace（智能代理执行轨迹）</summary>
          <p>Request ID：{trace.request_id}</p>
          <p>开始时间：{trace.started_at}</p>
          <p>总耗时：{trace.duration_ms} ms</p>
          <p>步骤：{trace.steps}</p>
          <p>Skill：{trace.skill || "无"}</p>
          <p>
            RAG：{trace.rag?.hit ? "命中" : "未命中"}（
            {trace.rag?.results || 0} 条）
          </p>
          <p>
            LLM Calls：{trace.llm_calls}，耗时 {trace.llm_duration_ms} ms
          </p>
          <h3>Tools（工具）</h3>
          {trace.tools?.length ? (
            <ul>
              {trace.tools.map((tool, index) => (
                <li key={`${tool.name}-${index}`}>
                  {tool.name} — {tool.source} — {tool.duration_ms} ms
                  {tool.server ? ` — ${tool.server}` : ""}
                </li>
              ))}
            </ul>
          ) : (
            <p>无</p>
          )}
        </details>
      )}
    </main>
  );
}

export default App;
