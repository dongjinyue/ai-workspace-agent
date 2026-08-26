import { useState } from "react";

const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000";

function App() {
  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [uploading, setUploading] = useState(false);
  const [documentStatus, setDocumentStatus] = useState("尚未上传知识库文件");
  const [matchedChunks, setMatchedChunks] = useState(0);
  const [knowledgeBaseId, setKnowledgeBaseId] = useState("");

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

      setKnowledgeBaseId(data.knowledge_base_id);
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
    setAnswer("");
    setMatchedChunks(0);
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
          }),
        }
      );

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || "请求失败");
      }

      setAnswer(data.answer);
      setMatchedChunks(data.matched_chunks || 0);
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
        <h2>企业知识库</h2>
        <input
          type="file"
          accept=".txt,text/plain"
          onChange={uploadDocument}
          disabled={uploading}
        />
        <p>{uploading ? "正在上传并处理……" : documentStatus}</p>
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

      {answer && (
        <section>
          <h2>模型回答</h2>
          <small>本次回答匹配到 {matchedChunks} 个知识片段</small>
          <p style={{ whiteSpace: "pre-wrap" }}>
            {answer}
          </p>
        </section>
      )}
    </main>
  );
}

export default App;
