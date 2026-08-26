import { useState } from "react";

const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000";

function App() {
  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  async function sendMessage() {
    if (!question.trim()) {
      setError("请输入问题");
      return;
    }

    setLoading(true);
    setAnswer("");
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
          }),
        }
      );

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || "请求失败");
      }

      setAnswer(data.answer);
    } catch (requestError) {
      setError(requestError.message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <main>
      <h1>AI Workspace Agent</h1>

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
          <p style={{ whiteSpace: "pre-wrap" }}>
            {answer}
          </p>
        </section>
      )}
    </main>
  );
}

export default App;
