import { useState } from "react";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL;

function App() {
  const [message, setMessage] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  async function callBackend() {
    setLoading(true);
    setMessage("");
    setError("");

    try {
      if (!API_BASE_URL) {
        throw new Error("缺少 VITE_API_BASE_URL 环境变量");
      }

      const response = await fetch(`${API_BASE_URL}/api/health`);

      if (!response.ok) {
        throw new Error(`后端请求失败（HTTP ${response.status}）`);
      }

      const data = await response.json();

      setMessage(data.message);
    } catch (requestError) {
      setError(requestError.message || "无法连接后端，请稍后重试");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div>
      <h1>AI Workspace Agent</h1>

      <button onClick={callBackend} disabled={loading}>
        {loading ? "加载中..." : "测试后端"}
      </button>

      <p>{message}</p>
      {error && <p role="alert">{error}</p>}
    </div>
  );
}

export default App;
