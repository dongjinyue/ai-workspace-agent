import { useState } from "react";

function App() {
  const [message, setMessage] = useState("");

  async function callBackend() {
    const response = await fetch(
      "http://127.0.0.1:8000/api/health"
    );

    const data = await response.json();

    setMessage(data.message);
  }

  return (
    <div>
      <h1>AI Workspace Agent</h1>

      <button onClick={callBackend}>
        测试后端
      </button>

      <p>{message}</p>
    </div>
  );
}

export default App;