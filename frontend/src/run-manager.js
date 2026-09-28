const ACTIVE_STATUSES = new Set([
  "queued",
  "running",
  "pause_requested",
  "paused",
]);

export function isRunActive(status) {
  return ACTIVE_STATUSES.has(status);
}

function safeError(error, fallback = "Agent 执行失败，请稍后重试。") {
  return {
    message: error?.message || fallback,
    status: error?.status,
    retryable: true,
  };
}

/**
 * 管理多个并发 Agent Run（执行任务），让 React（前端框架）只关心状态渲染。
 */
export function createRunManager({
  streamRequest,
  request,
  timeoutMs = 120000,
  onChange = () => {},
}) {
  const runs = new Map();
  let sequence = 0;

  function notify() {
    onChange([...runs.values()]);
  }

  function update(run, values) {
    Object.assign(run, values);
    notify();
  }

  function setTimer(run) {
    if (!Number.isFinite(timeoutMs) || timeoutMs <= 0) return;
    run.timer = setTimeout(async () => {
      if (!isRunActive(run.status)) return;
      update(run, {
        status: "timed_out",
        error: { message: "任务执行超时，请重试。", retryable: true },
      });
      if (run.runId) {
        try {
          await request(`/api/agent/runs/${run.runId}/cancel`, { method: "POST" });
        } catch {
          // 后端可能已经因超时关闭任务，前端仍保留可重试状态。
        }
      }
      run.controller?.abort();
    }, timeoutMs);
  }

  function clearTimer(run) {
    if (run.timer) clearTimeout(run.timer);
    run.timer = null;
  }

  function handleEvent(run, event) {
    const data = event.data || {};
    if (event.event === "run") {
      update(run, {
        runId: data.run_id || run.runId,
        conversationId: data.conversation_id || run.conversationId,
        status: data.status || "running",
      });
      return;
    }
    if (event.event === "token") {
      update(run, { content: `${run.content}${data.content || ""}` });
      return;
    }
    if (event.event === "trace") {
      update(run, { trace: data.trace || data });
      return;
    }
    if (event.event === "status") {
      update(run, { status: data.status || data.stage || run.status });
      return;
    }
    if (event.event === "paused") {
      update(run, { status: "paused" });
      return;
    }
    if (event.event === "cancelled") {
      update(run, { status: "stopped", error: safeError({ message: "任务已停止。" }) });
      return;
    }
    if (event.event === "timeout") {
      update(run, { status: "timed_out", error: safeError({ message: "任务执行超时，请重试。" }) });
      return;
    }
    if (event.event === "error") {
      update(run, { status: "failed", error: safeError(data) });
      return;
    }
    if (event.event === "done") {
      update(run, {
        status: "completed",
        runId: data.run_id || run.runId,
        conversationId: data.conversation_id || run.conversationId,
        answer: data.answer ?? run.content,
        content: data.answer ?? run.content,
        trace: data.trace || run.trace,
        result: data,
      });
    }
  }

  async function consume(run, path) {
    run.controller = new AbortController();
    setTimer(run);
    try {
      await streamRequest(
        path,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(run.payload),
          signal: run.controller.signal,
        },
        (event) => handleEvent(run, event),
      );
      if (run.status === "running" || run.status === "queued") {
        update(run, { status: "failed", error: { message: "流式响应未正常结束，请重试。", retryable: true } });
      }
    } catch (error) {
      if (run.status !== "timed_out" && run.status !== "stopped") {
        update(run, { status: "failed", error: safeError(error) });
      }
    } finally {
      clearTimer(run);
      run.controller = null;
      notify();
    }
    return run;
  }

  function createRun(payload) {
    const run = {
      localId: `local-${++sequence}`,
      runId: null,
      conversationId: payload.conversation_id || null,
      payload: { ...payload },
      status: "queued",
      content: "",
      answer: "",
      trace: null,
      error: null,
      timer: null,
      controller: null,
    };
    runs.set(run.localId, run);
    notify();
    run.done = consume(run, "/api/agent/chat/stream");
    return run;
  }

  async function control(run, action) {
    if (!run.runId) throw new Error("任务尚未取得执行标识，请稍后重试。");
    const data = await request(`/api/agent/runs/${run.runId}/${action}`, { method: "POST" });
    if (data?.status) update(run, { status: data.status });
    return run;
  }

  return {
    start(payload) {
      return createRun(payload);
    },
    pause(run) {
      return control(run, "pause");
    },
    cancel(run) {
      return control(run, "cancel").then((value) => {
        update(run, { status: "stopped", error: { message: "任务已停止。", retryable: false } });
        run.controller?.abort();
        return value;
      });
    },
    async resume(run) {
      if (!run.runId) throw new Error("任务尚未取得执行标识，请稍后重试。");
      update(run, { status: "running", error: null });
      return consume(run, `/api/agent/runs/${run.runId}/resume`);
    },
    retry(run, payload = run.payload) {
      return createRun({ ...payload });
    },
    activeRuns() {
      return [...runs.values()].filter((run) => isRunActive(run.status));
    },
    allRuns() {
      return [...runs.values()];
    },
  };
}
