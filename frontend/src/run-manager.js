const ACTIVE_STATUSES = new Set([
  "queued",
  "running",
  "pause_requested",
  "paused",
]);

// 暂停任务仍需保留在列表中用于恢复，但它已经不再生成内容。
const GENERATING_STATUSES = new Set(["queued", "running", "pause_requested"]);

const STATUS_LABELS = {
  queued: "排队中",
  running: "运行中",
  pause_requested: "正在暂停",
  paused: "已暂停",
  completed: "已完成",
  stopped: "已停止",
  timed_out: "已超时",
  failed: "失败",
};

const STATUS_ERROR_MESSAGES = {
  401: "登录状态已失效，请重新登录。",
  403: "没有权限执行此任务。",
  404: "任务不存在或已过期。",
  408: "任务执行超时，请重试。",
  429: "今日访客额度已用完，请稍后再试。",
  502: "模型服务暂时不可用，请重试。",
  503: "模型服务暂时不可用，请重试。",
  504: "任务执行超时，请重试。",
};

export function isRunActive(status) {
  return ACTIVE_STATUSES.has(status);
}

export function isRunGenerating(status) {
  return GENERATING_STATUSES.has(status);
}

/**
 * 找到当前会话最近一个仍在生成的 Run（执行任务）。
 *
 * 暂停 Run 不会占用输入框的暂停按钮，这样用户可以直接发送下一个问题。
 */
export function getCurrentGeneratingRun(runs, conversationId) {
  const targetConversationId = conversationId || null;
  return [...(runs || [])]
    .filter((run) => {
      // 优先使用发送时保存的会话 ID；新会话的 null 不能被服务端后续分配的 ID 覆盖。
      const hasPayloadConversationId = Object.prototype.hasOwnProperty.call(run?.payload || {}, "conversation_id");
      const runConversationId = hasPayloadConversationId ? run.payload.conversation_id : run?.conversationId || null;
      return isRunGenerating(run?.status) && runConversationId === targetConversationId;
    })
    .at(-1) || null;
}

export function getRunStatusLabel(status) {
  return STATUS_LABELS[status] || "处理中";
}

export function getRunControlState(status) {
  return {
    canPause: status === "queued" || status === "running",
    canResume: status === "paused",
    canCancel: isRunActive(status),
    canRetry: status === "failed" || status === "timed_out" || status === "stopped",
  };
}

export function getSafeRunError(error, fallback = "Agent 执行失败，请稍后重试。") {
  const status = Number(error?.status);
  const rawMessage = typeof error?.message === "string" ? error.message.trim() : "";
  const containsInternalDetail = /Traceback|Exception|API[_ ]?KEY|(?:^|\s)sk-[A-Za-z0-9]|\.py:\d+|\bat\s+[^\s]+\s*\(/i.test(rawMessage);
  const message = STATUS_ERROR_MESSAGES[status] || (rawMessage && !containsInternalDetail ? rawMessage.slice(0, 240) : fallback);
  return { message, status: Number.isFinite(status) ? status : undefined, retryable: true };
}

export function getSafeRagDebug(rag) {
  if (!rag || typeof rag !== "object") return null;
  const numberOrNull = (value) => (typeof value === "number" && Number.isFinite(value) ? value : null);
  const topK = numberOrNull(rag.top_k);
  const maxDistance = numberOrNull(rag.max_distance);
  const returned = numberOrNull(rag.returned);
  const matches = Array.isArray(rag.matches)
    ? rag.matches.map((match) => ({
      rank: numberOrNull(match?.rank),
      similarity: numberOrNull(match?.similarity),
      distance: numberOrNull(match?.distance),
    }))
    : [];
  return { topK, maxDistance, returned, matches };
}

function safeError(error, fallback = "Agent 执行失败，请稍后重试。") {
  return getSafeRunError(error, fallback);
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
      const pausePending = run.pausePending;
      update(run, {
        runId: data.run_id || run.runId,
        conversationId: data.conversation_id || run.conversationId,
        status: pausePending ? "pause_requested" : data.status || "running",
      });
      if (pausePending && run.runId) {
        run.pausePending = false;
        control(run, "pause").catch((error) => {
          update(run, { status: "running", error: safeError(error) });
        });
      }
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
      pausePending: false,
    };
    runs.set(run.localId, run);
    notify();
    run.done = consume(run, "/api/agent/chat/stream");
    return run;
  }

  async function control(run, action) {
    if (!run.runId) throw new Error("任务尚未取得执行标识，请稍后重试。");
    const data = await request(`/api/agent/runs/${run.runId}/${action}`, { method: "POST" });
    if (data?.status && (action !== "pause" || isRunGenerating(run.status))) {
      update(run, { status: data.status });
    }
    return run;
  }

  return {
    start(payload) {
      return createRun(payload);
    },
    pause(run) {
      if (!run.runId) {
        // SSE（服务器推送事件）还没发回 run_id 时，先记录意图，收到标识后立即补发暂停请求。
        run.pausePending = true;
        update(run, { status: "pause_requested" });
        return Promise.resolve(run);
      }
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
      // 恢复会替换当前订阅 Promise，让页面可以继续等待同一个 Run 的最终状态。
      run.done = consume(run, `/api/agent/runs/${run.runId}/resume`);
      return run.done;
    },
    retry(run, payload = run.payload) {
      return createRun({ ...payload });
    },
    restore(runData) {
      if (!runData?.run_id || !isRunActive(runData.status)) return null;
      const localId = `restored-${runData.run_id}`;
      const existing = runs.get(localId);
      if (existing) return existing;
      const run = {
        localId,
        runId: runData.run_id,
        conversationId: runData.conversation_id || null,
        payload: { conversation_id: runData.conversation_id || null, message: "" },
        status: runData.status,
        content: runData.answer_prefix || "",
        answer: runData.answer_prefix || "",
        trace: null,
        error: null,
        timer: null,
        controller: null,
      };
      runs.set(localId, run);
      notify();
      return run;
    },
    activeRuns() {
      return [...runs.values()].filter((run) => isRunActive(run.status));
    },
    allRuns() {
      return [...runs.values()];
    },
  };
}
