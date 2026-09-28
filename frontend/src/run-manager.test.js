import assert from "node:assert/strict";
import { test } from "node:test";

import { createRunManager } from "./run-manager.js";

function fakeStreamFactory() {
  const calls = [];
  const streamRequest = async (path, _options, onEvent) => {
    calls.push(path);
    const runId = path.includes("/runs/")
      ? path.split("/").at(-2)
      : `run-${calls.length}`;
    onEvent({ event: "run", data: { run_id: runId, status: "running" } });
    if (!path.includes("/runs/")) {
      onEvent({ event: "token", data: { run_id: runId, content: runId } });
      onEvent({
        event: "done",
        data: { run_id: runId, answer: `answer-${runId}`, conversation_id: "c1" },
      });
    } else {
      onEvent({ event: "token", data: { run_id: runId, content: "恢复" } });
      onEvent({
        event: "done",
        data: { run_id: runId, answer: "恢复完成", conversation_id: "c1" },
      });
    }
  };
  return { calls, streamRequest };
}

test("并发 Run 按 run_id 分离 token 和最终回答", async () => {
  const fake = fakeStreamFactory();
  const manager = createRunManager({
    streamRequest: fake.streamRequest,
    request: async () => ({}),
  });

  const first = manager.start({ message: "第一个问题" });
  const second = manager.start({ message: "第二个问题" });
  await Promise.all([first.done, second.done]);

  assert.equal(first.runId, "run-1");
  assert.equal(second.runId, "run-2");
  assert.equal(first.answer, "answer-run-1");
  assert.equal(second.answer, "answer-run-2");
  assert.equal(first.content, "answer-run-1");
  assert.equal(second.content, "answer-run-2");
  assert.deepEqual(manager.activeRuns(), []);
});

test("暂停、恢复和停止使用对应 Run 的控制接口", async () => {
  const fake = fakeStreamFactory();
  const requested = [];
  const manager = createRunManager({
    streamRequest: fake.streamRequest,
    request: async (path, options) => {
      requested.push([path, options?.method]);
      return { run_id: "run-1", status: path.endsWith("pause") ? "pause_requested" : "stopped" };
    },
  });
  const run = manager.start({ message: "问题" });
  await run.done;

  await manager.pause(run);
  await manager.cancel(run);
  await manager.resume(run).catch(() => {});

  assert.deepEqual(requested.slice(0, 2), [
    ["/api/agent/runs/run-1/pause", "POST"],
    ["/api/agent/runs/run-1/cancel", "POST"],
  ]);
  assert.equal(fake.calls.some((path) => path.endsWith("/resume")), true);
});

test("超时标记为可重试并主动通知后端停止", async () => {
  const requested = [];
  const manager = createRunManager({
    streamRequest: async (_path, _options, onEvent) => {
      onEvent({ event: "run", data: { run_id: "run-timeout", status: "running" } });
      await new Promise(() => {});
    },
    request: async (path, options) => {
      requested.push([path, options?.method]);
      return { status: "stopped" };
    },
    timeoutMs: 5,
  });
  const run = manager.start({ message: "会超时的问题" });

  await new Promise((resolve) => setTimeout(resolve, 20));
  assert.equal(run.status, "timed_out");
  assert.deepEqual(requested, [["/api/agent/runs/run-timeout/cancel", "POST"]]);
});

test("失败 Run 可以重试，重试会创建新的 Run", async () => {
  let starts = 0;
  const manager = createRunManager({
    streamRequest: async (_path, _options, onEvent) => {
      starts += 1;
      onEvent({ event: "run", data: { run_id: `run-${starts}`, status: "running" } });
      if (starts === 1) throw Object.assign(new Error("temporary"), { status: 502 });
      onEvent({ event: "done", data: { run_id: `run-${starts}`, answer: "重试成功" } });
    },
    request: async () => ({}),
  });
  const failed = manager.start({ message: "原问题" });
  await failed.done;
  const retried = manager.retry(failed);
  await retried.done;

  assert.equal(failed.status, "failed");
  assert.equal(retried.runId, "run-2");
  assert.equal(retried.answer, "重试成功");
});
