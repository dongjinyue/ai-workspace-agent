import assert from "node:assert/strict";
import test from "node:test";
import { createSseParser } from "./streaming.js";


test("SSE 解析器支持跨网络分块接收事件", () => {
  const events = [];
  const parser = createSseParser((event) => events.push(event));

  parser.push('event: token\ndata: {"content":"你"}\n\n');
  parser.push('event: done\ndata: {"answer":"你好"');
  parser.push("}\n\n");
  parser.finish();

  assert.deepEqual(events, [
    { event: "token", data: { content: "你" } },
    { event: "done", data: { answer: "你好" } },
  ]);
});
