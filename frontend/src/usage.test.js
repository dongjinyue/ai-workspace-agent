import test from "node:test";
import assert from "node:assert/strict";
import { formatQuotaStatus } from "./usage.js";

const now = new Date("2026-09-27T10:00:00.000Z");

test("访客额度显示剩余次数与上海时区重置时间", () => {
  assert.match(formatQuotaStatus({ remaining: 7, limit: 10, reset_at: "2026-09-27T20:00:00+08:00" }, now), /还可提问 7 次/);
  assert.match(formatQuotaStatus({ remaining: 7, limit: 10, reset_at: "2026-09-27T20:00:00+08:00" }, now), /20:00/);
});

test("额度耗尽时明确提示次日重置，不显示负数", () => {
  assert.match(formatQuotaStatus({ remaining: 0, limit: 10, reset_at: "2026-09-27T20:00:00+08:00" }, now), /今日次数已用完/);
});

test("管理员额度为 null 时显示不受访客次数限制", () => {
  assert.equal(formatQuotaStatus(null, now), "管理员模式 · 不受访客次数限制");
});

test("加载中和不完整数据提供稳定提示", () => {
  assert.equal(formatQuotaStatus(undefined, now), "正在读取使用额度…");
  assert.equal(formatQuotaStatus({ remaining: 3, limit: 10, reset_at: "not-a-date" }, now), "还可提问 3 次 · 每日重置");
});
