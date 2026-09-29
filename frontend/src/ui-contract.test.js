import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";

const appSource = readFileSync(new URL("./App.jsx", import.meta.url), "utf8");

test("回答卡片不再提供关闭操作", () => {
  assert.doesNotMatch(appSource, /className="run-close"/);
  assert.doesNotMatch(appSource, /dismissRun/);
});
