import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import test from "node:test";

const frontendDirectory = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const repositoryDirectory = resolve(frontendDirectory, "..");
const composeConfiguration = readFileSync(
  resolve(repositoryDirectory, "compose.yaml"),
  "utf8",
);
const frontendDockerfile = readFileSync(
  resolve(frontendDirectory, "dockerfile"),
  "utf8",
);
const backendDockerfile = readFileSync(
  resolve(repositoryDirectory, "backend", "Dockerfile"),
  "utf8",
);
const dockerEnvironmentBlock =
  frontendDockerfile.replace(/\\\r?\n/g, " ").match(/^ENV.*$/m)?.[0] ?? "";

test("生产镜像构建会把 Supabase 公开配置传入 Vite", () => {
  for (const variable of ["VITE_SUPABASE_URL", "VITE_SUPABASE_PUBLISHABLE_KEY"]) {
    assert.match(
      composeConfiguration,
      new RegExp(`${variable}:\\s*\\$\\{${variable}:-\\}`),
      `compose.yaml 必须把 ${variable} 作为前端构建参数传入`,
    );
    assert.match(
      frontendDockerfile,
      new RegExp(`ARG ${variable}\\b`),
      `Dockerfile 必须声明 ${variable} 构建参数`,
    );
    assert.match(
      dockerEnvironmentBlock,
      new RegExp(`${variable}=\\$${variable}`),
      `Dockerfile 必须在 Vite 构建环境中提供 ${variable}`,
    );
  }
});

test("后端镜像构建支持指定可访问的 Python 包索引", () => {
  assert.match(
    composeConfiguration,
    /PIP_INDEX_URL:\s*\$\{PIP_INDEX_URL:-https:\/\/pypi\.org\/simple\}/,
    "Compose 必须将包索引地址传给后端构建",
  );
  assert.match(
    backendDockerfile,
    /ARG PIP_INDEX_URL=https:\/\/pypi\.org\/simple/,
    "后端 Dockerfile 必须提供官方索引默认值",
  );
  assert.match(
    backendDockerfile,
    /pip install[^\n]*--index-url\s+"\$PIP_INDEX_URL"/,
    "后端 Python 依赖安装必须使用可配置的包索引",
  );
});
