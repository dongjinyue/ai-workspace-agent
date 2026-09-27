const shanghaiTime = new Intl.DateTimeFormat("zh-CN", {
  timeZone: "Asia/Shanghai",
  hour: "2-digit",
  minute: "2-digit",
  hourCycle: "h23",
});

/** 将服务端额度状态转换为面向访客的中文提示。 */
export function formatQuotaStatus(status) {
  if (status === null) return "管理员模式 · 不受访客次数限制";
  if (status === undefined) return "正在读取使用额度…";

  const remaining = Number(status.remaining);
  if (!Number.isFinite(remaining) || remaining < 0) return "额度状态暂不可用";
  if (remaining === 0) {
    const resetAt = new Date(status.reset_at);
    const resetCopy = Number.isNaN(resetAt.getTime())
      ? "每日重置"
      : `今日额度将于 ${shanghaiTime.format(resetAt)} 重置`;
    return `今日次数已用完 · ${resetCopy}`;
  }

  const resetAt = new Date(status.reset_at);
  const resetCopy = Number.isNaN(resetAt.getTime())
    ? "每日重置"
    : `每日 ${shanghaiTime.format(resetAt)} 重置`;
  return `还可提问 ${Math.floor(remaining)} 次 · ${resetCopy}`;
}
