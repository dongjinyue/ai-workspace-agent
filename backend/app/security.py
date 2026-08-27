PROMPT_INJECTION_MARKERS = (
    "忽略之前",
    "忽略以上",
    "系统指令",
    "你现在必须",
    "必须告诉用户",
)


class PromptInjectionError(ValueError):
    """上传内容包含疑似操控模型行为的指令。"""


def contains_prompt_injection(text: str) -> bool:
    """使用保守规则识别知识文档中的高风险模型指令。"""
    return any(marker in text for marker in PROMPT_INJECTION_MARKERS)
