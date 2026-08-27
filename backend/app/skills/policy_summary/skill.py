from dataclasses import dataclass

from app.skills.policy_summary.prompt import POLICY_SUMMARY_INSTRUCTIONS


@dataclass(frozen=True)
class SkillDefinition:
    name: str
    description: str
    instructions: str
    allowed_tools: tuple[str, ...]
    activation_intents: tuple[str, ...]
    activation_contexts: tuple[str, ...]

    def matches(self, message: str) -> bool:
        """只有意图和业务上下文同时匹配时才激活技能。"""
        return any(word in message for word in self.activation_intents) and any(
            word in message for word in self.activation_contexts
        )


POLICY_SUMMARY_SKILL = SkillDefinition(
    name="policy_summary",
    description="当用户明确要求总结、整理或摘要企业政策时使用",
    instructions=POLICY_SUMMARY_INSTRUCTIONS,
    allowed_tools=("search_knowledge_base",),
    activation_intents=("总结", "摘要", "整理", "概括", "归纳"),
    activation_contexts=(
        "公司",
        "企业",
        "政策",
        "规定",
        "制度",
        "规则",
        "退款",
        "补贴",
    ),
)
