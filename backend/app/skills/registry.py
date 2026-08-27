from app.skills.policy_summary import POLICY_SUMMARY_SKILL
from app.skills.policy_summary.skill import SkillDefinition


SKILLS: dict[str, SkillDefinition] = {
    POLICY_SUMMARY_SKILL.name: POLICY_SUMMARY_SKILL,
}

def get_skill(name: str) -> SkillDefinition | None:
    """只从固定注册表取技能，绝不根据名称动态导入代码。"""
    return SKILLS.get(name)


def select_skill(message: str) -> SkillDefinition | None:
    """遍历注册表，由每个技能自己的激活规则决定是否匹配。"""
    return next((skill for skill in SKILLS.values() if skill.matches(message)), None)
