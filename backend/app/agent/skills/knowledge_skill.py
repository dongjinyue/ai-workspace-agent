from pydantic import BaseModel, ConfigDict, Field

from app.agent.skill import Skill, ToolDefinition
from app.rag.service import semantic_search


class KnowledgeSearchArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: str = Field(min_length=1, max_length=4000)


class KnowledgeBaseInfoArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")


def get_knowledge_base_info(knowledge_base_id: str | None) -> dict:
    """返回当前知识库的文档数量和名称，不使用语义检索猜测管理信息。"""
    if not knowledge_base_id:
        return {"document_count": 0, "documents": []}
    # 延迟导入避免 memory 包初始化时反向加载 Agent Service 形成循环依赖。
    from app.rag.catalog import get_knowledge_base

    knowledge_base = get_knowledge_base(knowledge_base_id)
    documents = knowledge_base.get("documents", []) if knowledge_base else []
    return {
        "document_count": len(documents),
        "documents": [item["filename"] for item in documents],
    }


def knowledge_search(knowledge_base_id: str | None, query: str) -> dict:
    """检索当前后端上下文指定的知识库，不接受模型提供知识库 ID。"""
    if not knowledge_base_id:
        return {"matched": False, "chunks": []}
    matches = semantic_search(knowledge_base_id, query)
    return {
        "matched": bool(matches),
        "chunks": [match.document for match in matches],
        "similarities": [round(match.similarity, 4) for match in matches],
    }


KNOWLEDGE_SEARCH_SCHEMA = {
    "type": "function",
    "function": {
        "name": "search_knowledge_base",
        "description": (
            "搜索当前用户的企业知识库。"
            "当问题涉及公司政策、产品规则、员工制度或上传资料中的事实时使用。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "需要检索的问题"}
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
}

KNOWLEDGE_BASE_INFO_SCHEMA = {
    "type": "function",
    "function": {
        "name": "get_knowledge_base_info",
        "description": "查询当前企业知识库已上传的文档数量和文档名称。用户询问有多少文档、有哪些文件时必须使用。",
        "parameters": {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
    },
}

KNOWLEDGE_SKILL = Skill(
    name="knowledge",
    description="提供由后端注入资源标识的企业知识库检索能力。",
    tools=[
        ToolDefinition(
            "search_knowledge_base",
            KNOWLEDGE_SEARCH_SCHEMA,
            knowledge_search,
            KnowledgeSearchArguments,
            needs_knowledge_base=True,
        ),
        ToolDefinition(
            "get_knowledge_base_info",
            KNOWLEDGE_BASE_INFO_SCHEMA,
            get_knowledge_base_info,
            KnowledgeBaseInfoArguments,
            needs_knowledge_base=True,
        ),
    ],
)
