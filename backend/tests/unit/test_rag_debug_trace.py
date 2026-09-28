from unittest.mock import patch

import pytest

from app.agent.skills.knowledge_skill import knowledge_search
from app.configuration import ConfigurationError, get_rag_top_k
from app.rag.service import semantic_search
from app.rag.vector_store import SearchMatch


pytestmark = pytest.mark.unit


def test_rag_top_k_defaults_to_five_and_accepts_only_one_to_twenty(monkeypatch):
    monkeypatch.delenv("RAG_TOP_K", raising=False)
    assert get_rag_top_k() == 5

    monkeypatch.setenv("RAG_TOP_K", "1")
    assert get_rag_top_k() == 1
    monkeypatch.setenv("RAG_TOP_K", "20")
    assert get_rag_top_k() == 20

    for invalid in ("0", "21", "not-a-number"):
        monkeypatch.setenv("RAG_TOP_K", invalid)
        with pytest.raises(ConfigurationError, match="RAG_TOP_K"):
            get_rag_top_k()


def test_semantic_search_uses_bounded_top_k_and_distance(monkeypatch):
    monkeypatch.setenv("RAG_TOP_K", "3")
    captured = {}

    def fake_search(_knowledge_base_id, _embedding, **kwargs):
        captured.update(kwargs)
        return [SearchMatch(document="资料", distance=0.12)]

    with (
        patch("app.rag.service.find_collection", return_value=object()),
        patch("app.rag.service.embed_texts", return_value=[[0.1, 0.2]]),
        patch("app.rag.service.search_chunks", side_effect=fake_search),
    ):
        result = semantic_search("trusted-kb", "问题")

    assert len(result) == 1
    assert captured["top_k"] == 3
    assert captured["max_distance"] == 0.45


def test_knowledge_search_exposes_only_safe_retrieval_debug_metadata(monkeypatch):
    monkeypatch.setenv("RAG_TOP_K", "2")
    matches = [
        SearchMatch(document="第一份完整文档正文", distance=0.1),
        SearchMatch(document="第二份完整文档正文", distance=0.2),
    ]
    with patch("app.agent.skills.knowledge_skill.semantic_search", return_value=matches):
        result = knowledge_search("trusted-kb", "问题")

    assert result["matched"] is True
    assert result["chunks"] == ["第一份完整文档正文", "第二份完整文档正文"]
    assert result["retrieval_debug"] == {
        "top_k": 2,
        "max_distance": 0.45,
        "returned": 2,
        "matches": [
            {"rank": 1, "similarity": 0.9, "distance": 0.1},
            {"rank": 2, "similarity": 0.8, "distance": 0.2},
        ],
    }
    assert "完整文档正文" not in str(result["retrieval_debug"])


def test_knowledge_search_without_base_still_returns_debug_shape(monkeypatch):
    monkeypatch.setenv("RAG_TOP_K", "5")
    result = knowledge_search(None, "问题")

    assert result["matched"] is False
    assert result["chunks"] == []
    assert result["retrieval_debug"]["returned"] == 0
    assert result["retrieval_debug"]["matches"] == []
