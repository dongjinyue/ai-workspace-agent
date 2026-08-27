import os
import re

from app.rag.embeddings import embed_texts
from app.rag.vector_store import SearchMatch, add_chunks, find_collection, search_chunks


def split_text(
    text: str,
    chunk_size: int = 500,
    overlap: int = 80,
) -> list[str]:
    """把文档切成带重叠的文本块，减少答案被切断的情况。"""
    cleaned_text = text.strip()
    if not cleaned_text:
        return []
    if chunk_size <= 0:
        raise ValueError("chunk_size 必须大于 0")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap 必须大于等于 0 且小于 chunk_size")

    chunks = []
    step = chunk_size - overlap
    paragraphs = [
        paragraph.strip()
        for paragraph in re.split(r"\n\s*\n", cleaned_text)
        if paragraph.strip()
    ]

    for paragraph in paragraphs:
        if len(paragraph) <= chunk_size:
            chunks.append(paragraph)
            continue

        for start in range(0, len(paragraph), step):
            chunk = paragraph[start : start + chunk_size].strip()
            if chunk:
                chunks.append(chunk)
            if start + chunk_size >= len(paragraph):
                break

    return chunks


def index_document(
    knowledge_base_id: str,
    text: str,
) -> int:
    """切分文档、批量向量化并持久化到指定知识库。"""
    chunks = split_text(text)
    if not chunks:
        return 0

    embeddings = embed_texts(chunks, text_type="document")
    add_chunks(knowledge_base_id, chunks, embeddings)
    return len(chunks)


def semantic_search(
    knowledge_base_id: str,
    query: str,
    *,
    top_k: int = 3,
) -> list[SearchMatch]:
    """向量化用户问题，并过滤超过距离阈值的不可信结果。"""
    if not query.strip() or find_collection(knowledge_base_id) is None:
        return []

    max_distance = float(os.getenv("RAG_MAX_COSINE_DISTANCE", "0.45"))
    query_embedding = embed_texts([query], text_type="query")[0]
    return search_chunks(
        knowledge_base_id,
        query_embedding,
        top_k=top_k,
        max_distance=max_distance,
    )
