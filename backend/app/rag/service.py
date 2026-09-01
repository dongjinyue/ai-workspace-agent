import os
import re

from app.rag.embeddings import embed_texts
from app.rag.vector_store import SearchMatch, add_chunks, find_collection, search_chunks
from app.security import PromptInjectionError, contains_prompt_injection


def split_text(
    text: str,
    chunk_size: int = 800,
    overlap: int = 120,
) -> list[str]:
    """优先按段落和完整句子分块，避免回答停在半句话中间。"""
    cleaned_text = text.strip()
    if not cleaned_text:
        return []
    if chunk_size <= 0:
        raise ValueError("chunk_size 必须大于 0")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap 必须大于等于 0 且小于 chunk_size")

    # 标题行、段落和以句号/问号/分号结束的句子都是优先边界。
    units = [
        unit.strip()
        for unit in re.split(
            r"(?<=[。！？；!?])\s*|(?<=\.)\s+|\n+",
            cleaned_text,
        )
        if unit.strip()
    ]
    chunks: list[str] = []
    current: list[str] = []

    def flush() -> None:
        if current:
            chunks.append("\n".join(current).strip())

    for unit in units:
        # 极端超长单句按逗号等次级语义边界切分，最后才会硬切字符。
        pending = [unit]
        if len(unit) > chunk_size:
            pending = []
            buffer = ""
            for segment in re.split(r"(?<=[，、,：:])", unit):
                if len(buffer) + len(segment) <= chunk_size:
                    buffer += segment
                    continue
                if buffer:
                    pending.append(buffer.strip())
                    buffer = ""
                while len(segment) > chunk_size:
                    pending.append(segment[:chunk_size].strip())
                    segment = segment[chunk_size:]
                buffer = segment
            if buffer.strip():
                pending.append(buffer.strip())

        for part in pending:
            projected = len("\n".join([*current, part]))
            if current and projected > chunk_size:
                previous = current[-1] if len(current[-1]) <= overlap else ""
                flush()
                current = [previous] if previous else []
            current.append(part)
    flush()
    return chunks


def index_document(
    knowledge_base_id: str,
    text: str,
    source_filename: str | None = None,
    upload_batch: str | None = None,
) -> int:
    """切分文档、批量向量化并持久化到指定知识库。"""
    if contains_prompt_injection(text):
        raise PromptInjectionError("文档包含疑似提示词注入内容，已拒绝上传")
    chunks = split_text(text)
    if not chunks:
        return 0

    embeddings = embed_texts(chunks, text_type="document")
    add_chunks(
        knowledge_base_id,
        chunks,
        embeddings,
        source_filename=source_filename,
        upload_batch=upload_batch,
    )
    return len(chunks)


def semantic_search(
    knowledge_base_id: str,
    query: str,
    *,
    top_k: int = 5,
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
        query_text=query,
    )
