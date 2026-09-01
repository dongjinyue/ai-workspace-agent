import os
import re
from uuid import uuid4
from dataclasses import dataclass
from pathlib import Path

import chromadb
from chromadb.api.models.Collection import Collection


BACKEND_DIR = Path(__file__).resolve().parents[2]
configured_path = Path(os.getenv("CHROMA_PATH", "data/chroma"))
VECTOR_DB_PATH = (
    configured_path
    if configured_path.is_absolute()
    else BACKEND_DIR / configured_path
)

client = chromadb.PersistentClient(path=str(VECTOR_DB_PATH))


@dataclass(frozen=True)
class SearchMatch:
    document: str
    distance: float

    @property
    def similarity(self) -> float:
        return 1.0 - self.distance


def _collection_name(knowledge_base_id: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", knowledge_base_id):
        raise ValueError("knowledge_base_id 格式不正确")
    return f"kb_{knowledge_base_id}"


def get_collection(knowledge_base_id: str) -> Collection:
    return client.get_or_create_collection(
        name=_collection_name(knowledge_base_id),
        embedding_function=None,
        configuration={"hnsw": {"space": "cosine"}},
    )


def find_collection(knowledge_base_id: str) -> Collection | None:
    try:
        return client.get_collection(
            name=_collection_name(knowledge_base_id),
            embedding_function=None,
        )
    except Exception as error:
        if error.__class__.__name__ == "NotFoundError":
            return None
        raise


def delete_collection(knowledge_base_id: str) -> None:
    """删除指定知识库的全部向量；不存在时保持幂等。"""
    if find_collection(knowledge_base_id) is not None:
        client.delete_collection(name=_collection_name(knowledge_base_id))


def add_chunks(
    knowledge_base_id: str,
    chunks: list[str],
    embeddings: list[list[float]],
    *,
    source_filename: str | None = None,
    upload_batch: str | None = None,
) -> None:
    if len(chunks) != len(embeddings):
        raise ValueError("文本块数量与向量数量不一致")
    if not chunks:
        raise ValueError("没有可保存的文本块")

    collection = get_collection(knowledge_base_id)
    collection.add(
        # UUID 避免向同一个知识库连续写入多份文档时发生 ID 冲突。
        ids=[f"chunk_{uuid4().hex}" for _ in chunks],
        documents=chunks,
        embeddings=embeddings,
        metadatas=[
            {
                "chunk_index": index,
                "source": source_filename or "unknown",
                "upload_batch": upload_batch or "legacy",
            }
            for index in range(len(chunks))
        ],
    )


def delete_chunks_by_upload_batch(
    knowledge_base_id: str,
    upload_batch: str,
) -> None:
    """回滚一次追加上传产生的向量，不影响知识库中已有文档。"""
    collection = find_collection(knowledge_base_id)
    if collection is not None:
        collection.delete(where={"upload_batch": upload_batch})


def delete_chunks_by_source(
    knowledge_base_id: str,
    source_filename: str,
) -> None:
    """删除某份文档产生的全部向量块。"""
    collection = find_collection(knowledge_base_id)
    if collection is not None:
        collection.delete(where={"source": source_filename})


def search_chunks(
    knowledge_base_id: str,
    query_embedding: list[float],
    *,
    top_k: int = 3,
    max_distance: float = 0.45,
    query_text: str = "",
) -> list[SearchMatch]:
    collection = find_collection(knowledge_base_id)
    if collection is None or collection.count() == 0:
        return []

    # 多取一些候选，用关键词重合度补救向量距离刚好越过阈值的中文短问题。
    candidate_count = min(max(top_k * 4, 20), collection.count())
    result = collection.query(
        query_embeddings=[query_embedding],
        n_results=candidate_count,
        include=["documents", "distances"],
    )

    documents = (result.get("documents") or [[]])[0]
    distances = (result.get("distances") or [[]])[0]

    def lexical_score(document: str) -> float:
        normalized_query = re.sub(
            r"[\s，。！？、：；,.!?]|是什么|有什么|请问|一下|介绍|告诉我",
            "",
            query_text,
        )
        normalized_document = re.sub(r"\s+", "", document)
        if len(normalized_query) < 2:
            return 0.0
        bigrams = {
            normalized_query[index : index + 2]
            for index in range(len(normalized_query) - 1)
        }
        if not bigrams:
            return 0.0
        return sum(item in normalized_document for item in bigrams) / len(bigrams)

    matches = [
        SearchMatch(document=document, distance=float(distance))
        for document, distance in zip(documents, distances)
        if document is not None
        and distance is not None
        and (
            float(distance) <= max_distance
            or lexical_score(document) >= 0.35
        )
    ]
    return matches[:top_k]
