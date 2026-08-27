import os
import re
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


def add_chunks(
    knowledge_base_id: str,
    chunks: list[str],
    embeddings: list[list[float]],
) -> None:
    if len(chunks) != len(embeddings):
        raise ValueError("文本块数量与向量数量不一致")
    if not chunks:
        raise ValueError("没有可保存的文本块")

    collection = get_collection(knowledge_base_id)
    collection.add(
        ids=[f"chunk_{index}" for index in range(len(chunks))],
        documents=chunks,
        embeddings=embeddings,
        metadatas=[{"chunk_index": index} for index in range(len(chunks))],
    )


def search_chunks(
    knowledge_base_id: str,
    query_embedding: list[float],
    *,
    top_k: int = 3,
    max_distance: float = 0.45,
) -> list[SearchMatch]:
    collection = find_collection(knowledge_base_id)
    if collection is None or collection.count() == 0:
        return []

    result = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, collection.count()),
        include=["documents", "distances"],
    )

    documents = (result.get("documents") or [[]])[0]
    distances = (result.get("distances") or [[]])[0]

    return [
        SearchMatch(document=document, distance=float(distance))
        for document, distance in zip(documents, distances)
        if document is not None
        and distance is not None
        and float(distance) <= max_distance
    ]
