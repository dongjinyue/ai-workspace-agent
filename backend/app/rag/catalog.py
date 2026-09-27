import os
import re
from datetime import datetime, timezone

from app.memory.database import get_connection, init_database


_KNOWLEDGE_BASE_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def get_public_knowledge_base_ids() -> frozenset[str]:
    """解析公开知识库名单；任一格式错误都会关闭整份名单，默认不公开。"""
    configured = os.getenv("PUBLIC_KNOWLEDGE_BASE_IDS", "")
    ids = [item.strip() for item in configured.split(",") if item.strip()]
    if not ids or any(not _KNOWLEDGE_BASE_ID.fullmatch(item) for item in ids):
        return frozenset()
    return frozenset(ids)


def register_knowledge_base(
    knowledge_base_id: str,
    documents: list[dict[str, str | int]],
) -> None:
    """在一个事务中保存知识库和它包含的多份文档。"""
    init_database()
    created_at = datetime.now(timezone.utc).isoformat()
    total_chunks = sum(int(item["chunk_count"]) for item in documents)
    display_name = str(documents[0]["filename"])
    with get_connection() as connection:
        connection.execute(
            """
            INSERT INTO knowledge_bases (id, filename, chunk_count, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (
                knowledge_base_id,
                display_name,
                total_chunks,
                created_at,
            ),
        )
        connection.executemany(
            """
            INSERT INTO knowledge_documents
                (knowledge_base_id, filename, chunk_count, upload_batch, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            [
                (
                    knowledge_base_id,
                    str(item["filename"]),
                    int(item["chunk_count"]),
                    item.get("upload_batch"),
                    created_at,
                )
                for item in documents
            ],
        )


def list_knowledge_bases(*, limit: int = 50, offset: int = 0) -> list[dict]:
    init_database()
    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT id, filename, chunk_count, created_at
            FROM knowledge_bases
            ORDER BY created_at DESC
            LIMIT ? OFFSET ?
            """,
            (limit, offset),
        ).fetchall()
    knowledge_bases = [dict(row) for row in rows]
    with get_connection() as connection:
        for item in knowledge_bases:
            documents = connection.execute(
                """
                SELECT id, filename, chunk_count
                FROM knowledge_documents
                WHERE knowledge_base_id = ?
                ORDER BY id
                """,
                (item["id"],),
            ).fetchall()
            # 旧数据库只有 filename 字段，升级后仍能正常显示原文档名称。
            item["documents"] = (
                [dict(document) for document in documents]
                if documents
                else [
                    {
                        "filename": item["filename"],
                        "chunk_count": item["chunk_count"],
                    }
                ]
            )
    return knowledge_bases


def list_public_knowledge_bases(
    public_ids: frozenset[str], *, limit: int = 50, offset: int = 0
) -> list[dict]:
    """只查询并返回明确列入访客公开名单的知识库及其目录元数据。"""
    valid_ids = sorted(
        item for item in public_ids if _KNOWLEDGE_BASE_ID.fullmatch(item)
    )
    if not valid_ids:
        return []
    placeholders = ",".join("?" for _ in valid_ids)
    init_database()
    with get_connection() as connection:
        rows = connection.execute(
            f"""
            SELECT id, filename, chunk_count, created_at
            FROM knowledge_bases
            WHERE id IN ({placeholders})
            ORDER BY created_at DESC
            LIMIT ? OFFSET ?
            """,
            (*valid_ids, limit, offset),
        ).fetchall()
        knowledge_bases = [dict(row) for row in rows]
        for item in knowledge_bases:
            documents = connection.execute(
                "SELECT id, filename, chunk_count FROM knowledge_documents "
                "WHERE knowledge_base_id = ? ORDER BY id",
                (item["id"],),
            ).fetchall()
            item["documents"] = (
                [dict(document) for document in documents]
                if documents
                else [
                    {
                        "filename": item["filename"],
                        "chunk_count": item["chunk_count"],
                    }
                ]
            )
    return knowledge_bases


def get_knowledge_base(knowledge_base_id: str) -> dict | None:
    """读取一个知识库及其文档列表。"""
    init_database()
    with get_connection() as connection:
        row = connection.execute(
            """
            SELECT id, filename, chunk_count, created_at
            FROM knowledge_bases
            WHERE id = ?
            """,
            (knowledge_base_id,),
        ).fetchone()
        if row is None:
            return None
        item = dict(row)
        documents = connection.execute(
            """
            SELECT id, filename, chunk_count
            FROM knowledge_documents
            WHERE knowledge_base_id = ?
            ORDER BY id
            """,
            (knowledge_base_id,),
        ).fetchall()
    item["documents"] = (
        [dict(document) for document in documents]
        if documents
        else [{"filename": item["filename"], "chunk_count": item["chunk_count"]}]
    )
    return item


def get_public_knowledge_base(
    knowledge_base_id: str, public_ids: frozenset[str]
) -> dict | None:
    """在读取元数据之前先校验公开名单，避免枚举私有知识库是否存在。"""
    if (
        not _KNOWLEDGE_BASE_ID.fullmatch(knowledge_base_id)
        or knowledge_base_id not in public_ids
    ):
        return None
    return get_knowledge_base(knowledge_base_id)


def get_knowledge_document(
    knowledge_base_id: str,
    document_id: int,
) -> dict | None:
    """读取待删除文档，确保它确实属于指定知识库。"""
    init_database()
    with get_connection() as connection:
        row = connection.execute(
            """
            SELECT id, filename, chunk_count, upload_batch
            FROM knowledge_documents
            WHERE id = ? AND knowledge_base_id = ?
            """,
            (document_id, knowledge_base_id),
        ).fetchone()
    return dict(row) if row else None


def delete_knowledge_document(
    knowledge_base_id: str,
    document_id: int,
) -> bool:
    """删除文档元数据并同步扣减知识库的总分块数。"""
    init_database()
    with get_connection() as connection:
        row = connection.execute(
            """
            SELECT chunk_count
            FROM knowledge_documents
            WHERE id = ? AND knowledge_base_id = ?
            """,
            (document_id, knowledge_base_id),
        ).fetchone()
        if row is None:
            return False
        connection.execute(
            "DELETE FROM knowledge_documents WHERE id = ?",
            (document_id,),
        )
        connection.execute(
            """
            UPDATE knowledge_bases
            SET chunk_count = MAX(0, chunk_count - ?)
            WHERE id = ?
            """,
            (row["chunk_count"], knowledge_base_id),
        )
    return True


def append_knowledge_documents(
    knowledge_base_id: str,
    documents: list[dict[str, str | int]],
) -> None:
    """在一个事务中追加文档元数据并更新知识库总分块数。"""
    init_database()
    created_at = datetime.now(timezone.utc).isoformat()
    added_chunks = sum(int(item["chunk_count"]) for item in documents)
    with get_connection() as connection:
        cursor = connection.execute(
            """
            UPDATE knowledge_bases
            SET chunk_count = chunk_count + ?
            WHERE id = ?
            """,
            (added_chunks, knowledge_base_id),
        )
        if cursor.rowcount == 0:
            raise LookupError("知识库不存在")
        connection.executemany(
            """
            INSERT INTO knowledge_documents
                (knowledge_base_id, filename, chunk_count, upload_batch, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            [
                (
                    knowledge_base_id,
                    str(item["filename"]),
                    int(item["chunk_count"]),
                    item.get("upload_batch"),
                    created_at,
                )
                for item in documents
            ],
        )


def knowledge_base_exists(knowledge_base_id: str) -> bool:
    init_database()
    with get_connection() as connection:
        row = connection.execute(
            "SELECT 1 FROM knowledge_bases WHERE id = ?",
            (knowledge_base_id,),
        ).fetchone()
    return row is not None


def delete_knowledge_base(knowledge_base_id: str) -> bool:
    with get_connection() as connection:
        cursor = connection.execute(
            "DELETE FROM knowledge_bases WHERE id = ?",
            (knowledge_base_id,),
        )
    return cursor.rowcount > 0
