import os
import re
from datetime import datetime, timezone

from app.memory.database import get_connection, init_database


_KNOWLEDGE_BASE_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def get_public_knowledge_base_ids() -> frozenset[str]:
    """解析通过旧版环境配置共享给所有访客的知识库名单。"""
    configured = os.getenv("PUBLIC_KNOWLEDGE_BASE_IDS", "")
    ids = [item.strip() for item in configured.split(",") if item.strip()]
    if not ids or any(not _KNOWLEDGE_BASE_ID.fullmatch(item) for item in ids):
        return frozenset()
    return frozenset(ids)


def register_knowledge_base(
    knowledge_base_id: str,
    documents: list[dict[str, str | int]],
    *,
    owner_id: str = "admin",
) -> None:
    """在一个事务中保存知识库和它包含的多份文档。"""
    init_database()
    created_at = datetime.now(timezone.utc).isoformat()
    total_chunks = sum(int(item["chunk_count"]) for item in documents)
    display_name = str(documents[0]["filename"])
    with get_connection() as connection:
        connection.execute(
            """
            INSERT INTO knowledge_bases
                (id, filename, chunk_count, created_at, owner_id)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                knowledge_base_id,
                display_name,
                total_chunks,
                created_at,
                owner_id,
            ),
        )
        connection.executemany(
            """
            INSERT INTO knowledge_documents
                (knowledge_base_id, filename, chunk_count, upload_batch, size_bytes, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    knowledge_base_id,
                    str(item["filename"]),
                    int(item["chunk_count"]),
                    item.get("upload_batch"),
                    int(item.get("size_bytes", 0)),
                    created_at,
                )
                for item in documents
            ],
        )


def list_knowledge_bases(
    *, owner_id: str | None = None, limit: int = 50, offset: int = 0
) -> list[dict]:
    init_database()
    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT id, filename, chunk_count, created_at, owner_id
            FROM knowledge_bases
            WHERE (? IS NULL OR owner_id = ?)
            ORDER BY created_at DESC
            LIMIT ? OFFSET ?
            """,
            (owner_id, owner_id, limit, offset),
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
    public_ids: frozenset[str],
    *,
    owner_id: str | None = None,
    limit: int = 50,
    offset: int = 0,
) -> list[dict]:
    """返回当前访客自己的知识库及管理员明确共享的只读知识库。"""
    valid_ids = sorted(
        item for item in public_ids if _KNOWLEDGE_BASE_ID.fullmatch(item)
    )
    if not valid_ids and owner_id is None:
        return []
    clauses: list[str] = []
    parameters: list[str | int] = []
    if owner_id is not None:
        clauses.append("owner_id = ?")
        parameters.append(owner_id)
    if valid_ids:
        placeholders = ",".join("?" for _ in valid_ids)
        clauses.append(f"id IN ({placeholders})")
        parameters.extend(valid_ids)
    init_database()
    with get_connection() as connection:
        rows = connection.execute(
            f"""
            SELECT id, filename, chunk_count, created_at, owner_id
            FROM knowledge_bases
            WHERE {' OR '.join(clauses)}
            ORDER BY created_at DESC
            LIMIT ? OFFSET ?
            """,
            (*parameters, limit, offset),
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
            SELECT id, filename, chunk_count, created_at, owner_id
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
    knowledge_base_id: str,
    public_ids: frozenset[str],
    *,
    owner_id: str | None = None,
) -> dict | None:
    """校验资源属于当前访客或明确公开，未授权资源统一视为不存在。"""
    if not _KNOWLEDGE_BASE_ID.fullmatch(knowledge_base_id):
        return None
    knowledge_base = get_knowledge_base(knowledge_base_id)
    if knowledge_base is None:
        return None
    if knowledge_base["owner_id"] == owner_id or knowledge_base_id in public_ids:
        return knowledge_base
    return None


def get_owner_knowledge_base_count(owner_id: str) -> tuple[int, int, int]:
    """返回访客知识库数、文档数和已上传总字节数，用于防止资源滥用。"""
    init_database()
    with get_connection() as connection:
        row = connection.execute(
            """
            SELECT COUNT(DISTINCT bases.id) AS base_count,
                   COUNT(documents.id) AS document_count,
                   COALESCE(SUM(documents.size_bytes), 0) AS total_bytes
            FROM knowledge_bases AS bases
            LEFT JOIN knowledge_documents AS documents
                ON documents.knowledge_base_id = bases.id
            WHERE bases.owner_id = ?
            """,
            (owner_id,),
        ).fetchone()
    return (
        int(row["base_count"]),
        int(row["document_count"]),
        int(row["total_bytes"]),
    )


def knowledge_base_is_owned_by(knowledge_base_id: str, owner_id: str) -> bool:
    """判断知识库是否属于指定身份，不把共享知识库当作可编辑资源。"""
    init_database()
    with get_connection() as connection:
        row = connection.execute(
            "SELECT 1 FROM knowledge_bases WHERE id = ? AND owner_id = ?",
            (knowledge_base_id, owner_id),
        ).fetchone()
    return row is not None


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
                (knowledge_base_id, filename, chunk_count, upload_batch, size_bytes, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    knowledge_base_id,
                    str(item["filename"]),
                    int(item["chunk_count"]),
                    item.get("upload_batch"),
                    int(item.get("size_bytes", 0)),
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
