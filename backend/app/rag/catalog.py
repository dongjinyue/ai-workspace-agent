from datetime import datetime, timezone

from app.memory.database import get_connection, init_database


def register_knowledge_base(
    knowledge_base_id: str,
    filename: str,
    chunk_count: int,
) -> None:
    """保存可恢复的知识库元数据，向量正文仍由 Chroma 管理。"""
    init_database()
    with get_connection() as connection:
        connection.execute(
            """
            INSERT INTO knowledge_bases (id, filename, chunk_count, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (
                knowledge_base_id,
                filename,
                chunk_count,
                datetime.now(timezone.utc).isoformat(),
            ),
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
    return [dict(row) for row in rows]


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
