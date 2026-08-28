from datetime import datetime, timezone
from uuid import uuid4

from app.memory.database import get_connection, init_database


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def create_conversation() -> str:
    init_database()
    conversation_id = uuid4().hex
    now = _now()
    with get_connection() as connection:
        connection.execute(
            """
            INSERT INTO conversations (id, created_at, updated_at)
            VALUES (?, ?, ?)
            """,
            (conversation_id, now, now),
        )
    return conversation_id


def conversation_exists(conversation_id: str) -> bool:
    init_database()
    with get_connection() as connection:
        row = connection.execute(
            "SELECT 1 FROM conversations WHERE id = ?",
            (conversation_id,),
        ).fetchone()
    return row is not None


def save_message(conversation_id: str, role: str, content: str) -> int:
    if role not in {"user", "assistant"}:
        raise ValueError(f"不允许保存的消息角色：{role}")
    now = _now()
    with get_connection() as connection:
        cursor = connection.execute(
            """
            INSERT INTO messages (conversation_id, role, content, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (conversation_id, role, content, now),
        )
        connection.execute(
            "UPDATE conversations SET updated_at = ? WHERE id = ?",
            (now, conversation_id),
        )
    return int(cursor.lastrowid)


def get_messages(
    conversation_id: str,
    *,
    limit: int | None = None,
) -> list[dict[str, str]]:
    init_database()
    with get_connection() as connection:
        if limit is None:
            rows = connection.execute(
                """
                SELECT role, content, created_at
                FROM messages
                WHERE conversation_id = ?
                ORDER BY id ASC
                """,
                (conversation_id,),
            ).fetchall()
        else:
            if limit < 1 or limit > 100:
                raise ValueError("消息窗口必须在 1 到 100 之间")
            rows = connection.execute(
                """
                SELECT role, content, created_at
                FROM (
                    SELECT id, role, content, created_at
                    FROM messages
                    WHERE conversation_id = ?
                    ORDER BY id DESC
                    LIMIT ?
                )
                ORDER BY id ASC
                """,
                (conversation_id, limit),
            ).fetchall()
    return [dict(row) for row in rows]
