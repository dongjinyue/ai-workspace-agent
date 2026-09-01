import json
from datetime import datetime, timezone
from uuid import uuid4

from app.memory.database import get_connection, init_database


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def create_conversation(title: str = "新会话") -> str:
    init_database()
    conversation_id = uuid4().hex
    now = _now()
    with get_connection() as connection:
        connection.execute(
            """
            INSERT INTO conversations (id, title, created_at, updated_at)
            VALUES (?, ?, ?, ?)
            """,
            (conversation_id, title, now, now),
        )
    return conversation_id


def list_conversations(*, limit: int = 50, offset: int = 0) -> list[dict[str, str | int]]:
    """返回会话摘要，最近使用的会话排在最前面。"""
    init_database()
    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT c.id, c.title, c.created_at, c.updated_at,
                   COUNT(m.id) AS message_count
            FROM conversations AS c
            LEFT JOIN messages AS m ON m.conversation_id = c.id
            GROUP BY c.id
            ORDER BY c.updated_at DESC
            LIMIT ? OFFSET ?
            """,
            (limit, offset),
        ).fetchall()
    return [dict(row) for row in rows]


def rename_conversation(conversation_id: str, title: str) -> bool:
    with get_connection() as connection:
        cursor = connection.execute(
            "UPDATE conversations SET title = ?, updated_at = ? WHERE id = ?",
            (title, _now(), conversation_id),
        )
    return cursor.rowcount > 0


def delete_conversation(conversation_id: str) -> bool:
    with get_connection() as connection:
        cursor = connection.execute(
            "DELETE FROM conversations WHERE id = ?", (conversation_id,)
        )
    return cursor.rowcount > 0


def use_first_message_as_title(conversation_id: str, message: str) -> None:
    """仅替换默认标题，保留用户主动修改过的标题。"""
    title = message.strip().replace("\n", " ")[:28] or "新会话"
    with get_connection() as connection:
        connection.execute(
            "UPDATE conversations SET title = ? WHERE id = ? AND title = '新会话'",
            (title, conversation_id),
        )


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


def delete_message(message_id: int) -> None:
    """删除指定消息，用于 Agent 失败时回滚尚未完成的用户轮次。"""
    with get_connection() as connection:
        connection.execute("DELETE FROM messages WHERE id = ?", (message_id,))


def save_assistant_message_with_trace(
    conversation_id: str,
    content: str,
    trace: dict,
) -> int:
    """在同一事务中保存助手回答及其安全执行元数据。"""
    now = _now()
    with get_connection() as connection:
        cursor = connection.execute(
            """
            INSERT INTO messages (conversation_id, role, content, created_at)
            VALUES (?, 'assistant', ?, ?)
            """,
            (conversation_id, content, now),
        )
        message_id = int(cursor.lastrowid)
        connection.execute(
            """
            INSERT INTO agent_runs (
                assistant_message_id, conversation_id, trace_json, created_at
            ) VALUES (?, ?, ?, ?)
            """,
            (
                message_id,
                conversation_id,
                json.dumps(trace, ensure_ascii=False),
                now,
            ),
        )
        connection.execute(
            "UPDATE conversations SET updated_at = ? WHERE id = ?",
            (now, conversation_id),
        )
    return message_id


def get_messages_with_traces(
    conversation_id: str,
    *,
    limit: int = 100,
    offset: int = 0,
) -> list[dict]:
    """读取用户可见消息，并为助手回答附加可公开的执行轨迹。"""
    init_database()
    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT page.role, page.content, page.created_at, page.trace_json
            FROM (
                SELECT m.id, m.role, m.content, m.created_at, r.trace_json
                FROM messages AS m
                LEFT JOIN agent_runs AS r ON r.assistant_message_id = m.id
                WHERE m.conversation_id = ?
                ORDER BY m.id DESC
                LIMIT ? OFFSET ?
            ) AS page
            ORDER BY page.id ASC
            """,
            (conversation_id, limit, offset),
        ).fetchall()

    messages = []
    for row in rows:
        item = {
            "role": row["role"],
            "content": row["content"],
            "created_at": row["created_at"],
        }
        if row["trace_json"]:
            item["trace"] = json.loads(row["trace_json"])
        messages.append(item)
    return messages


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
