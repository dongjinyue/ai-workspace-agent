import json
from datetime import datetime, timedelta, timezone
from uuid import uuid4

from app.memory.database import get_connection, init_database


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def get_guest_ai_request_count(session_hash: str, local_date: str) -> int:
    """读取指定访客会话在上海自然日内已接受的 AI 请求数。"""
    init_database()
    with get_connection() as connection:
        row = connection.execute(
            "SELECT COUNT(*) AS total FROM guest_ai_events "
            "WHERE session_hash = ? AND local_date = ?",
            (session_hash, local_date),
        ).fetchone()
    return int(row["total"])


def reserve_guest_ai_event(
    session_hash: str,
    ip_hash: str,
    now: datetime,
    local_date: str,
    day_reset: datetime,
    *,
    session_limit: int,
    ip_minute_limit: int,
    ip_day_limit: int,
) -> tuple[bool, int, datetime | None]:
    """串行检查所有访客额度并记账，避免并发请求抢占同一个剩余额度。"""
    init_database()
    now_utc = now.astimezone(timezone.utc)
    now_iso = now_utc.isoformat(timespec="microseconds")
    minute_cutoff = (now_utc - timedelta(seconds=60)).isoformat(
        timespec="microseconds"
    )
    retention_cutoff = (now_utc - timedelta(days=2)).isoformat(
        timespec="microseconds"
    )

    with get_connection() as connection:
        # 在统计之前获取 SQLite 写锁，让检查+插入作为一个原子操作。
        connection.execute("BEGIN IMMEDIATE")
        connection.execute(
            "DELETE FROM guest_ai_events WHERE requested_at < ?",
            (retention_cutoff,),
        )
        session_count = int(
            connection.execute(
                "SELECT COUNT(*) AS total FROM guest_ai_events "
                "WHERE session_hash = ? AND local_date = ?",
                (session_hash, local_date),
            ).fetchone()["total"]
        )
        ip_day_count = int(
            connection.execute(
                "SELECT COUNT(*) AS total FROM guest_ai_events "
                "WHERE ip_hash = ? AND local_date = ?",
                (ip_hash, local_date),
            ).fetchone()["total"]
        )
        minute_rows = connection.execute(
            "SELECT requested_at FROM guest_ai_events "
            "WHERE ip_hash = ? AND requested_at > ? AND requested_at <= ? "
            "ORDER BY requested_at ASC",
            (ip_hash, minute_cutoff, now_iso),
        ).fetchall()

        retry_at: datetime | None = None
        if session_count >= session_limit or ip_day_count >= ip_day_limit:
            retry_at = day_reset
        if len(minute_rows) >= ip_minute_limit:
            minute_reset = datetime.fromisoformat(
                minute_rows[0]["requested_at"]
            ) + timedelta(seconds=60)
            retry_at = max(retry_at, minute_reset) if retry_at else minute_reset
        if retry_at is not None:
            return False, session_count, retry_at

        connection.execute(
            "INSERT INTO guest_ai_events "
            "(session_hash, ip_hash, requested_at, local_date) VALUES (?, ?, ?, ?)",
            (session_hash, ip_hash, now_iso, local_date),
        )
        return True, session_count + 1, None


def create_conversation(title: str = "新会话", *, owner_id: str) -> str:
    init_database()
    conversation_id = uuid4().hex
    now = _now()
    with get_connection() as connection:
        connection.execute(
            """
            INSERT INTO conversations (id, title, created_at, updated_at, owner_id)
            VALUES (?, ?, ?, ?, ?)
            """,
            (conversation_id, title, now, now, owner_id),
        )
    return conversation_id


def list_conversations(
    *, owner_id: str, limit: int = 50, offset: int = 0
) -> list[dict[str, str | int]]:
    """返回会话摘要，最近使用的会话排在最前面。"""
    init_database()
    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT c.id, c.title, c.created_at, c.updated_at,
                   COUNT(m.id) AS message_count
            FROM conversations AS c
            LEFT JOIN messages AS m ON m.conversation_id = c.id
            WHERE c.owner_id = ?
            GROUP BY c.id
            ORDER BY c.updated_at DESC
            LIMIT ? OFFSET ?
            """,
            (owner_id, limit, offset),
        ).fetchall()
    return [dict(row) for row in rows]


def rename_conversation(conversation_id: str, title: str, *, owner_id: str) -> bool:
    with get_connection() as connection:
        cursor = connection.execute(
            "UPDATE conversations SET title = ?, updated_at = ? WHERE id = ? AND owner_id = ?",
            (title, _now(), conversation_id, owner_id),
        )
    return cursor.rowcount > 0


def delete_conversation(conversation_id: str, *, owner_id: str) -> bool:
    with get_connection() as connection:
        cursor = connection.execute(
            "DELETE FROM conversations WHERE id = ? AND owner_id = ?",
            (conversation_id, owner_id),
        )
    return cursor.rowcount > 0


def use_first_message_as_title(
    conversation_id: str, message: str, *, owner_id: str
) -> None:
    """仅替换默认标题，保留用户主动修改过的标题。"""
    title = message.strip().replace("\n", " ")[:28] or "新会话"
    with get_connection() as connection:
        connection.execute(
            "UPDATE conversations SET title = ? WHERE id = ? AND owner_id = ? AND title = '新会话'",
            (title, conversation_id, owner_id),
        )


def conversation_exists(conversation_id: str, *, owner_id: str) -> bool:
    init_database()
    with get_connection() as connection:
        row = connection.execute(
            "SELECT 1 FROM conversations WHERE id = ? AND owner_id = ?",
            (conversation_id, owner_id),
        ).fetchone()
    return row is not None


def save_message(
    conversation_id: str, role: str, content: str, *, owner_id: str
) -> int:
    if role not in {"user", "assistant"}:
        raise ValueError(f"不允许保存的消息角色：{role}")
    now = _now()
    with get_connection() as connection:
        cursor = connection.execute(
            """
            INSERT INTO messages (conversation_id, role, content, created_at)
            SELECT ?, ?, ?, ? WHERE EXISTS (
                SELECT 1 FROM conversations WHERE id = ? AND owner_id = ?
            )
            """,
            (conversation_id, role, content, now, conversation_id, owner_id),
        )
        if cursor.rowcount == 0:
            raise LookupError("会话不存在")
        connection.execute(
            "UPDATE conversations SET updated_at = ? WHERE id = ? AND owner_id = ?",
            (now, conversation_id, owner_id),
        )
    return int(cursor.lastrowid)


def delete_message(message_id: int, *, owner_id: str) -> None:
    """删除指定消息，用于 Agent 失败时回滚尚未完成的用户轮次。"""
    with get_connection() as connection:
        connection.execute(
            "DELETE FROM messages WHERE id = ? AND conversation_id IN "
            "(SELECT id FROM conversations WHERE owner_id = ?)",
            (message_id, owner_id),
        )


def save_assistant_message_with_trace(
    conversation_id: str,
    content: str,
    trace: dict,
    *,
    owner_id: str,
) -> int:
    """在同一事务中保存助手回答及其安全执行元数据。"""
    now = _now()
    with get_connection() as connection:
        cursor = connection.execute(
            """
            INSERT INTO messages (conversation_id, role, content, created_at)
            SELECT ?, 'assistant', ?, ? WHERE EXISTS (
                SELECT 1 FROM conversations WHERE id = ? AND owner_id = ?
            )
            """,
            (conversation_id, content, now, conversation_id, owner_id),
        )
        if cursor.rowcount == 0:
            raise LookupError("会话不存在")
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
            "UPDATE conversations SET updated_at = ? WHERE id = ? AND owner_id = ?",
            (now, conversation_id, owner_id),
        )
    return message_id


def get_messages_with_traces(
    conversation_id: str,
    *,
    limit: int = 100,
    offset: int = 0,
    owner_id: str,
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
                JOIN conversations AS c ON c.id = m.conversation_id
                WHERE m.conversation_id = ? AND c.owner_id = ?
                ORDER BY m.id DESC
                LIMIT ? OFFSET ?
            ) AS page
            ORDER BY page.id ASC
            """,
            (conversation_id, owner_id, limit, offset),
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
    owner_id: str,
    limit: int | None = None,
) -> list[dict[str, str]]:
    init_database()
    with get_connection() as connection:
        if limit is None:
            rows = connection.execute(
                """
                SELECT role, content, created_at
                FROM messages
                WHERE conversation_id = ? AND EXISTS (
                    SELECT 1 FROM conversations WHERE id = ? AND owner_id = ?
                )
                ORDER BY id ASC
                """,
                (conversation_id, conversation_id, owner_id),
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
                    WHERE conversation_id = ? AND EXISTS (
                        SELECT 1 FROM conversations WHERE id = ? AND owner_id = ?
                    )
                    ORDER BY id DESC
                    LIMIT ?
                )
                ORDER BY id ASC
                """,
                (conversation_id, conversation_id, owner_id, limit),
            ).fetchall()
    return [dict(row) for row in rows]
