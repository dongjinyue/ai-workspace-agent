import os
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path


BACKEND_DIR = Path(__file__).resolve().parents[2]


def get_database_path() -> Path:
    """允许测试通过环境变量使用独立数据库。"""
    configured = os.getenv("APP_DATABASE_PATH")
    return Path(configured) if configured else BACKEND_DIR / "data" / "app.db"


@contextmanager
def get_connection() -> Iterator[sqlite3.Connection]:
    """提供事务连接，并在成功、异常两种路径上都可靠关闭文件句柄。"""
    database_path = get_database_path()
    database_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(database_path, timeout=10)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    try:
        with connection:
            yield connection
    finally:
        connection.close()


def init_database() -> None:
    """幂等创建会话表和消息表；重复调用不会删除已有数据。"""
    with get_connection() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id TEXT NOT NULL,
                role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
                content TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (conversation_id)
                    REFERENCES conversations(id) ON DELETE CASCADE
            )
            """
        )
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_messages_conversation_id_id
            ON messages (conversation_id, id)
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS agent_runs (
                assistant_message_id INTEGER PRIMARY KEY,
                conversation_id TEXT NOT NULL,
                trace_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (assistant_message_id)
                    REFERENCES messages(id) ON DELETE CASCADE,
                FOREIGN KEY (conversation_id)
                    REFERENCES conversations(id) ON DELETE CASCADE
            )
            """
        )
        # 兼容旧数据库，在不丢失已有会话的前提下补充标题列。
        columns = {
            row["name"]
            for row in connection.execute("PRAGMA table_info(conversations)").fetchall()
        }
        if "title" not in columns:
            connection.execute(
                "ALTER TABLE conversations ADD COLUMN title TEXT NOT NULL DEFAULT '新会话'"
            )
