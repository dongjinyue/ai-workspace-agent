import os
import tempfile
from pathlib import Path


# 测试不能读写正式的持久化知识库。
os.environ.setdefault(
    "CHROMA_PATH",
    str(Path(tempfile.gettempdir()) / "ai-workspace-agent-pytest-chroma"),
)
os.environ.setdefault(
    "APP_DATABASE_PATH",
    str(Path(tempfile.mkdtemp(prefix="ai-workspace-agent-pytest-")) / "app.db"),
)
