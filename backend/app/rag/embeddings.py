import os
from http import HTTPStatus
from typing import Literal

import dashscope
from dashscope import TextEmbedding


EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-v4")
EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION", "1024"))
EMBEDDING_BATCH_SIZE = 10


def embed_texts(
    texts: list[str],
    *,
    text_type: Literal["document", "query"] = "document",
) -> list[list[float]]:
    """批量生成向量，并保持结果与输入文本的顺序一致。"""
    if not texts:
        return []

    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise RuntimeError("DASHSCOPE_API_KEY 未配置")

    base_http_api_url = os.getenv("DASHSCOPE_HTTP_BASE_URL")
    if base_http_api_url:
        dashscope.base_http_api_url = base_http_api_url

    all_embeddings: list[list[float]] = []

    for start in range(0, len(texts), EMBEDDING_BATCH_SIZE):
        batch = texts[start : start + EMBEDDING_BATCH_SIZE]
        response = TextEmbedding.call(
            model=EMBEDDING_MODEL,
            input=batch,
            api_key=api_key,
            text_type=text_type,
            dimension=EMBEDDING_DIMENSION,
            output_type="dense",
        )

        if response.status_code != HTTPStatus.OK:
            message = getattr(response, "message", "未知错误")
            raise RuntimeError(f"Embedding 服务调用失败：{message}")

        ordered = sorted(
            response.output["embeddings"],
            key=lambda item: item["text_index"],
        )
        if len(ordered) != len(batch):
            raise RuntimeError("Embedding 服务返回的向量数量与输入不一致")

        all_embeddings.extend(item["embedding"] for item in ordered)

    return all_embeddings
