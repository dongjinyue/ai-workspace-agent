import re


def split_text(
    text: str,
    chunk_size: int = 500,
    overlap: int = 80,
) -> list[str]:
    """把文档切成带重叠的文本块，减少答案被切断的情况。"""
    cleaned_text = text.strip()
    if not cleaned_text:
        return []
    if chunk_size <= 0:
        raise ValueError("chunk_size 必须大于 0")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap 必须大于等于 0 且小于 chunk_size")

    chunks = []
    step = chunk_size - overlap
    for start in range(0, len(cleaned_text), step):
        chunk = cleaned_text[start : start + chunk_size].strip()
        if chunk:
            chunks.append(chunk)
        if start + chunk_size >= len(cleaned_text):
            break

    return chunks


def _search_terms(text: str) -> set[str]:
    """提取英文单词、数字和中文二元片段，支持简单中文检索。"""
    normalized = text.lower()
    terms = set(re.findall(r"[a-z0-9]+", normalized))

    for sequence in re.findall(r"[\u4e00-\u9fff]+", normalized):
        if len(sequence) == 1:
            terms.add(sequence)
        else:
            terms.update(sequence[index : index + 2] for index in range(len(sequence) - 1))

    return terms


def retrieve_chunks(
    chunks: list[str],
    query: str,
    top_k: int = 3,
) -> list[str]:
    query_terms = _search_terms(query)
    if not query_terms:
        return []

    scored_chunks = []

    for chunk in chunks:
        chunk_terms = _search_terms(chunk)
        score = len(query_terms & chunk_terms)
        scored_chunks.append((score, chunk))

    scored_chunks.sort(key=lambda item: item[0], reverse=True)

    return [
        chunk
        for score, chunk in scored_chunks[:top_k]
        if score > 0
    ]
