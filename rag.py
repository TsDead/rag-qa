"""RAG-ядро: чанкинг → эмбеддинги (fastembed, локально) → поиск по косинусу → ответ LLM.

Индекс держится в памяти. Для портфолио важнее прозрачность, чем масштаб:
видно, как устроен ретривал, а не чёрный ящик.
"""

import numpy as np

import llm

_model = None            # ленивая загрузка модели эмбеддингов
_CHUNKS: list[str] = []  # тексты чанков
_VECS: np.ndarray | None = None  # матрица эмбеддингов (N x dim), нормированная


def _embedder():
    global _model
    if _model is None:
        from fastembed import TextEmbedding
        _model = TextEmbedding("BAAI/bge-small-en-v1.5")
    return _model


def embed(texts: list[str]) -> np.ndarray:
    vecs = np.array(list(_embedder().embed(texts)), dtype=np.float32)
    # L2-нормализация → косинус = скалярное произведение
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / np.clip(norms, 1e-9, None)


def chunk_text(text: str, size: int = 90, overlap: int = 20) -> list[str]:
    """Режем по словам: ~size слов на чанк с перекрытием overlap."""
    words = text.split()
    if not words:
        return []
    chunks, i = [], 0
    step = max(1, size - overlap)
    while i < len(words):
        chunks.append(" ".join(words[i:i + size]))
        i += step
    return chunks


def ingest(text: str) -> int:
    """Проиндексировать документ (заменяет прошлый). Возвращает число чанков."""
    global _CHUNKS, _VECS
    _CHUNKS = chunk_text(text)
    _VECS = embed(_CHUNKS) if _CHUNKS else None
    return len(_CHUNKS)


def retrieve(query: str, k: int = 4):
    """Топ-k чанков по косинусной близости. Возвращает [(score, index, text)]."""
    if _VECS is None or not len(_CHUNKS):
        return []
    q = embed([query])[0]
    sims = _VECS @ q                      # косинус, т.к. всё нормировано
    idx = np.argsort(-sims)[:k]
    return [(float(sims[i]), int(i), _CHUNKS[i]) for i in idx]


SYSTEM = (
    "You answer questions strictly from the provided context passages. "
    "If the answer is not in the context, say you don't know. Be concise. "
    "Cite the passage numbers you used, like [1], [2]."
)


def answer(query: str, k: int = 4):
    """Ответ с опорой на источник. Возвращает {answer, sources}."""
    hits = retrieve(query, k)
    if not hits:
        return {"answer": "Документ не загружен.", "sources": []}
    context = "\n\n".join(f"[{n+1}] {text}" for n, (_, _, text) in enumerate(hits))
    user = f"CONTEXT:\n{context}\n\nQUESTION: {query}\n\nANSWER:"
    reply, _ = llm.chat(
        [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
        max_tokens=500, temperature=0.1,
    )
    sources = [{"n": n + 1, "score": round(s, 3), "text": t[:220]}
               for n, (s, _, t) in enumerate(hits)]
    return {"answer": reply, "sources": sources}


def stats():
    return {"chunks": len(_CHUNKS), "indexed": _VECS is not None}
