"""
Retriever - 4 chiến lược retrieval dùng trong các Scenario

A. Random     – baseline
B. Cosine     – standard RAG (dense retrieval)
C. Semantic   – dense retrieval + reranking theo keyword overlap
D. KnowledgeGraph – entity-based matching + semantic fallback
"""
import json
import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer, util

from config import DATA_DIR, EMBED_MODEL, TOP_K

# ── Singleton embedder ─────────────────────────────────────────────────────────
_embedder: SentenceTransformer | None = None


def get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMBED_MODEL)
    return _embedder


# ── Data loading ───────────────────────────────────────────────────────────────
def load_corpus() -> List[str]:
    with open(DATA_DIR / "corpus.json", encoding="utf-8") as f:
        return json.load(f)["corpus"]


def load_questions() -> List[Tuple[str, str]]:
    with open(DATA_DIR / "questions.json", encoding="utf-8") as f:
        data = json.load(f)["questions"]
    return [(q["text"], q["type"]) for q in data]


# ── Pre-compute corpus embeddings once ────────────────────────────────────────
def encode_corpus(corpus: List[str]):
    """Trả về tensor embeddings của toàn bộ corpus. Gọi 1 lần duy nhất."""
    return get_embedder().encode(corpus, convert_to_tensor=True, show_progress_bar=True)


# ── 4 Retrieval Systems ───────────────────────────────────────────────────────
def retrieve_random(corpus: List[str], _query: str, k: int = TOP_K) -> List[str]:
    """A: Random baseline"""
    return random.sample(corpus, min(k, len(corpus)))


def retrieve_cosine(corpus: List[str], query: str,
                    corpus_embs=None, k: int = TOP_K) -> List[str]:
    """B: Cosine similarity (Standard RAG)"""
    emb = get_embedder()
    q_emb = emb.encode(query, convert_to_tensor=True)
    if corpus_embs is None:
        corpus_embs = emb.encode(corpus, convert_to_tensor=True)
    scores = util.cos_sim(q_emb, corpus_embs)[0]
    top_idx = scores.topk(min(k, len(corpus))).indices.tolist()
    return [corpus[i] for i in top_idx]


def retrieve_semantic(corpus: List[str], query: str,
                      corpus_embs=None, k: int = TOP_K) -> List[str]:
    """C: Dense retrieval + keyword reranking"""
    candidates = retrieve_cosine(corpus, query, corpus_embs=corpus_embs, k=k * 2)
    query_words = set(query.lower().split())
    scored = []
    for doc in candidates:
        overlap = len(query_words & set(doc.lower().split()))
        scored.append((doc, overlap))
    scored.sort(key=lambda x: x[1], reverse=True)
    return [d for d, _ in scored[:k]]


def retrieve_kg(corpus: List[str], query: str,
                corpus_embs=None, k: int = TOP_K) -> List[str]:
    """D: Entity-based KG matching, fallback to semantic"""
    KG_ENTITIES = [
        "Alice", "Bob", "CEO", "CTO", "CloudCore", "DataMind",
        "Engineering", "Product", "Marketing", "HR",
    ]
    matched_entities = [e for e in KG_ENTITIES if e.lower() in query.lower()]
    if matched_entities:
        scored = []
        for doc in corpus:
            cnt = sum(e.lower() in doc.lower() for e in matched_entities)
            if cnt > 0:
                scored.append((doc, cnt))
        scored.sort(key=lambda x: x[1], reverse=True)
        if scored:
            return [d for d, _ in scored[:k]]
    # fallback
    return retrieve_semantic(corpus, query, corpus_embs=corpus_embs, k=k)
