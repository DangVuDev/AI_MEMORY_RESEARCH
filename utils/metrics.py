"""
Shared evaluation metrics used across scenarios.

The project does not include gold chunk annotations, so precision/recall are
estimated with a deterministic lexical relevance heuristic to keep comparisons
fair across systems.
"""
from __future__ import annotations

import re
from typing import Iterable

TOKEN_RE = re.compile(r"\w+", re.UNICODE)
STOPWORDS = {
    "la", "va", "cua", "cho", "mot", "nhung", "duoc", "trong", "voi", "cung",
    "the", "nay", "do", "khi", "neu", "thi", "co", "khong", "gi", "nao", "bao",
    "nhieu", "it", "tai", "tu", "den", "ve", "o", "bi", "da", "se", "can", "la",
}


def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in TOKEN_RE.findall(text)]


def _keyword_set(text: str) -> set[str]:
    return {t for t in _tokenize(text) if len(t) > 2 and t not in STOPWORDS}


def lexical_relevance(doc: str, question: str) -> float:
    q = _keyword_set(question)
    if not q:
        return 0.0
    d = _keyword_set(doc)
    if not d:
        return 0.0
    overlap = len(q & d)
    return overlap / len(q)


def is_relevant(doc: str, question: str, threshold: float = 0.20) -> bool:
    return lexical_relevance(doc, question) >= threshold


def context_precision(retrieved_docs: Iterable[str], question: str) -> float:
    docs = list(retrieved_docs)
    if not docs:
        return 0.0
    good = sum(1 for d in docs if is_relevant(d, question))
    return good / len(docs)


def context_recall(retrieved_docs: Iterable[str], all_candidates: Iterable[str], question: str) -> float:
    docs = list(retrieved_docs)
    relevant_pool = [d for d in all_candidates if is_relevant(d, question)]
    if not relevant_pool:
        return 0.0
    hit = sum(1 for d in docs if is_relevant(d, question))
    return min(1.0, hit / len(relevant_pool))


def faithfulness(answer: str, context: str) -> float:
    ans_tokens = _keyword_set(answer)
    if not ans_tokens:
        return 0.0
    ctx_tokens = _keyword_set(context)
    if not ctx_tokens:
        return 0.0
    supported = len(ans_tokens & ctx_tokens)
    return supported / len(ans_tokens)


def hallucination_rate(answer: str, context: str) -> float:
    return max(0.0, 1.0 - faithfulness(answer, context))


def evaluate_common_metrics(question: str, retrieved_docs: list[str], all_candidates: list[str], answer: str, context: str) -> dict[str, float]:
    return {
        "context_precision": float(context_precision(retrieved_docs, question)),
        "context_recall": float(context_recall(retrieved_docs, all_candidates, question)),
        "faithfulness": float(faithfulness(answer, context)),
        "hallucination_rate": float(hallucination_rate(answer, context)),
    }
