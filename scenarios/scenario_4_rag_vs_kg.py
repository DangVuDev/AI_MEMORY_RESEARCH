"""
Scenario 4: RAG vs Knowledge Graph
====================================
So sánh:
  A_RAG_Dense  – Dense retrieval (cosine)
  B_RAG_BM25   – BM25-style keyword retrieval (TF-IDF approximation)
  C_KG_Entity  – KG entity-matching
  D_KG_Hybrid  – KG entity + semantic fallback (= retrieve_kg)
"""
import json
import math
import time
from collections import Counter
from typing import List

import numpy as np
from scipy import stats
from sentence_transformers import util

from retriever import load_corpus, load_questions, get_embedder, retrieve_kg
from openai_client import generate_answer, evaluate_relevance
from config import RESULTS_DIR, TOP_K
from utils.metrics import evaluate_common_metrics


# ── BM25-style retrieval ───────────────────────────────────────────────────────

def _bm25_scores(corpus: List[str], query: str, k1: float = 1.5, b: float = 0.75) -> List[float]:
    """Approximate BM25 without external lib."""
    tokens = [doc.lower().split() for doc in corpus]
    avg_dl = np.mean([len(t) for t in tokens])
    query_tokens = query.lower().split()
    N = len(corpus)
    scores = []
    for i, doc_tokens in enumerate(tokens):
        tf_map = Counter(doc_tokens)
        dl = len(doc_tokens)
        score = 0.0
        for qt in query_tokens:
            tf  = tf_map.get(qt, 0)
            df  = sum(1 for t in tokens if qt in t)
            idf = math.log((N - df + 0.5) / (df + 0.5) + 1)
            score += idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / avg_dl))
        scores.append(score)
    return scores


def retrieve_bm25(corpus: List[str], query: str, k: int = TOP_K) -> List[str]:
    scores = _bm25_scores(corpus, query)
    top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    return [corpus[i] for i in top_idx]


def retrieve_dense(corpus: List[str], query: str, corpus_embs=None, k: int = TOP_K) -> List[str]:
    embedder = get_embedder()
    q_emb    = embedder.encode(query, convert_to_tensor=True)
    if corpus_embs is None:
        corpus_embs = embedder.encode(corpus, convert_to_tensor=True)
    scores  = util.cos_sim(q_emb, corpus_embs)[0]
    top_idx = scores.topk(min(k, len(corpus))).indices.tolist()
    return [corpus[i] for i in top_idx]


def retrieve_kg_entity_only(corpus: List[str], query: str, k: int = TOP_K) -> List[str]:
    """KG entity-only (no semantic fallback)."""
    KG_ENTITIES = ["Alice", "Bob", "CEO", "CTO", "CloudCore", "DataMind",
                   "Engineering", "Product", "Marketing", "HR"]
    matched = [e for e in KG_ENTITIES if e.lower() in query.lower()]
    if matched:
        scored = [(doc, sum(e.lower() in doc.lower() for e in matched))
                  for doc in corpus]
        scored = [(d, s) for d, s in scored if s > 0]
        scored.sort(key=lambda x: x[1], reverse=True)
        if scored:
            return [d for d, _ in scored[:k]]
    # no entities found → return first k
    return corpus[:k]


# ── Main ──────────────────────────────────────────────────────────────────────

def run() -> dict:
    print("\n" + "=" * 70)
    print("SCENARIO 4 — RAG vs Knowledge Graph")
    print("=" * 70)

    corpus    = load_corpus()
    questions = load_questions()
    print(f"\n    corpus={len(corpus)} docs  |  questions={len(questions)}\n")

    print("[1] Pre-computing corpus embeddings for dense retrieval...")
    from retriever import encode_corpus
    corpus_embs = encode_corpus(corpus)
    print(f"    Done.\n")

    systems = {
        "A_RAG_Dense": lambda q: retrieve_dense(corpus, q, corpus_embs=corpus_embs),
        "B_RAG_BM25":  lambda q: retrieve_bm25(corpus, q),
        "C_KG_Entity": lambda q: retrieve_kg_entity_only(corpus, q),
        "D_KG_Hybrid": lambda q: retrieve_kg(corpus, q, corpus_embs=corpus_embs),
    }

    print("[2] Running all systems...\n")
    all_results: dict[str, dict] = {}

    for sys_name, retriever_fn in systems.items():
        print(f"  ▶ {sys_name}")
        relevances, latencies = [], []
        precisions, recalls, faiths, hallucs = [], [], [], []

        for i, (question, _) in enumerate(questions, 1):
            t0 = time.time()
            docs    = retriever_fn(question)
            context = "\n---\n".join(docs)
            answer  = generate_answer(question, context)
            score   = evaluate_relevance(question, context, answer)
            m = evaluate_common_metrics(question, docs, corpus, answer, context)
            elapsed = (time.time() - t0) * 1000
            relevances.append(score)
            latencies.append(elapsed)
            precisions.append(m["context_precision"])
            recalls.append(m["context_recall"])
            faiths.append(m["faithfulness"])
            hallucs.append(m["hallucination_rate"])

            if i % 10 == 0 or i == len(questions):
                print(f"    {i:>2}/{len(questions)}  avg_rel={np.mean(relevances):.3f}")

        all_results[sys_name] = {
            "avg_relevance": float(np.mean(relevances)),
            "avg_latency_ms": float(np.mean(latencies)),
            "avg_context_precision": float(np.mean(precisions)),
            "avg_context_recall": float(np.mean(recalls)),
            "avg_faithfulness": float(np.mean(faiths)),
            "avg_hallucination_rate": float(np.mean(hallucs)),
            "relevances": [float(r) for r in relevances],
        }
        print(f"    ✓ avg_relevance={all_results[sys_name]['avg_relevance']:.3f}\n")

    # ── Stats ──────────────────────────────────────────────────────────────────
    print("[3] Statistical Analysis")
    a_scores = np.array(all_results["A_RAG_Dense"]["relevances"])
    stats_output = {}

    for name in ["B_RAG_BM25", "C_KG_Entity", "D_KG_Hybrid"]:
        b_scores = np.array(all_results[name]["relevances"])
        t_stat, p_val = stats.ttest_rel(a_scores, b_scores)
        pooled = np.sqrt((np.var(a_scores, ddof=1) + np.var(b_scores, ddof=1)) / 2)
        d = float((np.mean(b_scores) - np.mean(a_scores)) / pooled) if pooled > 0 else 0.0
        imp = (np.mean(b_scores) - np.mean(a_scores)) / max(np.mean(a_scores), 1e-9) * 100
        stats_output[f"{name}_vs_A"] = {
            "t_stat": float(t_stat), "p_value": float(p_val),
            "cohens_d": d, "improvement_pct": float(imp), "significant": bool(p_val < 0.05),
        }
        print(f"  {name} vs A_RAG_Dense: t={t_stat:.3f}, p={p_val:.4f}, +{imp:.1f}%")

    best = max(all_results, key=lambda k: all_results[k]["avg_relevance"])
    print(f"\n  Best system: {best} (avg={all_results[best]['avg_relevance']:.3f})")

    output = {
        "scenario": "4_rag_vs_knowledge_graph",
        "best_system": best,
        "systems": all_results,
        "statistical_tests": stats_output,
    }
    out_dir = RESULTS_DIR / "scenario4"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n  ✓ Saved → {out_file}\n")
    return output


if __name__ == "__main__":
    run()
