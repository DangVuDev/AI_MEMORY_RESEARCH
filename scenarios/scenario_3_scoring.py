"""
Scenario 3: Memory Scoring Strategies
======================================
So sánh các cách chấm điểm ưu tiên tài liệu khi rerank:

  A_Recency    – ưu tiên tài liệu mới (index cao = mới hơn)
  B_Importance – ưu tiên tài liệu dài/phong phú (proxy = độ dài)
  C_Relevance  – thuần cosine similarity
  D_Combined   – kết hợp cả 3 (weighted sum)
"""
import json
import time
from itertools import product
from typing import List

import numpy as np
from scipy import stats
from sentence_transformers import util

from retriever import load_corpus, load_questions, get_embedder
from openai_client import generate_answer, evaluate_relevance
from config import RESULTS_DIR, TOP_K
from utils.metrics import evaluate_common_metrics


# ── Scoring helpers ────────────────────────────────────────────────────────────

def score_recency(idx: int, total: int) -> float:
    """Tài liệu có index càng cao → càng mới → điểm cao."""
    return idx / max(total - 1, 1)


def score_importance(doc: str) -> float:
    """Proxy bằng độ dài chuẩn hoá (0-1)."""
    return min(len(doc) / 500.0, 1.0)


def retrieve_scored(
    corpus: List[str],
    query: str,
    mode: str,            # "recency" | "importance" | "relevance" | "combined"
    k: int = TOP_K,
    corpus_embs=None,
    combined_weights: tuple[float, float, float] = (0.5, 0.3, 0.2),
) -> List[str]:
    embedder = get_embedder()
    q_emb    = embedder.encode(query, convert_to_tensor=True)
    c_embs   = corpus_embs if corpus_embs is not None else embedder.encode(corpus, convert_to_tensor=True)
    cos_sims = util.cos_sim(q_emb, c_embs)[0].tolist()
    total    = len(corpus)
    alpha, beta, gamma = combined_weights

    scored = []
    for i, doc in enumerate(corpus):
        rel  = cos_sims[i]
        rec  = score_recency(i, total)
        imp  = score_importance(doc)

        if mode == "recency":
            final = rec
        elif mode == "importance":
            final = imp
        elif mode == "relevance":
            final = rel
        else:  # combined
            final = alpha * rec + beta * imp + gamma * rel

        scored.append((doc, final))

    scored.sort(key=lambda x: x[1], reverse=True)
    return [d for d, _ in scored[:k]]


def grid_search_weights(
    corpus: List[str],
    questions: List[tuple[str, str]],
    corpus_embs,
    k: int = TOP_K,
) -> tuple[tuple[float, float, float], float, list[dict]]:
    """Find best (alpha, beta, gamma) with alpha+beta+gamma=1 using retrieval proxy."""
    embedder = get_embedder()
    total = len(corpus)
    recency = np.array([score_recency(i, total) for i in range(total)], dtype=np.float32)
    importance = np.array([score_importance(doc) for doc in corpus], dtype=np.float32)

    candidates: list[tuple[float, float, float]] = []
    weights = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    for a, b, g in product(weights, repeat=3):
        if abs((a + b + g) - 1.0) <= 0.01:
            candidates.append((a, b, g))

    query_embs = embedder.encode([q for q, _ in questions], convert_to_tensor=True, show_progress_bar=False)
    rel_matrix = util.cos_sim(query_embs, corpus_embs).cpu().numpy()

    scored_candidates = []
    for alpha, beta, gamma in candidates:
        # Weighted rank score for docs per query.
        rank_base = alpha * recency + beta * importance
        top_rel_means = []
        for i in range(len(questions)):
            rel = rel_matrix[i]
            final = rank_base + gamma * rel
            top_idx = np.argsort(-final)[:k]
            top_rel_means.append(float(np.mean(rel[top_idx])))
        mean_proxy = float(np.mean(top_rel_means))
        scored_candidates.append(
            {"alpha": alpha, "beta": beta, "gamma": gamma, "proxy_retrieval_score": mean_proxy}
        )

    scored_candidates.sort(key=lambda x: x["proxy_retrieval_score"], reverse=True)
    best = scored_candidates[0]
    return (best["alpha"], best["beta"], best["gamma"]), best["proxy_retrieval_score"], scored_candidates[:10]


# ── Main ──────────────────────────────────────────────────────────────────────

def run() -> dict:
    print("\n" + "=" * 70)
    print("SCENARIO 3 — Memory Scoring Strategies")
    print("=" * 70)

    corpus    = load_corpus()
    questions = load_questions()
    print("[0] Pre-computing corpus embeddings...")
    corpus_embs = get_embedder().encode(corpus, convert_to_tensor=True, show_progress_bar=False)

    print("[0.1] Grid search for combined score weights (alpha, beta, gamma)...")
    best_weights, best_proxy, top_grid = grid_search_weights(corpus, questions, corpus_embs, k=TOP_K)
    print(
        f"    Best weights: alpha={best_weights[0]:.1f}, beta={best_weights[1]:.1f}, "
        f"gamma={best_weights[2]:.1f} (proxy={best_proxy:.4f})"
    )
    print(f"\n    corpus={len(corpus)} docs  |  questions={len(questions)}\n")

    systems = {
        "A_Recency":    "recency",
        "B_Importance": "importance",
        "C_Relevance":  "relevance",
        "D_Combined":   "combined",
    }

    print("[1] Running all scoring strategies...\n")
    all_results: dict[str, dict] = {}

    for sys_name, mode in systems.items():
        print(f"  ▶ {sys_name}")
        relevances, latencies = [], []
        precisions, recalls, faiths, hallucs = [], [], [], []

        for i, (question, _) in enumerate(questions, 1):
            t0 = time.time()
            if mode == "combined":
                docs = retrieve_scored(
                    corpus,
                    question,
                    mode=mode,
                    k=TOP_K,
                    corpus_embs=corpus_embs,
                    combined_weights=best_weights,
                )
            else:
                docs = retrieve_scored(
                    corpus,
                    question,
                    mode=mode,
                    k=TOP_K,
                    corpus_embs=corpus_embs,
                )
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
            "mode": mode,
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
    print("[2] Statistical Analysis")
    a_scores = np.array(all_results["A_Recency"]["relevances"])
    stats_output = {}

    for name in ["B_Importance", "C_Relevance", "D_Combined"]:
        b_scores = np.array(all_results[name]["relevances"])
        t_stat, p_val = stats.ttest_rel(a_scores, b_scores)
        pooled = np.sqrt((np.var(a_scores, ddof=1) + np.var(b_scores, ddof=1)) / 2)
        d = float((np.mean(b_scores) - np.mean(a_scores)) / pooled) if pooled > 0 else 0.0
        imp = (np.mean(b_scores) - np.mean(a_scores)) / max(np.mean(a_scores), 1e-9) * 100
        stats_output[f"{name}_vs_A"] = {
            "t_stat": float(t_stat), "p_value": float(p_val),
            "cohens_d": d, "improvement_pct": float(imp), "significant": bool(p_val < 0.05),
        }
        print(f"  {name} vs A_Recency: t={t_stat:.3f}, p={p_val:.4f}, +{imp:.1f}%")

    best = max(all_results, key=lambda k: all_results[k]["avg_relevance"])
    print(f"\n  Best scoring: {best} (avg={all_results[best]['avg_relevance']:.3f})")

    output = {
        "scenario": "3_memory_scoring_strategies",
        "best_strategy": best,
        "best_combined_weights": {
            "alpha_recency": float(best_weights[0]),
            "beta_importance": float(best_weights[1]),
            "gamma_relevance": float(best_weights[2]),
            "grid_search_proxy_score": float(best_proxy),
        },
        "grid_search_top10": top_grid,
        "systems": all_results,
        "statistical_tests": stats_output,
    }
    out_dir = RESULTS_DIR / "scenario3"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n  ✓ Saved → {out_file}\n")
    return output


if __name__ == "__main__":
    run()
