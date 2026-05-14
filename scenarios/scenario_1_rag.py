"""
Scenario 1: Retrieval Quality vs Context Size
============================================
Kiểm chứng giả thuyết:
  "Retrieval tốt + context nhỏ (4K)  >  random + context lớn (128K)"

4 Systems:
  A_Random   – random 100 docs (baseline, context lớn)
  B_Cosine   – top-5 cosine similarity (Standard RAG)
  C_Semantic – top-5 cosine + keyword reranking
  D_KG       – entity-based KG + semantic fallback
"""
import json
import time
from pathlib import Path

import numpy as np
from scipy import stats

from retriever import (
    load_corpus, load_questions, encode_corpus,
    retrieve_random, retrieve_cosine, retrieve_semantic, retrieve_kg,
)
from openai_client import generate_answer, evaluate_relevance
from config import RESULTS_DIR, TOP_K
from utils.metrics import evaluate_common_metrics


def run() -> dict:
    print("\n" + "=" * 70)
    print("SCENARIO 1 — Retrieval Quality vs Context Size")
    print("=" * 70)

    # ── Load data ──────────────────────────────────────────────────────────────
    print("\n[1] Loading data...")
    corpus    = load_corpus()
    questions = load_questions()
    print(f"    corpus={len(corpus)} docs  |  questions={len(questions)}")

    # ── Pre-compute corpus embeddings (only once) ──────────────────────────────
    print("\n[2] Pre-computing corpus embeddings...")
    corpus_embs = encode_corpus(corpus)
    print(f"    Done — shape {corpus_embs.shape}")

    # ── Define systems ─────────────────────────────────────────────────────────
    systems = {
        "A_Random":   lambda q: retrieve_random(corpus, q, k=min(100, len(corpus))),
        "B_Cosine":   lambda q: retrieve_cosine(corpus, q, corpus_embs=corpus_embs, k=TOP_K),
        "C_Semantic": lambda q: retrieve_semantic(corpus, q, corpus_embs=corpus_embs, k=TOP_K),
        "D_KG":       lambda q: retrieve_kg(corpus, q, corpus_embs=corpus_embs, k=TOP_K),
    }

    # ── Run ────────────────────────────────────────────────────────────────────
    print("\n[3] Running all systems...\n")
    all_results: dict[str, dict] = {}

    for sys_name, retriever_fn in systems.items():
        print(f"  ▶ {sys_name}")
        relevances, latencies = [], []
        precisions, recalls, faiths, hallucs = [], [], [], []

        for i, (question, qtype) in enumerate(questions, 1):
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

    # ── Statistical analysis ───────────────────────────────────────────────────
    print("[4] Statistical Analysis")
    a_scores = np.array(all_results["A_Random"]["relevances"])
    stats_output = {}

    for name in ["B_Cosine", "C_Semantic", "D_KG"]:
        b_scores = np.array(all_results[name]["relevances"])
        t_stat, p_val = stats.ttest_rel(a_scores, b_scores)
        pooled_std = np.sqrt((np.var(a_scores, ddof=1) + np.var(b_scores, ddof=1)) / 2)
        cohens_d = float((np.mean(b_scores) - np.mean(a_scores)) / pooled_std) if pooled_std > 0 else 0.0
        improvement = (np.mean(b_scores) - np.mean(a_scores)) / max(np.mean(a_scores), 1e-9) * 100
        stats_output[f"{name}_vs_A"] = {
            "t_stat": float(t_stat),
            "p_value": float(p_val),
            "cohens_d": cohens_d,
            "improvement_pct": float(improvement),
            "significant": bool(p_val < 0.05),
        }
        sig = "***" if p_val < 0.05 else "n.s."
        print(f"  {name} vs A: t={t_stat:.3f}, p={p_val:.4f} {sig}, d={cohens_d:.3f}, +{improvement:.1f}%")

    # ── Hypothesis check ───────────────────────────────────────────────────────
    avg_a = all_results["A_Random"]["avg_relevance"]
    hypothesis_passed = all(
        all_results[s]["avg_relevance"] > avg_a
        for s in ["B_Cosine", "C_Semantic", "D_KG"]
    )
    print(f"\n  Hypothesis (B,C,D > A): {'✓ PASSED' if hypothesis_passed else '✗ FAILED'}")

    # ── Save ───────────────────────────────────────────────────────────────────
    output = {
        "scenario": "1_retrieval_quality_vs_context_size",
        "hypothesis": "retrieval_quality + small_context > random + large_context",
        "hypothesis_passed": hypothesis_passed,
        "systems": all_results,
        "statistical_tests": stats_output,
    }
    out_dir = RESULTS_DIR / "scenario1"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n  ✓ Saved → {out_file}\n")
    return output


if __name__ == "__main__":
    run()
