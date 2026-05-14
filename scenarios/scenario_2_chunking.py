"""
Scenario 2: Chunking Strategy Impact
=====================================
Kiểm chứng: chiến lược chia chunk ảnh hưởng thế nào đến chất lượng retrieval.

Vấn đề thiết kế: corpus docs mỗi cái chỉ ~1 câu → chunking per-doc không
tạo ra sự khác biệt. Fix: gom nhiều docs thành "passages" rồi áp 3 strategies
chunking khác nhau → số chunks và nội dung khác nhau thực sự.

3 Chunking Strategies:
  A_Sentence  – mỗi câu (split ". ") từ passages lớn → nhiều chunks nhỏ
  B_FixedSize – chunk cố định 25 từ, overlap 5 → chunks vừa
  C_Semantic  – gom câu liên quan thành nhóm (threshold 0.6) → chunks lớn
"""
import json
import time
from typing import List

import numpy as np
from scipy import stats
from sentence_transformers import util

from retriever import load_corpus, load_questions, get_embedder
from openai_client import generate_answer, evaluate_relevance
from config import RESULTS_DIR, TOP_K
from utils.metrics import evaluate_common_metrics


# ── Build passages: gom N docs liền nhau thành 1 passage ──────────────────────

def build_passages(corpus: List[str], group_size: int = 5) -> List[str]:
    """Gom từng nhóm group_size docs liền nhau thành 1 passage để chunking."""
    passages = []
    for i in range(0, len(corpus), group_size):
        block = " ".join(corpus[i: i + group_size])
        passages.append(block)
    return passages


# ── Chunking functions ────────────────────────────────────────────────────────

def chunk_by_sentence(passages: List[str]) -> List[str]:
    """Tách từng câu (split '. ') → nhiều chunk nhỏ."""
    chunks = []
    for passage in passages:
        for sent in passage.replace("!", ".").replace("?", ".").split(". "):
            s = sent.strip().rstrip(".")
            if len(s) > 15:
                chunks.append(s)
    return chunks


def chunk_fixed_size(passages: List[str], size: int = 25, overlap: int = 5) -> List[str]:
    """Chunk cố định size từ, overlap từ → chunks vừa."""
    chunks = []
    for passage in passages:
        words = passage.split()
        start = 0
        while start < len(words):
            chunk = " ".join(words[start: start + size])
            if len(chunk) > 15:
                chunks.append(chunk)
            start += size - overlap
    return chunks


def chunk_semantic(passages: List[str], threshold: float = 0.60) -> List[str]:
    """Gom các câu liên tiếp có cosine sim > threshold thành 1 chunk → chunks lớn hơn."""
    embedder = get_embedder()
    chunks = []
    for passage in passages:
        sentences = [s.strip() for s in passage.replace("!", ".").replace("?", ".").split(". ")
                     if len(s.strip()) > 10]
        if len(sentences) <= 1:
            if sentences:
                chunks.append(sentences[0])
            continue
        embs = embedder.encode(sentences, convert_to_tensor=True, show_progress_bar=False)
        current_group = [sentences[0]]
        for i in range(1, len(sentences)):
            sim = float(util.cos_sim(embs[i - 1], embs[i]))
            if sim >= threshold:
                current_group.append(sentences[i])
            else:
                chunks.append(". ".join(current_group))
                current_group = [sentences[i]]
        chunks.append(". ".join(current_group))
    return chunks


def encode_chunks(chunks: List[str]):
    embedder = get_embedder()
    return embedder.encode(chunks, convert_to_tensor=True, show_progress_bar=False)


def retrieve_from_chunks(chunks: List[str], query: str, k: int = TOP_K, chunk_embs=None) -> List[str]:
    embedder = get_embedder()
    q_emb = embedder.encode(query, convert_to_tensor=True)
    c_embs = chunk_embs if chunk_embs is not None else encode_chunks(chunks)
    scores = util.cos_sim(q_emb, c_embs)[0]
    top_idx = scores.topk(min(k, len(chunks))).indices.tolist()
    return [chunks[i] for i in top_idx]


# ── Main ──────────────────────────────────────────────────────────────────────

def run() -> dict:
    print("\n" + "=" * 70)
    print("SCENARIO 2 — Chunking Strategy Impact")
    print("=" * 70)

    corpus    = load_corpus()
    questions = load_questions()
    print(f"\n    corpus={len(corpus)} docs  |  questions={len(questions)}\n")

    # Gom docs thành passages để chunking có ý nghĩa
    print("[1] Building passages (group_size=5) then chunking...")
    passages = build_passages(corpus, group_size=5)
    print(f"    {len(corpus)} docs → {len(passages)} passages")

    strategies = {
        "A_Sentence":  chunk_by_sentence(passages),
        "B_FixedSize": chunk_fixed_size(passages, size=25, overlap=5),
        "C_Semantic":  chunk_semantic(passages, threshold=0.60),
    }
    for name, chunks in strategies.items():
        avg_len = int(np.mean([len(c.split()) for c in chunks]))
        print(f"    {name}: {len(chunks)} chunks  (avg {avg_len} words/chunk)")

    print("\n[2] Running all strategies...\n")
    all_results: dict[str, dict] = {}

    for strat_name, chunks in strategies.items():
        print(f"  ▶ {strat_name}")
        relevances, latencies = [], []
        precisions, recalls, faiths, hallucs = [], [], [], []

        print(f"    Pre-computing {len(chunks)} chunk embeddings...")
        chunk_embs = encode_chunks(chunks)

        for i, (question, _) in enumerate(questions, 1):
            t0 = time.time()
            docs    = retrieve_from_chunks(chunks, question, chunk_embs=chunk_embs)
            context = "\n---\n".join(docs)
            answer  = generate_answer(question, context)
            score   = evaluate_relevance(question, context, answer)
            m = evaluate_common_metrics(question, docs, chunks, answer, context)
            elapsed = (time.time() - t0) * 1000
            relevances.append(score)
            latencies.append(elapsed)
            precisions.append(m["context_precision"])
            recalls.append(m["context_recall"])
            faiths.append(m["faithfulness"])
            hallucs.append(m["hallucination_rate"])

            if i % 10 == 0 or i == len(questions):
                print(f"    {i:>2}/{len(questions)}  avg_rel={np.mean(relevances):.3f}")

        all_results[strat_name] = {
            "num_chunks": len(chunks),
            "avg_relevance": float(np.mean(relevances)),
            "avg_latency_ms": float(np.mean(latencies)),
            "avg_context_precision": float(np.mean(precisions)),
            "avg_context_recall": float(np.mean(recalls)),
            "avg_faithfulness": float(np.mean(faiths)),
            "avg_hallucination_rate": float(np.mean(hallucs)),
            "relevances": [float(r) for r in relevances],
        }
        print(f"    ✓ avg_relevance={all_results[strat_name]['avg_relevance']:.3f}\n")

    # ── Stats ──────────────────────────────────────────────────────────────────
    print("[3] Statistical Analysis")
    a_scores = np.array(all_results["A_Sentence"]["relevances"])
    stats_output = {}

    for name in ["B_FixedSize", "C_Semantic"]:
        b_scores = np.array(all_results[name]["relevances"])
        t_stat, p_val = stats.ttest_rel(a_scores, b_scores)
        pooled = np.sqrt((np.var(a_scores, ddof=1) + np.var(b_scores, ddof=1)) / 2)
        d = float((np.mean(b_scores) - np.mean(a_scores)) / pooled) if pooled > 0 else 0.0
        imp = (np.mean(b_scores) - np.mean(a_scores)) / max(np.mean(a_scores), 1e-9) * 100
        stats_output[f"{name}_vs_A"] = {
            "t_stat": float(t_stat), "p_value": float(p_val),
            "cohens_d": d, "improvement_pct": float(imp), "significant": bool(p_val < 0.05),
        }
        print(f"  {name} vs A_Sentence: t={t_stat:.3f}, p={p_val:.4f}, +{imp:.1f}%")

    best = max(all_results, key=lambda k: all_results[k]["avg_relevance"])
    print(f"\n  Best strategy: {best} (avg={all_results[best]['avg_relevance']:.3f})")

    output = {
        "scenario": "2_chunking_strategy_impact",
        "best_strategy": best,
        "strategies": all_results,
        "statistical_tests": stats_output,
    }
    out_dir = RESULTS_DIR / "scenario2"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n  ✓ Saved → {out_file}\n")
    return output


if __name__ == "__main__":
    run()
