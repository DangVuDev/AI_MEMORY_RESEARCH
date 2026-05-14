"""
Scenario 5: Multi-context Synthesis Bottleneck
==============================================
Test hypothesis from DOC.md:
"Answer quality drops sharply as the number of facts to synthesize increases."

Independent variable: number of required facts N in [1, 2, 4, 8, 16]
Controls:
- High-noise context (95% irrelevant segments)
- Fixed prompt template
- Multiple runs per level (30)
"""
from __future__ import annotations

import json
import os
import random
import re
import time
from dataclasses import dataclass

import numpy as np

from config import RESULTS_DIR
from openai_client import evaluate_relevance, generate_answer
from retriever import load_corpus
from utils.metrics import evaluate_common_metrics

N_LEVELS = [1, 2, 4, 8, 16]
RUNS_PER_LEVEL = 30
NOISE_RATIO = 0.95
# Keep context length fixed across N to isolate synthesis complexity effect.
# You can set SCENARIO5_CONTEXT_TOKENS=100000 for a strict long-context run.
CONTEXT_TOKEN_BUDGET = int(os.getenv("SCENARIO5_CONTEXT_TOKENS", "12000"))
PROMPT_TEMPLATE = "Analyze all relevant facts and answer the question."

FACT_GRAPH = [
    "Alice manages Team Alpha.",
    "Team Alpha owns Service Billing.",
    "Billing depends on DB-01.",
    "DB-01 had outage at 10:00.",
    "Incident severity was SEV-1.",
    "SEV-1 triggered incident commander Bob.",
    "Bob coordinated rollback to previous release.",
    "Rollback restored API latency in 15 minutes.",
    "Service Billing sends events to Queue Q7.",
    "Queue Q7 backlog caused delayed invoices.",
    "Delayed invoices affected premium customers first.",
    "Premium customers are managed by Success Team Gold.",
    "Root cause was schema mismatch in release R42.",
    "R42 was deployed by deployment bot Delta.",
    "Remediation added schema validation gate.",
    "Postmortem action item owner is Alice.",
]

QUESTION_BY_N = {
    1: "Who manages Team Alpha?",
    2: "Which service does Alice indirectly own?",
    4: "What infrastructure issue affected Alice's service?",
    8: "Describe the full outage chain from team ownership to recovery.",
    16: "Summarize root cause, impacted users, and remediation from all linked facts.",
}

TOKEN_RE = re.compile(r"\w+", re.UNICODE)
ENTITY_RE = re.compile(r"\b[A-Z][A-Za-z0-9_-]*\b")
WORD_RE = re.compile(r"[A-Za-z0-9_-]+")


@dataclass
class RunMetrics:
    relevance: float
    fact_recall: float
    synthesis_accuracy: float
    missing_fact_count: int
    wrong_link_count: int
    hallucination_rate: float
    unsupported_claim_count: int
    total_claim_count: int
    ttft_ms: float
    latency_ms: float
    context_precision: float
    context_recall: float
    faithfulness: float


def _token_set(text: str) -> set[str]:
    return {t.lower() for t in TOKEN_RE.findall(text) if len(t) > 2}


def _fact_is_covered(fact: str, answer: str, min_overlap: float = 0.35) -> bool:
    f = _token_set(fact)
    if not f:
        return False
    a = _token_set(answer)
    overlap = len(f & a) / len(f)
    return overlap >= min_overlap


def _extract_entities(text: str) -> set[str]:
    return set(ENTITY_RE.findall(text))


def _extract_entity_phrases(text: str) -> set[str]:
    raw = text.lower()
    patterns = [
        r"db-\d+",
        r"sev-\d+",
        r"r\d+",
        r"q\d+",
        r"team\s+[a-z0-9_-]+",
        r"service\s+[a-z0-9_-]+",
        r"success\s+team\s+[a-z0-9_-]+",
        r"premium\s+customers?",
        r"premium\s+users?",
        r"incident\s+commander\s+[a-z0-9_-]+",
        r"deployment\s+bot\s+[a-z0-9_-]+",
        r"schema\s+mismatch",
        r"rollback",
        r"alice",
        r"bob",
        r"delta",
    ]
    out = set()
    for p in patterns:
        out.update(m.group(0) for m in re.finditer(p, raw))
    return out


def _estimate_claim_count(answer: str) -> int:
    # Rough claim segmentation for unsupported_claims / total_claims.
    clauses = [c.strip() for c in re.split(r"[\.;]|\band\b|\bva\b", answer, flags=re.IGNORECASE) if c.strip()]
    return max(1, len(clauses))


def _hallucination_from_links(answer: str, wrong_links: int, missing_facts: int, required_facts: int) -> tuple[float, int, int]:
    total_claims = _estimate_claim_count(answer)
    # Missing required facts also imply unsupported/incomplete synthesis claims.
    unsupported = max(0, wrong_links) + max(0, missing_facts)
    total_claims = max(total_claims, required_facts)
    unsupported = min(total_claims, unsupported)
    rate = unsupported / max(1, total_claims)
    return float(rate), int(unsupported), int(total_claims)


def _question_expected_keywords(n: int) -> list[str]:
    if n == 1:
        return ["alice"]
    if n == 2:
        return ["service", "billing"]
    if n == 4:
        return ["db-01", "outage"]
    if n == 8:
        return ["team alpha", "service billing", "db-01", "sev-1", "bob", "rollback"]
    return ["schema mismatch", "premium", "remediation"]


def _binary_synthesis_accuracy(answer: str, n: int, fact_recall: float, wrong_links: int) -> float:
    ans = answer.lower()
    keywords = _question_expected_keywords(n)
    keyword_hits = sum(1 for k in keywords if k in ans)
    keyword_ratio = keyword_hits / max(1, len(keywords))

    # Unified quality score: stable across all N and less brittle than piecewise rules.
    wrong_link_penalty = min(1.0, wrong_links / max(1, n))
    quality = 0.7 * fact_recall + 0.3 * keyword_ratio - 0.1 * wrong_link_penalty
    threshold = 0.66 + 0.03 * np.log2(max(1, n))
    return 1.0 if quality >= threshold else 0.0


def _word_count(text: str) -> int:
    return len(WORD_RE.findall(text))


def _pad_context_to_budget(base_segments: list[str], noise_pool: list[str], rng: random.Random, token_budget: int) -> list[str]:
    segments = list(base_segments)
    current_tokens = sum(_word_count(s) for s in segments)
    safety = 0
    while current_tokens < token_budget and safety < token_budget * 2:
        noise = rng.choice(noise_pool)
        segments.append(noise)
        current_tokens += _word_count(noise)
        safety += 1
    rng.shuffle(segments)
    return segments


def _wrong_link_count(answer: str, required_facts: list[str]) -> int:
    answer_entities = _extract_entity_phrases(answer)
    allowed_entities = set()
    for fact in required_facts:
        allowed_entities.update(_extract_entity_phrases(fact))
    # Entities introduced in answer but not present in supporting fact chain.
    return max(0, len(answer_entities - allowed_entities))


def build_context(required_facts: list[str], noise_pool: list[str], rng: random.Random) -> tuple[str, list[str]]:
    # Keep 95% noise by segment count: relevant/(relevant+noise)=0.05.
    n_rel = len(required_facts)
    n_noise = int(round((NOISE_RATIO / (1 - NOISE_RATIO)) * n_rel))
    sampled_noise = [rng.choice(noise_pool) for _ in range(max(1, n_noise))]

    # Control context length to reduce context-window side effects.
    base = required_facts + sampled_noise
    segments = _pad_context_to_budget(base, noise_pool, rng, CONTEXT_TOKEN_BUDGET)
    return "\n---\n".join(segments), segments


def run_level(n: int, corpus_noise: list[str], rng: random.Random) -> tuple[dict, list[RunMetrics]]:
    question = QUESTION_BY_N[n]
    required_facts = FACT_GRAPH[:n]
    run_metrics: list[RunMetrics] = []

    for _ in range(RUNS_PER_LEVEL):
        context, segments = build_context(required_facts, corpus_noise, rng)

        full_prompt = f"{PROMPT_TEMPLATE}\n\nQuestion: {question}"

        t0 = time.time()
        answer = generate_answer(full_prompt, context)
        relevance = evaluate_relevance(question, context, answer)
        latency_ms = (time.time() - t0) * 1000
        # Non-streaming API: use total latency as TTFT estimate.
        ttft_ms = latency_ms

        covered = sum(1 for fact in required_facts if _fact_is_covered(fact, answer))
        fact_recall = covered / len(required_facts)
        missing = len(required_facts) - covered
        wrong_links = _wrong_link_count(answer, required_facts)
        synth_acc = _binary_synthesis_accuracy(answer, n, fact_recall, wrong_links)
        halluc_rate, unsupported_claims, total_claims = _hallucination_from_links(
            answer, wrong_links, missing, len(required_facts)
        )

        cm = evaluate_common_metrics(
            question=question,
            retrieved_docs=required_facts,
            all_candidates=segments,
            answer=answer,
            context=context,
        )

        run_metrics.append(
            RunMetrics(
                relevance=float(relevance),
                fact_recall=float(fact_recall),
                synthesis_accuracy=float(synth_acc),
                missing_fact_count=int(missing),
                wrong_link_count=int(wrong_links),
                hallucination_rate=float(halluc_rate),
                unsupported_claim_count=int(unsupported_claims),
                total_claim_count=int(total_claims),
                ttft_ms=float(ttft_ms),
                latency_ms=float(latency_ms),
                context_precision=float(cm["context_precision"]),
                context_recall=float(cm["context_recall"]),
                faithfulness=float(cm["faithfulness"]),
            )
        )

    agg = {
        "n_facts": n,
        "runs": RUNS_PER_LEVEL,
        "avg_relevance": float(np.mean([m.relevance for m in run_metrics])),
        "avg_fact_recall": float(np.mean([m.fact_recall for m in run_metrics])),
        "avg_synthesis_accuracy": float(np.mean([m.synthesis_accuracy for m in run_metrics])),
        "avg_missing_fact_count": float(np.mean([m.missing_fact_count for m in run_metrics])),
        "avg_wrong_link_count": float(np.mean([m.wrong_link_count for m in run_metrics])),
        "avg_hallucination_rate": float(np.mean([m.hallucination_rate for m in run_metrics])),
        "avg_unsupported_claim_count": float(np.mean([m.unsupported_claim_count for m in run_metrics])),
        "avg_total_claim_count": float(np.mean([m.total_claim_count for m in run_metrics])),
        "avg_ttft_ms": float(np.mean([m.ttft_ms for m in run_metrics])),
        "avg_latency_ms": float(np.mean([m.latency_ms for m in run_metrics])),
        "avg_context_precision": float(np.mean([m.context_precision for m in run_metrics])),
        "avg_context_recall": float(np.mean([m.context_recall for m in run_metrics])),
        "avg_faithfulness": float(np.mean([m.faithfulness for m in run_metrics])),
        "relevances": [float(m.synthesis_accuracy) for m in run_metrics],
        "runs_detail": [
            {
                "fact_recall": float(m.fact_recall),
                "synthesis_accuracy": float(m.synthesis_accuracy),
                "missing_fact_count": int(m.missing_fact_count),
                "wrong_link_count": int(m.wrong_link_count),
                "unsupported_claim_count": int(m.unsupported_claim_count),
                "total_claim_count": int(m.total_claim_count),
                "hallucination_rate": float(m.hallucination_rate),
                "ttft_ms": float(m.ttft_ms),
                "latency_ms": float(m.latency_ms),
            }
            for m in run_metrics
        ],
    }
    return agg, run_metrics


def run() -> dict:
    print("\n" + "=" * 70)
    print("SCENARIO 5 — Multi-context Synthesis Bottleneck")
    print("=" * 70)

    rng = random.Random(42)
    corpus_noise = load_corpus()

    levels = {}
    systems = {}

    for n in N_LEVELS:
        print(f"\n[Level N={n}] running {RUNS_PER_LEVEL} iterations...")
        agg, _ = run_level(n, corpus_noise, rng)
        levels[f"N_{n}"] = agg

        # For report/visualize compatibility: use synthesis accuracy as primary score.
        systems[f"N_{n}"] = {
            "avg_relevance": agg["avg_synthesis_accuracy"],
            "avg_latency_ms": agg["avg_latency_ms"],
            "avg_ttft_ms": agg["avg_ttft_ms"],
            "avg_context_precision": agg["avg_context_precision"],
            "avg_context_recall": agg["avg_context_recall"],
            "avg_faithfulness": agg["avg_faithfulness"],
            "avg_hallucination_rate": agg["avg_hallucination_rate"],
            "relevances": agg["relevances"],
        }

        print(
            f"  avg_acc={agg['avg_synthesis_accuracy']:.3f} | "
            f"fact_recall={agg['avg_fact_recall']:.3f} | "
            f"halluc={agg['avg_hallucination_rate']:.3f} | "
            f"missing={agg['avg_missing_fact_count']:.2f}"
        )

    acc = {n: levels[f"N_{n}"]["avg_synthesis_accuracy"] for n in N_LEVELS}
    halluc = {n: levels[f"N_{n}"]["avg_hallucination_rate"] for n in N_LEVELS}
    missing = {n: levels[f"N_{n}"]["avg_missing_fact_count"] for n in N_LEVELS}
    drop_1_2 = acc[1] - acc[2]
    drop_8_16 = acc[8] - acc[16]
    pair_drops = {
        "1_to_2": float(acc[1] - acc[2]),
        "2_to_4": float(acc[2] - acc[4]),
        "4_to_8": float(acc[4] - acc[8]),
        "8_to_16": float(acc[8] - acc[16]),
    }
    breaking_point = max(pair_drops.items(), key=lambda kv: kv[1])[0]
    eps = 0.02
    monotonic_drop = all(
        acc[N_LEVELS[i]] + eps >= acc[N_LEVELS[i + 1]]
        for i in range(len(N_LEVELS) - 1)
    )
    hypothesis_passed = bool(monotonic_drop and drop_8_16 > drop_1_2)

    output = {
        "scenario": "5_multi_context_synthesis_bottleneck",
        "hypothesis": "quality drops superlinearly as number of synthesized facts increases",
        "hypothesis_passed": hypothesis_passed,
        "controls": {
            "noise_ratio": NOISE_RATIO,
            "runs_per_level": RUNS_PER_LEVEL,
            "n_levels": N_LEVELS,
            "context_token_budget": CONTEXT_TOKEN_BUDGET,
            "prompt_template": PROMPT_TEMPLATE,
            "ttft_note": "Estimated from non-streaming call (ttft_ms ~= latency_ms)",
        },
        "systems": systems,
        "levels": levels,
        "curves": {
            "accuracy_by_n": {str(k): float(v) for k, v in acc.items()},
            "hallucination_by_n": {str(k): float(v) for k, v in halluc.items()},
            "missing_fact_by_n": {str(k): float(v) for k, v in missing.items()},
        },
        "drop_analysis": {
            "drop_1_to_2": float(drop_1_2),
            "drop_8_to_16": float(drop_8_16),
            "pair_drops": pair_drops,
            "superlinear_drop": bool(drop_8_16 > drop_1_2),
            "monotonic_drop": bool(monotonic_drop),
            "breaking_point": breaking_point,
        },
    }

    out_dir = RESULTS_DIR / "scenario5"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n  Hypothesis: {'✓ PASSED' if hypothesis_passed else '✗ FAILED'}")
    print(f"  ✓ Saved → {out_file}\n")
    return output


if __name__ == "__main__":
    run()
