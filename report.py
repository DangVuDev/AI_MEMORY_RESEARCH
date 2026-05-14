"""
report.py — Đọc kết quả từ tất cả 4 scenario và xuất báo cáo tổng hợp.

Usage:
    python report.py
"""
from __future__ import annotations
import json
from pathlib import Path
from datetime import datetime

RESULTS_DIR = Path("results")

SCENARIO_META = {
    "scenario1": {
        "title": "Retrieval Quality vs Context Size",
        "hypothesis": "Dense retrieval (B,C,D) > Random baseline (A)",
    },
    "scenario2": {
        "title": "Chunking Strategy Impact",
        "hypothesis": "Semantic chunking (C) > rule-based (A,B)",
    },
    "scenario3": {
        "title": "Memory Scoring Strategies",
        "hypothesis": "Relevance-based scoring (C,D) > recency/importance (A,B)",
    },
    "scenario4": {
        "title": "RAG vs Knowledge Graph",
        "hypothesis": "Dense RAG (A) ≈ KG-Hybrid (D) > BM25 (B) > Entity-only (C)",
    },
    "scenario5": {
        "title": "Multi-context Synthesis Bottleneck",
        "hypothesis": "Quality drops superlinearly as required fact count increases (N=1→16)",
    },
}


def load_results() -> dict[str, dict]:
    loaded = {}
    for sid in ["scenario1", "scenario2", "scenario3", "scenario4", "scenario5"]:
        path = RESULTS_DIR / sid / "results.json"
        if path.exists():
            with open(path, encoding="utf-8") as f:
                loaded[sid] = json.load(f)
        else:
            loaded[sid] = None
    return loaded


def bar(value: float, width: int = 30) -> str:
    filled = int(round(value * width))
    return "█" * filled + "░" * (width - filled)


def print_scenario(sid: str, data: dict | None) -> dict:
    meta = SCENARIO_META[sid]
    num  = sid.replace("scenario", "")
    sep  = "═" * 70

    print(f"\n{sep}")
    print(f"  SCENARIO {num} — {meta['title']}")
    print(f"{sep}")
    print(f"  Hypothesis : {meta['hypothesis']}")
    if data is None:
        print("  ⚠  Chưa có kết quả — chạy: python run_all.py --scenario " + num)
        return {"scenario": sid, "status": "missing"}

    passed = data.get("hypothesis_passed")
    # Scenario 2/3 dùng best_strategy thay vì hypothesis_passed
    if passed is None:
        best_sys = data.get("best_strategy") or data.get("best_system", "")
        passed = bool(best_sys)
    print(f"  Result     : {'✓ PASSED' if passed else '✗ FAILED'}")
    print()

    systems = data.get("systems") or data.get("strategies") or {}
    if not systems:
        print("  (Không có dữ liệu systems)")
        return {"scenario": sid, "status": "no_data"}

    # Sort by avg_relevance desc
    ranked = sorted(
        systems.items(),
        key=lambda kv: kv[1].get("avg_relevance", 0),
        reverse=True,
    )

    best_score = ranked[0][1].get("avg_relevance", 0)

    print(f"  {'System':<18} {'avg_relevance':>13}  {'Latency':>10}  Chart")
    print(f"  {'─'*18} {'─'*13}  {'─'*10}  {'─'*30}")
    for rank, (name, stats) in enumerate(ranked):
        rel  = stats.get("avg_relevance", 0)
        lat  = stats.get("avg_latency_ms", 0)
        flag = " ◀ best" if rank == 0 else ""
        print(f"  {name:<18} {rel:>13.4f}  {lat:>9.1f}ms  {bar(rel)}{flag}")

    # Stats — support both key names used across scenarios
    stats_data = data.get("statistics") or data.get("statistical_tests") or {}
    if stats_data:
        print()
        print(f"  {'Comparison':<35} {'t':>7}  {'p':>9}  {'d':>7}  Δ%")
        print(f"  {'─'*35} {'─'*7}  {'─'*9}  {'─'*7}  {'─'*8}")
        for comp, s in stats_data.items():
            t = s.get("t_stat", s.get("t", 0))
            p = s.get("p_value", s.get("p", 1))
            d = s.get("cohens_d", s.get("d", 0))
            delta = s.get("delta_pct", s.get("improvement_pct", 0))
            sig = "***" if p < 0.001 else ("** " if p < 0.01 else ("*  " if p < 0.05 else "   "))
            print(f"  {comp:<35} {t:>7.3f}  {p:>9.4f}{sig}  {d:>7.3f}  {delta:>+.1f}%")

    if sid == "scenario5":
        levels = data.get("levels") or {}
        if levels:
            print()
            print(f"  {'N':<8} {'Accuracy':>10}  {'Halluc':>10}  {'Missing':>10}  {'TTFT(ms)':>10}  {'Latency(ms)':>12}")
            print(f"  {'─'*8} {'─'*10}  {'─'*10}  {'─'*10}  {'─'*10}  {'─'*12}")
            for n in [1, 2, 4, 8, 16]:
                lv = levels.get(f"N_{n}", {})
                print(
                    f"  {f'N={n}':<8} "
                    f"{lv.get('avg_synthesis_accuracy', 0):>10.4f}  "
                    f"{lv.get('avg_hallucination_rate', 0):>10.4f}  "
                    f"{lv.get('avg_missing_fact_count', 0):>10.2f}  "
                    f"{lv.get('avg_ttft_ms', 0):>10.1f}  "
                    f"{lv.get('avg_latency_ms', 0):>12.1f}"
                )

            drop = data.get("drop_analysis", {})
            bp = drop.get("breaking_point", "n/a")
            print(f"\n  Breaking point: {bp}")
            print(
                "  Superlinear check: "
                f"drop(8→16)={drop.get('drop_8_to_16', 0):.4f} vs "
                f"drop(1→2)={drop.get('drop_1_to_2', 0):.4f}"
            )

    return {
        "scenario": sid,
        "status": "completed",
        "hypothesis_passed": passed,
        "best_system": ranked[0][0],
        "best_avg_relevance": round(best_score, 4),
        "ranking": [
            {
                "system": name,
                "avg_relevance": round(stats.get("avg_relevance", 0), 4),
                "avg_latency_ms": round(stats.get("avg_latency_ms", 0), 1),
            }
            for name, stats in ranked
        ],
    }


def main() -> None:
    print("\n" + "█" * 70)
    print("  AI MEMORY RESEARCH — Consolidated Report")
    print("  Generated:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("█" * 70)

    all_results = load_results()
    report_records = []

    for sid in ["scenario1", "scenario2", "scenario3", "scenario4", "scenario5"]:
        rec = print_scenario(sid, all_results[sid])
        report_records.append(rec)

    # Overall summary table
    completed = [r for r in report_records if r["status"] == "completed"]
    missing   = [r for r in report_records if r["status"] == "missing"]

    print("\n" + "═" * 70)
    print("  OVERALL SUMMARY")
    print("═" * 70)
    total = len(SCENARIO_META)
    print(f"\n  Scenarios completed : {len(completed)}/{total}")
    if missing:
        print(f"  Missing             : {', '.join(r['scenario'] for r in missing)}")

    if completed:
        passed = sum(1 for r in completed if r.get("hypothesis_passed"))
        print(f"  Hypotheses passed   : {passed}/{len(completed)}")
        print()
        print(f"  {'Scenario':<14} {'Best system':<20} {'avg_relevance':>13}  Hypothesis")
        print(f"  {'─'*14} {'─'*20} {'─'*13}  {'─'*12}")
        for r in completed:
            num   = r["scenario"].replace("scenario", "")
            check = "✓ PASSED" if r.get("hypothesis_passed") else "✗ FAILED"
            print(
                f"  Scenario {num:<5} {r['best_system']:<20} "
                f"{r['best_avg_relevance']:>13.4f}  {check}"
            )

    # Save consolidated report
    report_path = RESULTS_DIR / "full_report.json"
    report_obj  = {
        "generated_at": datetime.now().isoformat(),
        "scenarios": report_records,
        "summary": {
            "total": total,
            "completed": len(completed),
            "hypotheses_passed": sum(1 for r in completed if r.get("hypothesis_passed")),
        },
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report_obj, f, ensure_ascii=False, indent=2)

    print(f"\n  ✓ Full report saved → {report_path}")
    print("═" * 70 + "\n")


if __name__ == "__main__":
    main()
