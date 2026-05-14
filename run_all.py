"""
run_all.py — Pipeline đầy đủ: Scenarios → Report → Visualizations
-------------------------------------------------------------------
Dùng:
    python run_all.py              ← chạy tất cả 5 scenarios + report + charts
    python run_all.py --scenario 2 ← chỉ chạy 1 scenario (không gọi report/viz)
    python run_all.py --report     ← chỉ tạo report + charts từ kết quả đã có
"""
import sys
import json
import time
import argparse
from datetime import datetime
from pathlib import Path


def banner(text: str) -> None:
    print("\n" + "█" * 70)
    print(f"  {text}")
    print("█" * 70)


def section(text: str) -> None:
    print("\n" + "─" * 70)
    print(f"  {text}")
    print("─" * 70)


def run_scenario(num: int) -> dict | None:
    """Chạy 1 scenario, trả về kết quả hoặc None nếu lỗi."""
    sys.path.insert(0, str(Path(__file__).parent))

    try:
        if num == 1:
            from scenarios.scenario_1_rag import run
        elif num == 2:
            from scenarios.scenario_2_chunking import run
        elif num == 3:
            from scenarios.scenario_3_scoring import run
        elif num == 4:
            from scenarios.scenario_4_rag_vs_kg import run
        elif num == 5:
            from scenarios.scenario_5_synthesis_bottleneck import run
        else:
            print(f"  ✗ Scenario {num} không tồn tại")
            return None
        return run()
    except Exception as e:
        print(f"\n  ✗ Scenario {num} lỗi: {e}")
        import traceback
        traceback.print_exc()
        return None


def run_report() -> bool:
    """Gọi report.py để tổng hợp kết quả → full_report.json."""
    try:
        import importlib.util, sys as _sys
        spec = importlib.util.spec_from_file_location(
            "report", Path(__file__).parent / "report.py"
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        mod.main()
        return True
    except Exception as e:
        print(f"  ✗ Report lỗi: {e}")
        import traceback; traceback.print_exc()
        return False


def run_visualize() -> bool:
    """Gọi visualize.py để vẽ biểu đồ từ JSON kết quả."""
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "visualize", Path(__file__).parent / "visualize.py"
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        files = mod.run()
        print(f"  ✓ {len(files)} biểu đồ đã lưu → results/visualizations/")
        return True
    except Exception as e:
        print(f"  ✗ Visualize lỗi: {e}")
        import traceback; traceback.print_exc()
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="AI Memory Research – Full Pipeline")
    parser.add_argument("--scenario", type=int, choices=[1, 2, 3, 4, 5],
                        help="Chạy 1 scenario cụ thể (không kèm report/viz).")
    parser.add_argument("--report", action="store_true",
                        help="Chỉ tạo report + visualizations từ kết quả có sẵn.")
    args = parser.parse_args()

    banner("AI MEMORY RESEARCH — Full Pipeline")
    print(f"  Started : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # ── Chế độ chỉ report/viz ──────────────────────────────────────────────────
    if args.report:
        section("Generating Report + Visualizations")
        run_report()
        run_visualize()
        return

    # ── Chế độ chạy 1 scenario cụ thể ─────────────────────────────────────────
    if args.scenario:
        banner(f"SCENARIO {args.scenario}")
        t0 = time.time()
        result = run_scenario(args.scenario)
        elapsed = time.time() - t0
        if result:
            print(f"  ✓ Scenario {args.scenario} xong trong {elapsed/60:.1f} phút")
        Path("results").mkdir(exist_ok=True)
        out = {
            "run_at": datetime.now().isoformat(),
            "scenarios_run": [args.scenario],
            "total_elapsed_sec": round(elapsed, 1),
            "results": {f"scenario_{args.scenario}": {
                "status": "completed" if result else "failed",
                "elapsed_sec": round(elapsed, 1),
            }},
        }
        with open("results/run_summary.json", "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print("  ✓ run_summary.json saved\n")
        return

    # ── Chế độ chạy tất cả (mặc định) ─────────────────────────────────────────
    to_run      = [1, 2, 3, 4, 5]
    total_start = time.time()
    run_results = {}

    for num in to_run:
        banner(f"SCENARIO {num} / {len(to_run)}")
        t0     = time.time()
        result = run_scenario(num)
        elapsed = time.time() - t0

        if result is not None:
            run_results[f"scenario_{num}"] = {
                "status": "completed",
                "elapsed_sec": round(elapsed, 1),
            }
            print(f"\n  ✓ Scenario {num} xong trong {elapsed/60:.1f} phút")
        else:
            run_results[f"scenario_{num}"] = {"status": "failed"}
            print(f"\n  ✗ Scenario {num} thất bại — tiếp tục sang scenario tiếp theo")

    total_elapsed = time.time() - total_start

    # ── Bước 2: Tổng hợp báo cáo ──────────────────────────────────────────────
    banner("STEP 2 / 3 — Generating Consolidated Report")
    report_ok = run_report()

    # ── Bước 3: Vẽ biểu đồ ────────────────────────────────────────────────────
    banner("STEP 3 / 3 — Generating Visualizations")
    viz_ok = run_visualize()

    # ── Tổng kết ───────────────────────────────────────────────────────────────
    banner("PIPELINE COMPLETE")
    for name, info in run_results.items():
        icon  = "✓" if info["status"] == "completed" else "✗"
        mins  = f"  ({info.get('elapsed_sec', 0)/60:.1f} min)" if "elapsed_sec" in info else ""
        print(f"  {icon} {name}{mins}")

    print(f"\n  {'✓' if report_ok else '✗'} full_report.json")
    print(f"  {'✓' if viz_ok else '✗'} visualizations (results/visualizations/)")
    print(f"\n  Tổng thời gian : {total_elapsed/60:.1f} phút")
    print(f"  Kết quả        : results/")

    # Lưu run_summary.json
    Path("results").mkdir(exist_ok=True)
    out = {
        "run_at": datetime.now().isoformat(),
        "scenarios_run": to_run,
        "total_elapsed_sec": round(total_elapsed, 1),
        "report_generated": report_ok,
        "visualizations_generated": viz_ok,
        "results": run_results,
    }
    with open("results/run_summary.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print("  ✓ run_summary.json saved\n")


if __name__ == "__main__":
    main()
