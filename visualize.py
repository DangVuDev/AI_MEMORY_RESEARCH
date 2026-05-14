"""
visualize.py — Vẽ biểu đồ từ kết quả thực nghiệm 5 scenarios.

Được gọi tự động bởi run_all.py sau khi tất cả scenarios hoàn thành.
Cũng có thể chạy độc lập: python visualize.py

Output: results/visualizations/
  - scenario1_retrieval.png
  - scenario2_chunking.png
  - scenario3_scoring.png
  - scenario4_rag_vs_kg.png
    - scenario5_synthesis_bottleneck.png
  - summary_overview.png
"""
from __future__ import annotations
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend (không cần GUI)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

RESULTS_DIR = Path("results")
VIZ_DIR     = RESULTS_DIR / "visualizations"

# ── Màu sắc nhất quán ──────────────────────────────────────────────────────────
COLORS = ["#2196F3", "#4CAF50", "#FF9800", "#F44336", "#9C27B0", "#00BCD4"]

SCENARIO_META = {
    "scenario1": {
        "title": "Scenario 1 — Retrieval Quality vs Context Size",
        "xlabel": "Retrieval System",
        "systems_key": "systems",
        "hypothesis": "Dense retrieval (B,C,D) > Random baseline (A)",
    },
    "scenario2": {
        "title": "Scenario 2 — Chunking Strategy Impact",
        "xlabel": "Chunking Strategy",
        "systems_key": "strategies",
        "hypothesis": "Sentence chunking tốt nhất với corpus này",
    },
    "scenario3": {
        "title": "Scenario 3 — Memory Scoring Strategies",
        "xlabel": "Scoring Strategy",
        "systems_key": "systems",
        "hypothesis": "Relevance-based scoring (C,D) > Recency/Importance (A,B)",
    },
    "scenario4": {
        "title": "Scenario 4 — RAG vs Knowledge Graph",
        "xlabel": "System",
        "systems_key": "systems",
        "hypothesis": "Dense RAG (A) ≈ KG-Hybrid (D) > BM25 (B) > Entity-only (C)",
    },
    "scenario5": {
        "title": "Scenario 5 — Multi-context Synthesis Bottleneck",
        "xlabel": "Required Facts (N)",
        "systems_key": "systems",
        "hypothesis": "Synthesis accuracy drops as the number of linked facts increases",
    },
}


def load_scenario(sid: str) -> dict | None:
    path = RESULTS_DIR / sid / "results.json"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def fig_style(fig, ax, title: str) -> None:
    ax.set_title(title, fontsize=13, fontweight="bold", pad=14)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    fig.tight_layout()


# ── Biểu đồ 1 scenario: bar (avg_relevance) + line (latency) ──────────────────

def plot_scenario(sid: str, data: dict, out_stem: str) -> Path:
    meta    = SCENARIO_META[sid]
    systems = data.get(meta["systems_key"]) or {}

    # Sắp xếp theo avg_relevance giảm dần
    items = sorted(systems.items(), key=lambda kv: kv[1].get("avg_relevance", 0), reverse=True)
    names = [k for k, _ in items]
    rels  = [v.get("avg_relevance", 0) for _, v in items]
    lats  = [v.get("avg_latency_ms", 0) for _, v in items]

    fig, ax1 = plt.subplots(figsize=(10, 5))
    x = np.arange(len(names))
    w = 0.55

    bars = ax1.bar(x, rels, width=w, color=[COLORS[i % len(COLORS)] for i in range(len(names))],
                   edgecolor="white", linewidth=1.2, zorder=3)

    # Nhãn giá trị trên mỗi bar
    for bar_, rel in zip(bars, rels):
        ax1.text(bar_.get_x() + bar_.get_width() / 2, bar_.get_height() + 0.003,
                 f"{rel:.4f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax1.set_ylim(0, max(rels) * 1.18)
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, fontsize=10)
    ylabel = "avg_relevance (cosine sim)"
    if sid == "scenario5":
        ylabel = "avg_synthesis_accuracy"
    ax1.set_ylabel(ylabel, fontsize=10)
    ax1.set_xlabel(meta["xlabel"], fontsize=10)

    # Đường latency trục phụ
    ax2 = ax1.twinx()
    ax2.plot(x, lats, color="#795548", marker="o", linewidth=2, markersize=6,
             linestyle="--", label="Latency (ms)", zorder=4)
    ax2.set_ylabel("avg latency (ms)", fontsize=10, color="#795548")
    ax2.tick_params(axis="y", labelcolor="#795548")

    # Hypothesis text
    ax1.annotate(f"Hypothesis: {meta['hypothesis']}",
                 xy=(0.01, 0.02), xycoords="axes fraction",
                 fontsize=8, color="#555", style="italic")

    lat_patch = mpatches.Patch(color="#795548", label="Latency (ms)")
    ax1.legend(handles=lat_patch if isinstance(lat_patch, list) else [lat_patch],
               loc="upper right", fontsize=9)

    fig_style(fig, ax1, meta["title"])

    out_path = VIZ_DIR / f"{out_stem}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ── Boxplot: phân phối relevance scores ───────────────────────────────────────

def plot_boxplot(sid: str, data: dict, out_stem: str) -> Path:
    meta    = SCENARIO_META[sid]
    systems = data.get(meta["systems_key"]) or {}
    items   = sorted(systems.items(), key=lambda kv: kv[1].get("avg_relevance", 0), reverse=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    box_data = [v.get("relevances", []) for _, v in items]
    names    = [k for k, _ in items]

    bp = ax.boxplot(box_data, patch_artist=True, notch=False,
                    medianprops={"color": "white", "linewidth": 2},
                    whiskerprops={"linewidth": 1.5},
                    capprops={"linewidth": 1.5},
                    flierprops={"marker": "o", "markersize": 4, "alpha": 0.4})

    for patch, color in zip(bp["boxes"], COLORS):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)

    ax.set_xticklabels(names, fontsize=10)
    ax.set_ylabel("Relevance Score", fontsize=10)
    ax.set_xlabel(meta["xlabel"], fontsize=10)
    fig_style(fig, ax, meta["title"] + " — Score Distribution")

    out_path = VIZ_DIR / f"{out_stem}_dist.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ── Summary overview: 5 scenarios × best system ───────────────────────────────

def plot_summary(all_data: dict[str, dict | None]) -> Path:
    scenario_labels, best_names, best_scores, hyp_colors = [], [], [], []

    for sid in ["scenario1", "scenario2", "scenario3", "scenario4", "scenario5"]:
        data = all_data.get(sid)
        if data is None:
            continue
        num     = sid.replace("scenario", "")
        meta    = SCENARIO_META[sid]
        systems = data.get(meta["systems_key"]) or {}
        if not systems:
            continue
        best_name, best_val = max(systems.items(), key=lambda kv: kv[1].get("avg_relevance", 0))
        scenario_labels.append(f"S{num}\n{best_name}")
        best_names.append(best_name)
        best_scores.append(best_val.get("avg_relevance", 0))

        # Màu xanh nếu hypothesis passed, đỏ nếu không
        passed = data.get("hypothesis_passed")
        if passed is None:
            passed = bool(data.get("best_strategy") or data.get("best_system"))
        hyp_colors.append("#4CAF50" if passed else "#F44336")

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(scenario_labels))
    bars = ax.bar(x, best_scores, color=hyp_colors, edgecolor="white",
                  linewidth=1.5, width=0.5, zorder=3)

    for bar_, score, name in zip(bars, best_scores, best_names):
        ax.text(bar_.get_x() + bar_.get_width() / 2, bar_.get_height() + 0.002,
                f"{score:.4f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(scenario_labels, fontsize=10)
    ax.set_ylabel("Best avg_relevance per Scenario", fontsize=10)
    ax.set_ylim(0, max(best_scores) * 1.15)

    passed_patch  = mpatches.Patch(color="#4CAF50", label="Hypothesis PASSED")
    failed_patch  = mpatches.Patch(color="#F44336", label="Hypothesis FAILED")
    ax.legend(handles=[passed_patch, failed_patch], fontsize=9, loc="lower right")

    fig_style(fig, ax, "AI Memory Research — Best System per Scenario")

    out_path = VIZ_DIR / "summary_overview.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ── Radar chart: so sánh tổng thể các metrics ─────────────────────────────────

def plot_radar_s1(data: dict) -> Path:
    """Radar chart cho Scenario 1: 4 systems × multiple question types."""
    systems = data.get("systems") or {}
    if not systems:
        return None

    items      = list(systems.items())
    sys_names  = [k for k, _ in items]
    qtypes     = ["simple", "multi_hop_2", "multi_hop_3", "aggregation"]
    qtype_labels = ["Simple", "Multi-hop 2", "Multi-hop 3", "Aggregation"]

    # Tính avg_relevance theo từng question type nếu có; nếu không dùng overall
    all_questions_file = Path("data/questions.json")
    if not all_questions_file.exists():
        return None

    with open(all_questions_file, encoding="utf-8") as f:
        qs = json.load(f)["questions"]
    qtype_indices = {qt: [i for i, q in enumerate(qs) if q["type"] == qt]
                     for qt in qtypes}

    angles     = np.linspace(0, 2 * np.pi, len(qtypes), endpoint=False).tolist()
    angles    += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={"polar": True})

    for (name, stats), color in zip(items, COLORS):
        rels = stats.get("relevances", [])
        if not rels:
            continue
        vals = []
        for qt in qtypes:
            idxs = qtype_indices.get(qt, [])
            if idxs:
                vals.append(np.mean([rels[i] for i in idxs if i < len(rels)]))
            else:
                vals.append(stats.get("avg_relevance", 0))
        vals += vals[:1]
        ax.plot(angles, vals, "o-", linewidth=2, color=color, label=name)
        ax.fill(angles, vals, alpha=0.12, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(qtype_labels, fontsize=10)
    ax.set_ylim(0.5, 1.0)
    ax.set_title("Scenario 1 — Relevance by Question Type", fontsize=12,
                 fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=9)

    fig.tight_layout()
    out_path = VIZ_DIR / "scenario1_radar.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ── Entry point ────────────────────────────────────────────────────────────────

def run() -> list[str]:
    """Vẽ tất cả biểu đồ, trả về danh sách file đã tạo."""
    VIZ_DIR.mkdir(parents=True, exist_ok=True)

    all_data   = {sid: load_scenario(sid) for sid in
                  ["scenario1", "scenario2", "scenario3", "scenario4", "scenario5"]}
    generated  = []

    scenario_file_map = [
        ("scenario1", "scenario1_retrieval"),
        ("scenario2", "scenario2_chunking"),
        ("scenario3", "scenario3_scoring"),
        ("scenario4", "scenario4_rag_vs_kg"),
        ("scenario5", "scenario5_synthesis_bottleneck"),
    ]

    for sid, stem in scenario_file_map:
        data = all_data.get(sid)
        if data is None:
            print(f"  ⚠  {sid}: chưa có kết quả, bỏ qua")
            continue

        p1 = plot_scenario(sid, data, stem)
        p2 = plot_boxplot(sid, data, stem)
        generated.extend([str(p1), str(p2)])
        print(f"  ✓ {sid}: bar chart + boxplot → {p1.name}, {p2.name}")

    # Radar cho scenario 1
    if all_data.get("scenario1"):
        p = plot_radar_s1(all_data["scenario1"])
        if p:
            generated.append(str(p))
            print(f"  ✓ scenario1 radar → {p.name}")

    # Summary overview
    p_sum = plot_summary(all_data)
    generated.append(str(p_sum))
    print(f"  ✓ summary overview → {p_sum.name}")

    return generated


if __name__ == "__main__":
    print("\n  Generating visualizations...")
    files = run()
    print(f"\n  {len(files)} charts saved to results/visualizations/")
