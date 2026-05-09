#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Visualization utilities for research results.
Generates graphs from scenario result JSON files using matplotlib and seaborn.
"""

import json
import os
from pathlib import Path
import numpy as np

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
except ImportError:
    print("Warning: matplotlib or seaborn not installed. Install with: pip install matplotlib seaborn")
    plt = None
    sns = None

# Set style if available
if sns:
    sns.set_style("whitegrid")
    sns.set_palette("husl")

class ResultsVisualizer:
    """Visualize scenario results"""
    
    def __init__(self, results_dir="results"):
        self.results_dir = results_dir
        self.output_dir = os.path.join(results_dir, "visualizations")
        os.makedirs(self.output_dir, exist_ok=True)
    
    def load_scenario_result(self, scenario_name):
        """Load scenario result JSON"""
        path = os.path.join(self.results_dir, scenario_name, "results.json")
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        return None
    
    def load_master_report(self):
        """Load master report"""
        path = os.path.join(self.results_dir, "MASTER_REPORT.json")
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        return None
    
    def visualize_scenario_1(self):
        """Visualize Scenario 1: Basic Retrieval Quality vs Context Size"""
        if not plt:
            print("Matplotlib not available")
            return
        
        result = self.load_scenario_result("scenario1")
        if not result:
            print("Scenario 1 results not found")
            return
        
        systems = result.get("systems", {})
        
        # Extract data
        system_names = list(systems.keys())
        avg_relevances = [systems[s].get("avg_relevance", 0) for s in system_names]
        context_sizes = [int(systems[s].get("context_size", "0K").rstrip("K")) for s in system_names]
        
        # Create figure with subplots
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot 1: Relevance by System
        colors = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#FFA07A"]
        axes[0].bar(system_names, avg_relevances, color=colors, alpha=0.8, edgecolor="black")
        axes[0].set_ylabel("Average Relevance", fontsize=11, fontweight="bold")
        axes[0].set_title("Scenario 1: Retrieval Quality by System", fontsize=12, fontweight="bold")
        axes[0].set_ylim(0, max(avg_relevances) * 1.2)
        for i, v in enumerate(avg_relevances):
            axes[0].text(i, v + 0.005, f"{v:.3f}", ha="center", fontweight="bold")
        axes[0].grid(axis="y", alpha=0.3)
        
        # Plot 2: Context Size vs Relevance
        axes[1].scatter(context_sizes, avg_relevances, s=300, alpha=0.6, c=colors, edgecolor="black", linewidth=2)
        for i, name in enumerate(system_names):
            axes[1].annotate(name.replace("_", " "), (context_sizes[i], avg_relevances[i]), 
                           xytext=(5, 5), textcoords="offset points", fontweight="bold")
        axes[1].set_xlabel("Context Size (KB)", fontsize=11, fontweight="bold")
        axes[1].set_ylabel("Average Relevance", fontsize=11, fontweight="bold")
        axes[1].set_title("Context Size vs Retrieval Quality", fontsize=12, fontweight="bold")
        axes[1].grid(alpha=0.3)
        
        plt.tight_layout()
        output_path = os.path.join(self.output_dir, "scenario1_retrieval_quality.png")
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"✓ Saved: {output_path}")
        plt.close()
    
    def visualize_scenario_2(self):
        """Visualize Scenario 2: Chunking Strategy Impact"""
        if not plt:
            print("Matplotlib not available")
            return
        
        result = self.load_scenario_result("scenario2")
        if not result:
            print("Scenario 2 results not found")
            return
        
        strategies = result.get("strategies", {})
        
        # Extract data
        strategy_names = list(strategies.keys())
        quality_scores = [strategies[s].get("avg_retrieval_quality", 0) for s in strategy_names]
        num_chunks = [strategies[s].get("num_chunks", 0) for s in strategy_names]
        avg_sizes = [strategies[s].get("avg_chunk_size", 0) for s in strategy_names]
        
        # Create figure
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot 1: Retrieval Quality by Strategy
        colors = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#FFA07A"]
        bars = axes[0].barh(strategy_names, quality_scores, color=colors, alpha=0.8, edgecolor="black")
        axes[0].set_xlabel("Average Retrieval Quality", fontsize=11, fontweight="bold")
        axes[0].set_title("Scenario 2: Chunking Strategy Impact", fontsize=12, fontweight="bold")
        axes[0].set_xlim(0, 1.0)
        for i, v in enumerate(quality_scores):
            axes[0].text(v + 0.02, i, f"{v:.3f}", va="center", fontweight="bold")
        axes[0].grid(axis="x", alpha=0.3)
        
        # Plot 2: Chunks vs Avg Size
        axes[1].scatter(num_chunks, avg_sizes, s=300, alpha=0.6, c=colors, edgecolor="black", linewidth=2)
        for i, name in enumerate(strategy_names):
            axes[1].annotate(name.replace("_", " "), (num_chunks[i], avg_sizes[i]), 
                           xytext=(5, 5), textcoords="offset points", fontsize=9, fontweight="bold")
        axes[1].set_xlabel("Number of Chunks", fontsize=11, fontweight="bold")
        axes[1].set_ylabel("Average Chunk Size (chars)", fontsize=11, fontweight="bold")
        axes[1].set_title("Chunking Strategy Comparison", fontsize=12, fontweight="bold")
        axes[1].grid(alpha=0.3)
        
        plt.tight_layout()
        output_path = os.path.join(self.output_dir, "scenario2_chunking_impact.png")
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"✓ Saved: {output_path}")
        plt.close()
    
    def visualize_scenario_3(self):
        """Visualize Scenario 3: Memory Scoring Optimization"""
        if not plt:
            print("Matplotlib not available")
            return
        
        result = self.load_scenario_result("scenario3")
        if not result:
            print("Scenario 3 results not found")
            return
        
        combinations = result.get("top_10_configurations", [])
        
        # Extract data
        alphas = [c.get("alpha", 0) for c in combinations]
        betas = [c.get("beta", 0) for c in combinations]
        gammas = [c.get("gamma", 0) for c in combinations]
        scores = [c.get("avg_score", 0) for c in combinations]
        
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Plot 1: Top 10 Scores
        indices = range(1, 11)
        axes[0, 0].plot(indices, scores, marker="o", linewidth=2.5, markersize=8, color="#4ECDC4")
        axes[0, 0].fill_between(indices, scores, alpha=0.3, color="#4ECDC4")
        axes[0, 0].set_xlabel("Rank", fontsize=11, fontweight="bold")
        axes[0, 0].set_ylabel("Score", fontsize=11, fontweight="bold")
        axes[0, 0].set_title("Top 10 Weight Combinations", fontsize=12, fontweight="bold")
        axes[0, 0].grid(alpha=0.3)
        axes[0, 0].set_ylim(0.98, 1.001)
        
        # Plot 2: Weight Distribution (Stacked Bar)
        width = 0.6
        axes[0, 1].bar(indices, alphas, width, label="α (Recency)", color="#FF6B6B", alpha=0.8)
        axes[0, 1].bar(indices, betas, width, bottom=alphas, label="β (Importance)", color="#4ECDC4", alpha=0.8)
        axes[0, 1].bar(indices, gammas, width, bottom=[a+b for a,b in zip(alphas, betas)], 
                       label="γ (Relevance)", color="#45B7D1", alpha=0.8)
        axes[0, 1].set_xlabel("Combination Rank", fontsize=11, fontweight="bold")
        axes[0, 1].set_ylabel("Weight Value", fontsize=11, fontweight="bold")
        axes[0, 1].set_title("Weight Distribution", fontsize=12, fontweight="bold")
        axes[0, 1].legend(loc="upper right", fontsize=9)
        axes[0, 1].set_ylim(0, 1.0)
        axes[0, 1].grid(axis="y", alpha=0.3)
        
        # Plot 3: Alpha vs Score
        axes[1, 0].scatter(alphas, scores, s=200, alpha=0.6, c=range(10), cmap="viridis", edgecolor="black")
        axes[1, 0].set_xlabel("α (Recency Weight)", fontsize=11, fontweight="bold")
        axes[1, 0].set_ylabel("Score", fontsize=11, fontweight="bold")
        axes[1, 0].set_title("Recency Weight Impact", fontsize=12, fontweight="bold")
        axes[1, 0].grid(alpha=0.3)
        
        # Plot 4: Beta vs Score
        axes[1, 1].scatter(betas, scores, s=200, alpha=0.6, c=range(10), cmap="plasma", edgecolor="black")
        axes[1, 1].set_xlabel("β (Importance Weight)", fontsize=11, fontweight="bold")
        axes[1, 1].set_ylabel("Score", fontsize=11, fontweight="bold")
        axes[1, 1].set_title("Importance Weight Impact", fontsize=12, fontweight="bold")
        axes[1, 1].grid(alpha=0.3)
        
        plt.tight_layout()
        output_path = os.path.join(self.output_dir, "scenario3_memory_scoring.png")
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"✓ Saved: {output_path}")
        plt.close()
    
    def visualize_scenario_4(self):
        """Visualize Scenario 4: RAG vs Knowledge Graph"""
        if not plt:
            print("Matplotlib not available")
            return
        
        result = self.load_scenario_result("scenario4")
        if not result:
            print("Scenario 4 results not found")
            return
        
        # Overall comparison
        rag_avg = result.get("rag_avg_score", 0)
        kg_avg = result.get("kg_avg_score", 0)
        
        # By question type
        by_type = result.get("by_question_type", {})
        
        # Create figure
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot 1: Overall Average
        systems = ["RAG", "Knowledge Graph"]
        averages = [rag_avg, kg_avg]
        colors = ["#4ECDC4", "#FF6B6B"]
        bars = axes[0].bar(systems, averages, color=colors, alpha=0.8, edgecolor="black", width=0.6)
        axes[0].set_ylabel("Average Score", fontsize=11, fontweight="bold")
        axes[0].set_title("Scenario 4: RAG vs Knowledge Graph (Overall)", fontsize=12, fontweight="bold")
        axes[0].set_ylim(0, 1.0)
        for i, (bar, avg) in enumerate(zip(bars, averages)):
            height = bar.get_height()
            axes[0].text(bar.get_x() + bar.get_width()/2., height + 0.02,
                        f"{avg:.3f}", ha="center", va="bottom", fontweight="bold", fontsize=11)
        axes[0].grid(axis="y", alpha=0.3)
        
        # Plot 2: By Question Type
        if by_type:
            q_types = list(by_type.keys())
            rag_scores = [by_type[qt].get("rag_avg", 0) for qt in q_types]
            kg_scores = [by_type[qt].get("kg_avg", 0) for qt in q_types]
            
            x = np.arange(len(q_types))
            width = 0.35
            
            axes[1].bar(x - width/2, rag_scores, width, label="RAG", color="#4ECDC4", alpha=0.8, edgecolor="black")
            axes[1].bar(x + width/2, kg_scores, width, label="Knowledge Graph", color="#FF6B6B", alpha=0.8, edgecolor="black")
            
            axes[1].set_ylabel("Score", fontsize=11, fontweight="bold")
            axes[1].set_xlabel("Question Type", fontsize=11, fontweight="bold")
            axes[1].set_title("Performance by Question Type", fontsize=12, fontweight="bold")
            axes[1].set_xticks(x)
            axes[1].set_xticklabels(q_types, rotation=45, ha="right")
            axes[1].legend(fontsize=10)
            axes[1].set_ylim(0, 1.1)
            axes[1].grid(axis="y", alpha=0.3)
        
        plt.tight_layout()
        output_path = os.path.join(self.output_dir, "scenario4_rag_vs_kg.png")
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"✓ Saved: {output_path}")
        plt.close()
    
    def visualize_master_report(self):
        """Visualize Master Report Summary"""
        if not plt:
            print("Matplotlib not available")
            return
        
        report = self.load_master_report()
        if not report:
            print("Master report not found")
            return
        
        # Extract confirmation info
        confirmations_text = report.get("hypothesis_confirmations", [])
        confirmations = len([c for c in confirmations_text if "✓" in c])
        total_scenarios = 4
        
        # Create figure
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot 1: Hypothesis Confirmation Rate
        confirmed = confirmations
        unconfirmed = total_scenarios - confirmations
        colors_pie = ["#4ECDC4", "#FFB6B9"]
        wedges, texts, autotexts = axes[0].pie([confirmed, unconfirmed], 
                                                labels=["Confirmed", "Unconfirmed"],
                                                autopct="%1.1f%%",
                                                colors=colors_pie,
                                                startangle=90,
                                                textprops={"fontsize": 11, "fontweight": "bold"})
        axes[0].set_title(f"Hypothesis Confirmation: {confirmed}/{total_scenarios} Scenarios", 
                         fontsize=12, fontweight="bold")
        
        # Plot 2: Status Summary
        status_text = report.get("overall_status", "Unknown")
        evidence_strength = report.get("evidence_strength", "Unknown")
        
        summary_text = f"Status: {status_text}\nEvidence: {evidence_strength}"
        axes[1].text(0.5, 0.5, summary_text, 
                    ha="center", va="center",
                    fontsize=14, fontweight="bold",
                    bbox=dict(boxstyle="round", facecolor="#FFE66D", alpha=0.8, edgecolor="black", linewidth=2),
                    transform=axes[1].transAxes)
        axes[1].axis("off")
        axes[1].set_title("Research Summary", fontsize=12, fontweight="bold")
        
        plt.tight_layout()
        output_path = os.path.join(self.output_dir, "master_report_summary.png")
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"✓ Saved: {output_path}")
        plt.close()
    
    def visualize_all(self):
        """Generate all visualizations"""
        if not plt:
            print("Error: matplotlib and seaborn required for visualizations")
            print("Install with: pip install matplotlib seaborn")
            return
        
        print("\nGenerating visualizations...")
        print("=" * 60)
        
        self.visualize_scenario_1()
        self.visualize_scenario_2()
        self.visualize_scenario_3()
        self.visualize_scenario_4()
        self.visualize_master_report()
        
        print("=" * 60)
        print(f"✓ All visualizations saved to: {self.output_dir}")
        
        # List generated files
        viz_files = os.listdir(self.output_dir)
        print(f"\nGenerated {len(viz_files)} visualization files:")
        for f in sorted(viz_files):
            print(f"  - {f}")

def main():
    """Main entry point"""
    visualizer = ResultsVisualizer()
    visualizer.visualize_all()

if __name__ == "__main__":
    main()
