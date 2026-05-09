"""
SCENARIO 3: Episodic Memory Scoring Optimization

Kiểm chứng: Tìm trọng số tối ưu cho scoring formula
Score(m) = α*Recency + β*Importance + γ*Relevance

Grid search các tổ hợp α, β, γ với α+β+γ = 1
"""

import json
import os
import numpy as np
from typing import List, Tuple, Dict
from itertools import product


class EpisodicMemory:
    """Represent a memory with recency, importance, and relevance"""
    
    def __init__(self, content: str, age_hours: int, importance: int):
        self.content = content
        self.age_hours = age_hours
        self.importance = importance  # Scale 1-10
    
    def compute_recency(self, max_age_hours: int = 1000) -> float:
        """Recency score: inversely proportional to age"""
        return 1.0 - min(self.age_hours / max_age_hours, 1.0)
    
    def compute_importance(self) -> float:
        """Importance score: normalized 1-10 to 0-1"""
        return self.importance / 10.0
    
    def compute_relevance(self, query: str) -> float:
        """Relevance score: word overlap with query"""
        query_words = set(query.lower().split())
        content_words = set(self.content.lower().split())
        
        if not query_words or not content_words:
            return 0.0
        
        overlap = len(query_words & content_words)
        union = len(query_words | content_words)
        
        return overlap / union if union > 0 else 0.0
    
    def compute_score(self, query: str, alpha: float, beta: float, gamma: float) -> float:
        """Compute combined score with given weights"""
        recency = self.compute_recency()
        importance = self.compute_importance()
        relevance = self.compute_relevance(query)
        
        return alpha * recency + beta * importance + gamma * relevance


def prepare_test_memories() -> List[EpisodicMemory]:
    """Create test memories with varied ages and importance"""
    memories = [
        EpisodicMemory("Chính sách hoàn tiền 30 ngày", age_hours=10, importance=9),
        EpisodicMemory("Sản phẩm A tính năng ML", age_hours=20, importance=8),
        EpisodicMemory("Giá gói Standard 29 USD", age_hours=5, importance=7),
        EpisodicMemory("Alice CEO Công ty XYZ", age_hours=100, importance=9),
        EpisodicMemory("Bob CTO quản lý Engineering", age_hours=50, importance=8),
        EpisodicMemory("CloudCore dự án 2024", age_hours=15, importance=10),
        EpisodicMemory("DataMind ML platform", age_hours=200, importance=7),
        EpisodicMemory("Support 24/7 Premium", age_hours=30, importance=8),
        EpisodicMemory("Roadmap 2024 công ty", age_hours=2, importance=9),
        EpisodicMemory("Payment methods credit card paypal", age_hours=25, importance=6),
    ]
    return memories


def prepare_test_queries() -> List[str]:
    """Create test queries for evaluation"""
    queries = [
        "Chính sách hoàn tiền",
        "Sản phẩm A tính năng",
        "Alice Bob công ty",
        "CloudCore dự án",
    ]
    return queries


def evaluate_weight_combination(memories: List[EpisodicMemory], 
                               queries: List[str],
                               alpha: float, 
                               beta: float, 
                               gamma: float) -> float:
    """
    Evaluate a weight combination by computing average score
    for finding the top memory for each query
    """
    total_score = 0.0
    
    for query in queries:
        scores = [mem.compute_score(query, alpha, beta, gamma) for mem in memories]
        max_score = max(scores) if scores else 0.0
        total_score += max_score
    
    return total_score / len(queries) if queries else 0.0


def run_scenario_3():
    """
    Run Scenario 3: Memory Scoring Optimization
    """
    print("\n" + "="*80)
    print("SCENARIO 3: Episodic Memory Scoring Weight Optimization")
    print("="*80)
    
    # Prepare data
    print("\n[1] Preparing test data...")
    memories = prepare_test_memories()
    queries = prepare_test_queries()
    
    print(f"  ✓ Memories: {len(memories)}")
    print(f"  ✓ Queries: {len(queries)}")
    
    # Grid search
    print("\n[2] Grid search for optimal weights...\n")
    
    weights_to_test = np.linspace(0, 1, 11)  # [0.0, 0.1, 0.2, ..., 1.0]
    results_list = []
    
    for alpha, beta, gamma in product(weights_to_test, repeat=3):
        # Only test combinations where weights sum to 1.0
        if abs(alpha + beta + gamma - 1.0) > 0.01:
            continue
        
        score = evaluate_weight_combination(memories, queries, alpha, beta, gamma)
        
        results_list.append({
            "alpha": float(alpha),
            "beta": float(beta),
            "gamma": float(gamma),
            "avg_score": float(score),
        })
    
    # Sort by score
    results_list.sort(key=lambda x: x["avg_score"], reverse=True)
    
    print(f"✓ Tested {len(results_list)} weight combinations\n")
    print("Top 10 Weight Configurations:\n")
    
    for i, result in enumerate(results_list[:10]):
        print(f"#{i+1}: α={result['alpha']:.1f}, β={result['beta']:.1f}, γ={result['gamma']:.1f}")
        print(f"    Score: {result['avg_score']:.3f}")
        print()
    
    best = results_list[0]
    
    # Analysis
    print("[3] Analysis & Findings\n")
    
    print(f"Optimal weight configuration:")
    print(f"  α (Recency):    {best['alpha']:.1f}")
    print(f"  β (Importance): {best['beta']:.1f}")
    print(f"  γ (Relevance):  {best['gamma']:.1f}")
    print(f"  Score: {best['avg_score']:.3f}")
    print()
    
    # Interpretation
    print("✓ KEY FINDINGS:\n")
    
    if best['gamma'] >= 0.7:
        print(f"  Relevance (γ={best['gamma']:.1f}) dominates the optimal scoring!")
        print("  → Quality matches are most important")
    elif best['beta'] >= 0.7:
        print(f"  Importance (β={best['beta']:.1f}) dominates the optimal scoring!")
        print("  → Priority/importance matters most")
    elif best['alpha'] >= 0.7:
        print(f"  Recency (α={best['alpha']:.1f}) dominates the optimal scoring!")
        print("  → Recent memories matter most")
    else:
        print("  Balanced weights are optimal")
        print("  → All factors contribute equally")
    
    # Save results
    output_data = {
        "scenario": "scenario_3_scoring",
        "total_combinations_tested": len(results_list),
        "best_weights": {
            "alpha_recency": best['alpha'],
            "beta_importance": best['beta'],
            "gamma_relevance": best['gamma'],
        },
        "best_score": best['avg_score'],
        "top_10_configurations": results_list[:10],
        "all_results": results_list,  # Save all for later analysis
    }
    
    os.makedirs("results/scenario3", exist_ok=True)
    
    with open("results/scenario3/results.json", "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Results saved to: results/scenario3/results.json")
    print("="*80)
    
    return output_data


if __name__ == "__main__":
    try:
        run_scenario_3()
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
