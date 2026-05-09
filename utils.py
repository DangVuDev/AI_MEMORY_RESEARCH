"""
Metrics for evaluating Memory AI system performance
Based on Memory AI Research documentation
"""

import numpy as np
from typing import List, Tuple
from scipy import stats


class RAGMetrics:
    """Metrics for evaluating RAG system quality"""
    
    @staticmethod
    def answer_relevance(question_emb: np.ndarray, answer_emb: np.ndarray) -> float:
        """
        Cosine similarity between question and answer embeddings
        Range: 0-1 (1 = perfect relevance)
        """
        if len(question_emb) == 0 or len(answer_emb) == 0:
            return 0.0
        
        question_emb = question_emb / (np.linalg.norm(question_emb) + 1e-8)
        answer_emb = answer_emb / (np.linalg.norm(answer_emb) + 1e-8)
        
        return float(np.dot(question_emb, answer_emb))
    
    @staticmethod
    def context_precision(retrieved_chunks: List[str], relevant_chunks: List[str]) -> float:
        """
        Tỷ lệ chunk retrieved thực sự có ích
        precision = relevant_retrieved / total_retrieved
        """
        if not retrieved_chunks:
            return 0.0
        
        relevant_count = sum(1 for chunk in retrieved_chunks if chunk in relevant_chunks)
        return relevant_count / len(retrieved_chunks)
    
    @staticmethod
    def context_recall(retrieved_chunks: List[str], relevant_chunks: List[str]) -> float:
        """
        Bao phủ thông tin cần thiết
        recall = relevant_retrieved / total_relevant
        """
        if not relevant_chunks:
            return 1.0
        
        relevant_count = sum(1 for chunk in retrieved_chunks if chunk in relevant_chunks)
        return relevant_count / len(relevant_chunks)
    
    @staticmethod
    def faithfulness(claims: List[str], context_claims: List[str]) -> float:
        """
        Tỷ lệ claims trong answer được hỗ trợ bởi context
        """
        if not claims:
            return 1.0
        
        supported = sum(1 for claim in claims if claim in context_claims)
        return supported / len(claims)


class EpisodicMemoryMetrics:
    """Metrics for evaluating Episodic Memory retrieval"""
    
    @staticmethod
    def memory_score(recency: float, importance: float, relevance: float,
                    alpha: float, beta: float, gamma: float) -> float:
        """
        Combined memory scoring formula
        Score = α*Recency + β*Importance + γ*Relevance
        """
        return alpha * recency + beta * importance + gamma * relevance


class StatisticalAnalysis:
    """Statistical significance testing"""
    
    @staticmethod
    def paired_t_test(system_a_scores: List[float], 
                     system_b_scores: List[float]) -> Tuple[float, float]:
        """
        Paired t-test between two systems
        Returns: (t_statistic, p_value)
        """
        if len(system_a_scores) != len(system_b_scores) or len(system_a_scores) < 2:
            return 0.0, 1.0
        
        t_stat, p_value = stats.ttest_rel(system_a_scores, system_b_scores)
        return float(t_stat), float(p_value)
    
    @staticmethod
    def cohens_d(system_a_scores: List[float], 
                system_b_scores: List[float]) -> float:
        """
        Calculate Cohen's d effect size
        Interpretation: 0.2=small, 0.5=medium, 0.8=large
        """
        mean_a = np.mean(system_a_scores)
        mean_b = np.mean(system_b_scores)
        
        n_a = len(system_a_scores)
        n_b = len(system_b_scores)
        
        var_a = np.var(system_a_scores, ddof=1)
        var_b = np.var(system_b_scores, ddof=1)
        
        pooled_std = np.sqrt(((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2))
        
        if pooled_std == 0:
            return 0.0
        
        return float((mean_b - mean_a) / pooled_std)
    
    @staticmethod
    def describe_stats(scores: List[float]) -> dict:
        """Descriptive statistics for a list of scores"""
        scores_arr = np.array(scores)
        return {
            "mean": float(np.mean(scores_arr)),
            "std": float(np.std(scores_arr, ddof=1)),
            "min": float(np.min(scores_arr)),
            "max": float(np.max(scores_arr)),
            "median": float(np.median(scores_arr)),
        }
