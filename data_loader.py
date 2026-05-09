#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Data loading utilities for the research framework.
Loads corpus and questions from JSON files.
"""

import json
import os
from typing import Dict, List, Tuple

class DataLoader:
    """Load corpus and questions from data files"""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
    
    def load_corpus(self) -> List[str]:
        """Load corpus from corpus.json"""
        corpus_path = os.path.join(self.data_dir, "corpus.json")
        if not os.path.exists(corpus_path):
            raise FileNotFoundError(f"Corpus file not found: {corpus_path}")
        
        with open(corpus_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        return data.get("corpus", [])
    
    def load_questions(self) -> List[Dict]:
        """Load questions from questions.json"""
        questions_path = os.path.join(self.data_dir, "questions.json")
        if not os.path.exists(questions_path):
            raise FileNotFoundError(f"Questions file not found: {questions_path}")
        
        with open(questions_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        return data.get("questions", [])
    
    def get_questions_by_type(self, q_type: str) -> List[Dict]:
        """Get questions filtered by type"""
        questions = self.load_questions()
        return [q for q in questions if q.get("type") == q_type]
    
    def get_corpus_stats(self) -> Dict:
        """Get corpus statistics"""
        corpus = self.load_corpus()
        return {
            "total_documents": len(corpus),
            "avg_length": sum(len(doc) for doc in corpus) / len(corpus) if corpus else 0,
            "min_length": min(len(doc) for doc in corpus) if corpus else 0,
            "max_length": max(len(doc) for doc in corpus) if corpus else 0,
        }
    
    def get_questions_stats(self) -> Dict:
        """Get questions statistics"""
        questions = self.load_questions()
        type_counts = {}
        for q in questions:
            q_type = q.get("type", "unknown")
            type_counts[q_type] = type_counts.get(q_type, 0) + 1
        
        return {
            "total_questions": len(questions),
            "by_type": type_counts,
        }

def load_and_display_info():
    """Load and display data information"""
    loader = DataLoader()
    
    print("=" * 70)
    print("DATA LOADING INFORMATION")
    print("=" * 70)
    
    # Corpus stats
    corpus_stats = loader.get_corpus_stats()
    print("\nCorpus Statistics:")
    print(f"  Total Documents: {corpus_stats['total_documents']}")
    print(f"  Average Length: {corpus_stats['avg_length']:.0f} characters")
    print(f"  Min/Max Length: {corpus_stats['min_length']}/{corpus_stats['max_length']}")
    
    # Questions stats
    q_stats = loader.get_questions_stats()
    print(f"\nQuestions Statistics:")
    print(f"  Total Questions: {q_stats['total_questions']}")
    print(f"  By Type:")
    for q_type, count in sorted(q_stats['by_type'].items()):
        print(f"    - {q_type}: {count}")
    
    print("\n" + "=" * 70)

if __name__ == "__main__":
    load_and_display_info()
