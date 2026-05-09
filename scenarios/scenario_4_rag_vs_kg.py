"""
SCENARIO 4: RAG vs Knowledge Graph for Complex Questions

Kiểm chứng: KG retrieval vượt trội RAG trên các câu hỏi phức tạp

Loại câu hỏi test:
- Simple fact: Lookup đơn giản
- Multi-hop 2: 2 bước traversal
- Multi-hop 3+: 3+ bước traversal
- Aggregation: Tổng hợp từ nhiều entity
- Comparative: So sánh
"""

import json
import os
import numpy as np
from typing import List, Dict, Set


class SimpleKnowledgeGraph:
    """Simple knowledge graph representation"""
    
    def __init__(self):
        """Initialize with entities and relationships from our test data"""
        self.entities = {
            # People
            "Alice": {"role": "CEO", "dept": "Product", "manages": ["Portal"], "age_hours": 100},
            "Bob": {"role": "CTO", "dept": "Engineering", "manages": ["CloudCore", "DataMind"], "age_hours": 50},
            
            # Departments
            "Engineering": {"size": 20, "projects": ["CloudCore", "DataMind", "Portal"]},
            "Product": {"size": 8, "projects": ["Portal"]},
            
            # Projects
            "CloudCore": {"status": "active", "leads": ["Bob", "Alice"], "q_target": "Q4 2024"},
            "DataMind": {"status": "active", "leads": ["Bob"]},
            "Portal": {"status": "active", "leads": ["Alice"]},
            
            # Policies
            "Refund_Policy": {"days": 30, "percentage": 100},
            "Support": {"level": "24/7", "channels": ["email", "chat", "phone"]},
        }
        
        self.relationships = {
            # Person -> Project
            ("Alice", "leads"): ["Portal", "CloudCore"],
            ("Bob", "leads"): ["CloudCore", "DataMind"],
            
            # Person -> Department
            ("Alice", "manages"): ["Product"],
            ("Bob", "manages"): ["Engineering"],
            
            # Department -> Project
            ("Engineering", "owns"): ["CloudCore", "DataMind"],
            ("Product", "owns"): ["Portal"],
            
            # Department -> Members
            ("Engineering", "has_members"): 20,
            ("Product", "has_members"): 8,
        }
    
    def traverse(self, start_entity: str, depth: int = 3) -> Set[str]:
        """
        Traverse KG from a starting entity to find all reachable entities
        """
        visited = {start_entity}
        current_level = {start_entity}
        
        for _ in range(depth):
            next_level = set()
            
            for entity in current_level:
                # Check direct relationships
                for (source, relation), targets in self.relationships.items():
                    if source == entity:
                        if isinstance(targets, list):
                            for target in targets:
                                if target not in visited:
                                    next_level.add(target)
                                    visited.add(target)
                
                # Check entity properties
                if entity in self.entities:
                    for key, val in self.entities[entity].items():
                        if isinstance(val, list):
                            for item in val:
                                if item not in visited:
                                    next_level.add(item)
                                    visited.add(item)
            
            current_level = next_level
            if not current_level:
                break
        
        return visited
    
    def answer_question_kg(self, question: str) -> Set[str]:
        """Answer using KG traversal"""
        # Find entities mentioned in question
        mentioned_entities = []
        for entity in self.entities.keys():
            if entity.lower() in question.lower():
                mentioned_entities.append(entity)
        
        # Traverse from those entities
        result = set()
        for entity in mentioned_entities:
            result.update(self.traverse(entity, depth=3))
        
        return result


def rag_retrieval(question: str, documents: List[str], k: int = 5) -> List[str]:
    """
    Simple RAG retrieval using word overlap
    """
    query_words = set(question.lower().split())
    
    scored_docs = []
    for doc in documents:
        doc_words = set(doc.lower().split())
        overlap = len(query_words & doc_words)
        scored_docs.append((doc, overlap))
    
    scored_docs.sort(key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in scored_docs[:k]]


def evaluate_answer_quality(question: str, retrieved: Set[str], answer_type: str) -> float:
    """Evaluate quality of retrieval for this question type"""
    
    # Define expected answers for each question type
    expected = {
        "simple_fact": {"Refund_Policy", "Support"},
        "multi_hop_2": {"Alice", "Bob", "CloudCore", "DataMind", "Portal"},
        "multi_hop_3": {"Alice", "Bob", "Engineering", "Product", "CloudCore"},
        "aggregation": {"CloudCore", "DataMind", "Portal", "Alice", "Bob"},
        "comparative": {"Alice", "Bob", "Product", "Engineering"},
    }
    
    expected_set = expected.get(answer_type, set())
    
    if not expected_set:
        return 0.0
    
    # Calculate how many expected entities are in the retrieved result
    hits = len(retrieved & expected_set)
    return min(1.0, hits / len(expected_set))


def run_scenario_4():
    """
    Run Scenario 4: RAG vs Knowledge Graph
    """
    print("\n" + "="*80)
    print("SCENARIO 4: RAG vs Knowledge Graph Comparison")
    print("="*80)
    
    # Setup
    print("\n[1] Preparing test setup...")
    kg = SimpleKnowledgeGraph()
    
    # Test documents for RAG
    documents = [
        "Alice is CEO managing Product department and Portal project",
        "Bob is CTO managing Engineering with 20 people",
        "CloudCore project led by Alice and Bob targets Q4 2024",
        "DataMind is AI/ML project led by Bob",
        "Portal project managed by Alice in Product",
        "Engineering department owns CloudCore and DataMind projects",
        "Refund policy: 100% refund within 30 days",
        "Support: 24/7 via email, chat, and phone",
        "Company has 50 employees in multiple departments",
        "Roadmap 2024: CloudCore, DataMind, Portal completion",
    ]
    
    # Test questions
    test_questions = [
        ("Chính sách hoàn tiền?", "simple_fact"),
        ("Alice quản lý gì?", "multi_hop_2"),
        ("Bob dẫn dắt những dự án nào?", "multi_hop_2"),
        ("Alice và Bob cùng làm việc trên dự án nào?", "multi_hop_3"),
        ("Mối quan hệ giữa Engineering và các dự án là gì?", "multi_hop_3"),
        ("Liệt kê tất cả các dự án và người lãnh đạo?", "aggregation"),
        ("So sánh Alice và Bob về quyền hạn?", "comparative"),
        ("Công ty được tổ chức như thế nào?", "aggregation"),
        ("Roadmap 2024 bao gồm gì?", "simple_fact"),
    ]
    
    print(f"  ✓ Knowledge Graph initialized")
    print(f"  ✓ Documents: {len(documents)}")
    print(f"  ✓ Questions: {len(test_questions)}")
    
    # Test both approaches
    print("\n[2] Testing both retrieval methods...\n")
    
    rag_scores = []
    kg_scores = []
    results_by_type = {}
    
    for question, q_type in test_questions:
        # RAG approach
        retrieved_docs = rag_retrieval(question, documents, k=3)
        rag_answer_words = set(" ".join(retrieved_docs).split())
        rag_score = evaluate_answer_quality(question, rag_answer_words, q_type)
        rag_scores.append(rag_score)
        
        # KG approach
        kg_entities = kg.answer_question_kg(question)
        kg_score = evaluate_answer_quality(question, kg_entities, q_type)
        kg_scores.append(kg_score)
        
        # Track by question type
        if q_type not in results_by_type:
            results_by_type[q_type] = {"rag": [], "kg": []}
        results_by_type[q_type]["rag"].append(rag_score)
        results_by_type[q_type]["kg"].append(kg_score)
    
    rag_avg = np.mean(rag_scores)
    kg_avg = np.mean(kg_scores)
    
    print(f"RAG System (Similarity-based):      {rag_avg:.3f}")
    print(f"KG System (Graph-based):           {kg_avg:.3f}")
    
    improvement = ((kg_avg - rag_avg) / max(rag_avg, 0.001)) * 100
    print(f"\nImprovement: +{improvement:.1f}%")
    
    # Analysis by question type
    print("\n[3] Analysis by Question Type\n")
    
    for q_type in sorted(results_by_type.keys()):
        rag_type_avg = np.mean(results_by_type[q_type]["rag"])
        kg_type_avg = np.mean(results_by_type[q_type]["kg"])
        type_improvement = ((kg_type_avg - rag_type_avg) / max(rag_type_avg, 0.001)) * 100 if rag_type_avg > 0 else 0
        
        print(f"{q_type}:")
        print(f"  RAG: {rag_type_avg:.3f}, KG: {kg_type_avg:.3f} ({type_improvement:+.1f}%)\n")
    
    # Key findings
    print("✓ KEY FINDINGS:\n")
    
    if kg_avg > rag_avg:
        print(f"  Knowledge Graph outperforms RAG by {improvement:.1f}%")
        
        # Check where KG excels
        multi_hop_rag = np.mean(results_by_type.get("multi_hop_2", {}).get("rag", [0]))
        multi_hop_kg = np.mean(results_by_type.get("multi_hop_2", {}).get("kg", [0]))
        
        if multi_hop_kg > multi_hop_rag * 1.5:
            print("  → KG excels on multi-hop questions requiring relationship traversal")
        print("  → Relationship-aware retrieval is more effective for complex queries")
    else:
        print("  RAG is competitive with KG on this test set")
    
    # Save results
    output_data = {
        "scenario": "scenario_4_rag_vs_kg",
        "rag_avg_score": float(rag_avg),
        "kg_avg_score": float(kg_avg),
        "improvement_percent": float(improvement),
        "by_question_type": {
            q_type: {
                "rag_avg": float(np.mean(data["rag"])),
                "kg_avg": float(np.mean(data["kg"])),
            }
            for q_type, data in results_by_type.items()
        },
        "individual_scores": {
            "rag": [float(s) for s in rag_scores],
            "kg": [float(s) for s in kg_scores],
        }
    }
    
    os.makedirs("results/scenario4", exist_ok=True)
    
    with open("results/scenario4/results.json", "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Results saved to: results/scenario4/results.json")
    print("="*80)
    
    return output_data


if __name__ == "__main__":
    try:
        run_scenario_4()
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
