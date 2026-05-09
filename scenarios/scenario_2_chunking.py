"""
SCENARIO 2: Impact of Chunking Strategies

Kiểm chứng: Semantic Chunking nâng cao retrieval quality rõ rệt so với Fixed-size

4 chiến lược được test:
- Fixed-size 256: Chia theo fixed size 256 tokens, overlap 20
- Fixed-size 512: Chia theo fixed size 512 tokens, overlap 50
- Semantic: Chia dựa trên cosine similarity breakpoints
- Recursive: Chia theo cấu trúc văn bản đệ quy
"""

import json
import os
import numpy as np
from typing import List
from pathlib import Path


def load_corpus():
    """Load test corpus"""
    corpus_file = Path("data/corpus.json")
    if corpus_file.exists():
        with open(corpus_file, encoding="utf-8") as f:
            data = json.load(f)
            return data["corpus"]
    return []


def fixed_size_chunking(documents: List[str], chunk_size: int, overlap: int = 0) -> List[str]:
    """Split documents into fixed-size chunks"""
    chunks = []
    
    for doc in documents:
        words = doc.split()
        for i in range(0, len(words), chunk_size - overlap):
            chunk = " ".join(words[i:i + chunk_size])
            if chunk.strip():
                chunks.append(chunk)
    
    return chunks


def semantic_chunking(documents: List[str], threshold: float = 0.5) -> List[str]:
    """Split documents where semantic similarity drops below threshold"""
    chunks = []
    
    try:
        from sentence_transformers import SentenceTransformer, util
        embedder = SentenceTransformer("all-MiniLM-L6-v2")
        
        for doc in documents:
            sentences = doc.split(". ")
            if len(sentences) <= 1:
                chunks.append(doc)
                continue
            
            sentence_embs = embedder.encode(sentences, convert_to_tensor=True)
            
            current_chunk = [sentences[0]]
            
            for i in range(1, len(sentences)):
                if i > 0:
                    similarity = util.cos_sim(sentence_embs[i-1], sentence_embs[i]).item()
                    
                    if similarity >= threshold:
                        current_chunk.append(sentences[i])
                    else:
                        chunks.append(". ".join(current_chunk))
                        current_chunk = [sentences[i]]
            
            if current_chunk:
                chunks.append(". ".join(current_chunk))
        
        return chunks
    
    except Exception as e:
        print(f"  ⚠ Semantic chunking error: {e}, using fixed-size instead")
        return fixed_size_chunking(documents, 256)


def recursive_chunking(documents: List[str], max_size: int = 512) -> List[str]:
    """Recursively split documents respecting structure"""
    chunks = []
    
    for doc in documents:
        # Split by sentences first
        sentences = doc.split(". ")
        
        current_chunk = []
        current_len = 0
        
        for sentence in sentences:
            sentence_len = len(sentence.split())
            
            if current_len + sentence_len <= max_size:
                current_chunk.append(sentence)
                current_len += sentence_len
            else:
                if current_chunk:
                    chunks.append(". ".join(current_chunk))
                current_chunk = [sentence]
                current_len = sentence_len
        
        if current_chunk:
            chunks.append(". ".join(current_chunk))
    
    return chunks


def evaluate_chunking_quality(chunks: List[str], query: str) -> float:
    """Evaluate chunking quality by measuring retrieval effectiveness"""
    query_words = set(query.lower().split())
    
    # Find most relevant chunks
    chunk_scores = []
    for chunk in chunks:
        chunk_words = set(chunk.lower().split())
        overlap = len(query_words & chunk_words)
        chunk_scores.append(overlap)
    
    if not chunk_scores:
        return 0.0
    
    # Average relevance of top 3 chunks
    top_scores = sorted(chunk_scores, reverse=True)[:3]
    return np.mean(top_scores) / max(len(query_words), 1)


def run_scenario_2():
    """
    Run Scenario 2: Chunking Strategy Comparison
    """
    print("\n" + "="*80)
    print("SCENARIO 2: Chunking Strategy Impact Analysis")
    print("="*80)
    
    # Load corpus
    print("\n[1] Loading test data...")
    corpus = load_corpus()
    
    if not corpus:
        print("  ✗ Corpus not found. Run: python prepare_data.py")
        return None
    
    print(f"  ✓ Loaded {len(corpus)} documents")
    
    # Create long document for testing (simulate long manual/documentation)
    long_doc = " ".join(corpus) * 2  # Simulate longer document
    
    test_queries = [
        "Chính sách hoàn tiền",
        "Sản phẩm A tính năng",
        "Công ty XYZ Alice Bob",
        "Dự án CloudCore",
        "Support 24/7",
    ]
    
    # Test chunking strategies
    print("\n[2] Testing chunking strategies...\n")
    
    strategies = {
        "Fixed_256": {
            "func": lambda docs: fixed_size_chunking(docs, chunk_size=256, overlap=20),
            "label": "Fixed-size 256 tokens"
        },
        "Fixed_512": {
            "func": lambda docs: fixed_size_chunking(docs, chunk_size=512, overlap=50),
            "label": "Fixed-size 512 tokens"
        },
        "Semantic": {
            "func": lambda docs: semantic_chunking(docs, threshold=0.5),
            "label": "Semantic (similarity > 0.5)"
        },
        "Recursive": {
            "func": lambda docs: recursive_chunking(docs, max_size=512),
            "label": "Recursive (structure-aware)"
        },
    }
    
    results = {}
    
    for strategy_name, strategy_config in strategies.items():
        print(f"Strategy {strategy_name} ({strategy_config['label']}):")
        
        # Create chunks
        chunks = strategy_config["func"]([long_doc])
        
        # Evaluate on test queries
        qualities = []
        for query in test_queries:
            quality = evaluate_chunking_quality(chunks, query)
            qualities.append(quality)
        
        avg_quality = np.mean(qualities)
        avg_chunk_size = len(long_doc) / len(chunks) if chunks else 0
        
        print(f"  Number of chunks: {len(chunks)}")
        print(f"  Avg chunk size: {avg_chunk_size:.0f} chars")
        print(f"  Avg retrieval quality: {avg_quality:.3f}")
        print()
        
        results[strategy_name] = {
            "strategy": strategy_config["label"],
            "num_chunks": len(chunks),
            "avg_chunk_size": float(avg_chunk_size),
            "avg_quality": float(avg_quality),
            "quality_scores": [float(q) for q in qualities],
        }
    
    # Analysis
    print("[3] Comparative Analysis\n")
    
    fixed_256_quality = results["Fixed_256"]["avg_quality"]
    semantic_quality = results["Semantic"]["avg_quality"]
    recursive_quality = results["Recursive"]["avg_quality"]
    
    semantic_improvement = ((semantic_quality - fixed_256_quality) / max(fixed_256_quality, 0.001)) * 100
    recursive_improvement = ((recursive_quality - fixed_256_quality) / max(fixed_256_quality, 0.001)) * 100
    
    print(f"Fixed 256 (baseline):      {fixed_256_quality:.3f}")
    print(f"Fixed 512:                 {results['Fixed_512']['avg_quality']:.3f}")
    print(f"Semantic (intelligent):    {semantic_quality:.3f} ({semantic_improvement:+.1f}%)")
    print(f"Recursive (structured):    {recursive_quality:.3f} ({recursive_improvement:+.1f}%)")
    print()
    
    # Determine best strategy
    best_strategy = max(results.items(), key=lambda x: x[1]["avg_quality"])[0]
    
    print("✓ KEY FINDING:")
    if best_strategy in ["Semantic", "Recursive"]:
        print(f"  {best_strategy} is the best strategy for retrieval quality")
        print("  → Intelligent chunking outperforms fixed-size")
    else:
        print("  Fixed-size chunking is most effective for this corpus")
    
    # Save results
    output_data = {
        "scenario": "scenario_2_chunking",
        "strategies": results,
        "best_strategy": best_strategy,
        "improvements": {
            "Semantic_vs_Fixed256": float(semantic_improvement),
            "Recursive_vs_Fixed256": float(recursive_improvement),
        }
    }
    
    os.makedirs("results/scenario2", exist_ok=True)
    
    with open("results/scenario2/results.json", "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Results saved to: results/scenario2/results.json")
    print("="*80)
    
    return output_data


if __name__ == "__main__":
    try:
        run_scenario_2()
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
