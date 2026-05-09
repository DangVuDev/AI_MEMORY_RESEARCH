"""Debug version of Scenario 1"""
import json
import time
from pathlib import Path

print("[Step 1] Testing corpus load...")
corpus_file = Path("data/corpus.json")
t0 = time.time()
with open(corpus_file, encoding="utf-8") as f:
    data = json.load(f)
    corpus = data["corpus"]
print(f"  ✓ Loaded {len(corpus)} documents in {time.time()-t0:.2f}s")

print("\n[Step 2] Testing questions load...")
questions_file = Path("data/questions.json")
t0 = time.time()
with open(questions_file, encoding="utf-8") as f:
    data = json.load(f)
    questions = [(q["text"], q["type"]) for q in data.get("questions", [])]
print(f"  ✓ Loaded {len(questions)} questions in {time.time()-t0:.2f}s")

print("\n[Step 3] Loading embedding model...")
t0 = time.time()
try:
    from sentence_transformers import SentenceTransformer
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    print(f"  ✓ Model loaded in {time.time()-t0:.2f}s")
except Exception as e:
    print(f"  ✗ Error: {e}")
    exit(1)

print("\n[Step 4] Computing first 5 corpus embeddings...")
t0 = time.time()
test_corpus = corpus[:5]
embs = embedder.encode(test_corpus, convert_to_tensor=True)
print(f"  ✓ Computed 5 embeddings in {time.time()-t0:.2f}s")
print(f"    Shape: {embs.shape}")

print("\n[Step 5] Computing all 300 corpus embeddings (slow)...")
t0 = time.time()
all_embs = embedder.encode(corpus, convert_to_tensor=True)
elapsed = time.time() - t0
print(f"  ✓ Computed {len(corpus)} embeddings in {elapsed:.2f}s ({elapsed/len(corpus):.3f}s per doc)")
print(f"    Shape: {all_embs.shape}")

print("\n[Step 6] Testing cosine similarity...")
query = questions[0][0]
t0 = time.time()
query_emb = embedder.encode(query, convert_to_tensor=True)
from sentence_transformers import util
scores = util.cos_sim(query_emb, all_embs)[0]
top_idx = scores.topk(5).indices.tolist()
elapsed = time.time() - t0
print(f"  ✓ Similarity computed in {elapsed:.3f}s")
print(f"    Top results: {[corpus[i][:50] for i in top_idx]}")

print("\n✓ All steps completed successfully!")
