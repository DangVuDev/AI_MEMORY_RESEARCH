"""
LLM Client — thử OpenAI trước, tự động fallback sang Local Scoring
nếu OpenAI không có quota / không cấu hình.

Local Scoring:
  - generate_answer  : extractive (trả về câu liên quan nhất trong context)
  - evaluate_relevance: cosine similarity (question, context) bằng sentence-transformers
"""
from __future__ import annotations

# ── Local scoring (luôn hoạt động, không cần API) ─────────────────────────────

def _local_generate(question: str, context: str) -> str:
    """Extractive QA: trả về câu trong context có từ chung nhiều nhất với câu hỏi."""
    q_words = set(question.lower().split())
    best_sent, best_score = context, 0
    for sent in context.replace("\n---\n", "\n").split(". "):
        s = sent.strip()
        if not s:
            continue
        score = len(q_words & set(s.lower().split()))
        if score > best_score:
            best_score, best_sent = score, s
    return best_sent


def _local_evaluate(question: str, context: str, _answer: str) -> float:
    """
    Relevance = cosine_sim(embed(question), embed(context))
    Dùng chung embedder singleton với retriever.py (load 1 lần duy nhất).
    """
    from retriever import get_embedder
    from sentence_transformers import util

    model = get_embedder()
    q_emb = model.encode(question, convert_to_tensor=True)
    c_emb = model.encode(context[:2000], convert_to_tensor=True)
    sim   = float(util.cos_sim(q_emb, c_emb))
    return max(0.0, min(1.0, (sim + 1) / 2))


# ── OpenAI (optional) ─────────────────────────────────────────────────────────

_USE_OPENAI: bool | None = None          # None = not determined yet


def _openai_available() -> bool:
    global _USE_OPENAI
    if _USE_OPENAI is not None:
        return _USE_OPENAI
    try:
        from config import OPENAI_API_KEY, OPENAI_MODEL
        if not OPENAI_API_KEY:
            _USE_OPENAI = False
            return False
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)
        client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=3,
        )
        _USE_OPENAI = True
        print("  ℹ️  Using OpenAI gpt-4o-mini")
    except Exception:
        _USE_OPENAI = False
        print("  ℹ️  OpenAI unavailable — using Local Scoring (cosine similarity)")
    return _USE_OPENAI


def _openai_generate(question: str, context: str) -> str:
    from config import OPENAI_API_KEY, OPENAI_MODEL
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)
    prompt = (
        "Bạn là trợ lý AI. Dựa trên tài liệu sau, trả lời câu hỏi ngắn gọn.\n\n"
        f"TÀI LIỆU:\n{context}\n\nCÂU HỎI: {question}\n\nTRẢ LỜI:"
    )
    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=200, temperature=0.1,
    )
    return resp.choices[0].message.content.strip()


def _openai_evaluate(question: str, context: str, answer: str) -> float:
    from config import OPENAI_API_KEY, OPENAI_MODEL
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)
    prompt = (
        f"CÂU HỎI: {question}\nTÀI LIỆU: {context}\nTRẢ LỜI: {answer}\n\n"
        "Đánh giá mức liên quan 0.0-1.0. Chỉ trả về 1 số thập phân."
    )
    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=5, temperature=0.0,
    )
    try:
        return max(0.0, min(1.0, float(resp.choices[0].message.content.strip().split()[0])))
    except Exception:
        return 0.5


# ── Public API ─────────────────────────────────────────────────────────────────

def generate_answer(question: str, context: str) -> str:
    if _openai_available():
        try:
            return _openai_generate(question, context)
        except Exception:
            pass
    return _local_generate(question, context)


def evaluate_relevance(question: str, context: str, answer: str) -> float:
    if _openai_available():
        try:
            return _openai_evaluate(question, context, answer)
        except Exception:
            pass
    return _local_evaluate(question, context, answer)
