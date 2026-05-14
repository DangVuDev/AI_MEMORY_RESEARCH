MEMORY AI RESEARCH

Test Scenarios & Hypothesis Verification


🧪 GIẢ ĐỊNH CẦN KIỂM CHỨNG

"Memory retrieval quality quan trọng hơn context window size"


Ý nghĩa: Một hệ thống Memory AI có cơ chế retrieval tốt

(chính xác, liên quan) sẽ cho ra kết quả tốt hơn một hệ thống

có context window lớn nhưng retrieval kém chất lượng.

Tài liệu: Memory_AI Research Notes | Ngày: 07/05/2026


1. Tổng Quan Giả Định & Mục Tiêu Test

1.1 Phân Tích Giả Định
Dựa trên tài liệu nghiên cứu, giả định được hiểu như sau:

Biến số

Giải thích

Biến độc lập

Retrieval Quality (chất lượng truy xuất) vs Context Window Size (kích thước cửa sổ ngữ cảnh)

Biến phụ thuộc

Chất lượng phản hồi cuối cùng của hệ thống Memory AI

Điều kiện kiểm soát

Model LLM, dữ liệu test, prompt template giống nhau

Kết quả kỳ vọng

Retrieval quality cao + context nhỏ > Retrieval kém + context lớn


1.2 Liên Hệ Với Nghiên Cứu
Từ tài liệu Memory AI, các yếu tố ảnh hưởng đến retrieval quality bao gồm:

Chunking strategy: Fixed-size vs Semantic vs Recursive vs Document-structure
Embedding model: Chất lượng vector representation trong không gian cao chiều (1536 chiều)
Retrieval mechanism: Standard, Sentence Window, Auto-merging, Knowledge Graph traversal
Retrieval scoring: Tính gần đây (Recency) + Tầm quan trọng (Importance) + Độ liên quan (Relevance)
Cosine similarity trong không gian vector để đo độ liên quan

1.3 Các Mức Đo Lường
Các metric được sử dụng để đánh giá chất lượng:

Metric

Đo điều gì

Công thức / Cách tính

Answer Relevance

Độ liên quan câu trả lời với câu hỏi

Cosine Sim(answer_emb, question_emb)

Context Precision

Tỷ lệ chunk retrieved thực sự có ích

Relevant chunks / Total retrieved chunks

Context Recall

Bao phủ thông tin cần thiết

Retrieved relevant / Total relevant in DB

Faithfulness

Câu trả lời có trình bày đúng thông tin được retrieve không

Claims from context / Total claims in answer

Latency (ms)

Tốc độ xử lý của hệ thống

Tổng thời gian từ query → response


2. Các Kịch Bản Test

Kịch Bản 1: So Sánh Retrieval Quality vs Context Size cơ bản
🎯 Mục Tiêu

Kiểm chứng trực tiếp: retrieval tốt + context nhỏ có thể địch được context lớn + retrieval kém


Thiết kế Thí nghiệm

Sử dụng 1 corpus dữ liệu giả định (FAQ sản phẩm, tài liệu kỹ thuật ~500 chunks). Chạy 4 hệ thống:

Hệ thống

Retrieval Method

Context Window

Kỳ vọng

A (Baseline)

Random (kông liên quan)

128K tokens

Xấu

B

Standard RAG (Cosine Sim)

4K tokens

Trung bình

C

Semantic Chunking + Rerank

4K tokens

Tốt

D

KG Traversal + Rerank

4K tokens

Tốt nhất


Bộ Câu Hỏi Test (20 câu)

Phân loại thế này theo loại quốc câu hỏi:

Loại 1 – Câu hỏi đơn giản (Fact Lookup): "Chính sách hoàn tiền là gì?" → 5 câu
Loại 2 – Câu hỏi đa bước (Multi-hop): "So sánh sản phẩm A và B về tính năng X?" → 10 câu
Loại 3 – Câu hỏi tổng hợp (Synthesis): "Tóm tắt các lỗi phổ biến nhất?" → 5 câu

Code Kiểm Tra (Python)

# scenario_1_basic.py

import os, time, json

from anthropic import Anthropic

from sentence_transformers import SentenceTransformer, util


client = Anthropic()

embedder = SentenceTransformer("all-MiniLM-L6-v2")


def retrieve_random(query, chunks, top_k=5):

    import random

    return random.sample(chunks, min(top_k, len(chunks)))


def retrieve_cosine(query, chunks, top_k=5):

    q_emb = embedder.encode(query, convert_to_tensor=True)

    c_embs = embedder.encode(chunks, convert_to_tensor=True)

    scores = util.cos_sim(q_emb, c_embs)[0]

    top_idx = scores.topk(top_k).indices.tolist()

    return [chunks[i] for i in top_idx]


def run_system(query, retrieved, label):

    context = "\n---\n".join(retrieved)

    t0 = time.time()

    resp = client.messages.create(

        model="claude-sonnet-4-20250514",

        max_tokens=512,

        messages=[{"role":"user","content":

            f"Context:\n{context}\n\nQuestion: {query}"}])

    latency = (time.time() - t0) * 1000

    return resp.content[0].text, latency


Bảng Kết Quả Dự Kiến

Hệ thống

Relevance

Precision

Recall

Faithfulness

Latency

A: Random+128K

~0.4

~0.2

~0.9

~0.5

Cao

B: RAG+4K

~0.7

~0.6

~0.6

~0.7

Trung bình

C: Semantic+4K

~0.85

~0.8

~0.7

~0.85

Thấp

D: KG+4K

~0.9

~0.85

~0.75

~0.9

Thấp


Kịch Bản 2: Ảnh Hưởng của Chunking Strategy
🎯 Mục Tiêu

Kiểm tra xem Semantic Chunking có nâng cao retrieval quality rõ rệt so với Fixed-size Chunking không.

Nếu có: đây là bằng chứng cho giả định (chunking tác động đến retrieval quality).


Thiết kế

Cử định: 1 tài liệu dài (manual kỹ thuật 200 trang PDF). Test với 4 chunking strategy:

Fixed-size: 256 tokens, overlap 20 tokens
Fixed-size: 512 tokens, overlap 50 tokens
Semantic Chunking: dựa trên cosine similarity breakpoints
Recursive Chunking: chia theo cấu trúc văn bản đệ quy

Code Kiểm Tra (Python)

# scenario_2_chunking.py

from langchain.text_splitter import (

    RecursiveCharacterTextSplitter,

    SemanticChunker)

from langchain_openai import OpenAIEmbeddings


configs = {

    "fixed_256": RecursiveCharacterTextSplitter(

        chunk_size=256, chunk_overlap=20),

    "fixed_512": RecursiveCharacterTextSplitter(

        chunk_size=512, chunk_overlap=50),

    "semantic": SemanticChunker(

        OpenAIEmbeddings(),

        breakpoint_threshold_type="percentile"),

}


results = {}

for name, splitter in configs.items():

    chunks = splitter.split_text(document_text)

    scores = evaluate_retrieval(chunks, test_questions)

    results[name] = {"n_chunks": len(chunks), **scores}

    print(f"{name}: {scores}")


Kịch Bản 3: Tối ưu Episodic Memory Retrieval Scoring
🎯 Mục Tiêu

Test công thức scoring của Episodic Memory: Recency + Importance + Relevance.

Tìm trọng số α, β, γ tối ưu cho Retrieval Quality cao nhất.


Công Thức Scoring (từ tài liệu)

Theo tài liệu nghiên cứu, điểm của mỗi memory được tính:

Score(m) = α * Recency(m) + β * Importance(m) + γ * Relevance(m)

với: α + β + γ = 1 và α, β, γ ≥ 0


Grid Search các Tổ Hợp Trọng Số

α (Recency)

β (Importance)

γ (Relevance)

Use Case Phù Hợp

Kết Quả Dự Kiến

0.6

0.2

0.2

Chat bot đội thoại thờờng

Tốt cho nghiên cứu conversational

0.2

0.6

0.2

Hệ thống knowledge base

Tốt cho truy vấn sự kiện quan trọng

0.1

0.1

0.8

QA System chính xác cao

Tốt cho fact retrieval

0.33

0.33

0.34

Balanced (Baseline)

Điểm tham chiếu


Code Grid Search

# scenario_3_scoring.py

import numpy as np

from itertools import product


def memory_score(memory, query_emb, alpha, beta, gamma):

    recency = np.exp(-0.01 * memory["age_hours"])

    importance = memory["importance"] / 10.0  # normalize 1-10

    relevance = cos_sim(memory["embedding"], query_emb)

    return alpha*recency + beta*importance + gamma*relevance


best_config, best_score = None, 0

weights = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

for a, b, g in product(weights, repeat=3):

    if abs(a + b + g - 1.0) > 0.01: continue

    score = evaluate_with_weights(a, b, g, test_memories, queries)

    if score > best_score: best_config, best_score = (a,b,g), score

print(f"Best: alpha={best_config[0]}, beta={best_config[1]}, gamma={best_config[2]}")


Kịch Bản 4: RAG vs Knowledge Graph đối với Câu Hỏi Quan Hệ Phức Tạp
🎯 Mục Tiêu

Xác định khi nào KG retrieval vưửt trội RAG.

Củng cố giả định: retrieval quality (KG traversal) > context size (RAG nhồi nhiều chunk).


Loại Câu Hỏi Test

Tập trung vào câu hỏi mà RAG yếu theo tài liệu nghiên cứu:

Multi-hop: "Dự án nào Alice và Bob cùng tham gia?" (cần duyệt Alice → Dự án → Bob)
Aggregation: "Nhân viên nào tham gia nhiều dự án nhất năm 2025?"
Path reasoning: "Con đường ngắn nhất từ Manager A đến Engineer B qua cấp bậc tổ chức?"
Comparative: "So sánh chi phí giữa dự án X và Y theo từng quý?"

Kết Quả Dự Kiến Được Bằng Chứng của Giả Định

Loại câu hỏi

RAG Score

KG Score

Kết luận

Fact Lookup đơn giản

~0.85

~0.82

Tương đương, RAG đủ dùng

Multi-hop 2 bước

~0.55

~0.88

KG vưửt trội rõ rệt

Multi-hop 3+ bước

~0.3

~0.85

RAG thất bại, KG giữ vững

Aggregation

~0.4

~0.9

KG vưửt trội hoàn toàn


3. Kế Hoạch Triển Khai & Phân Tích Kết Quả

3.1 Thứ Tự Chạy Test
#

Kịch bản

Tool / Thư viện

Mục tiêu kiểm chứng

1

Basic retrieval comparison

sentence-transformers, Anthropic API

Bằng chứng cơ bản cho giả định

2

Chunking strategy

LangChain, OpenAI Embeddings

Ảnh hưởng chunking đến retrieval

3

Episodic scoring weights

NumPy, custom scoring

Tối ưu αβγ cho Episodic Memory

4

RAG vs KG

Neo4j / NetworkX, ChromaDB

Biết khi nào KG tốt hơn RAG


3.2 Phương Pháp Phân Tích Kết Quả
Sau khi chạy tất cả 4 kịch bản, sử dụng phương pháp phân tích sau:

Statistical significance: paired t-test giữa các hệ thống (p < 0.05 = kết quả ý nghĩa)
Effect size: Cohen's d để đo mức độ cải thiện thực sự
Trade-off analysis: vẽ biểu đồ Quality vs Latency vs Cost
Ablation study: tắt từng component để xác định đóng góp

3.3 Tiêu Chí Kết Luận
✅ Giả định Được Xác Nhận Nếu:

1. Hệ thống C (Semantic+4K) vượt qua A (Random+128K) trên tất cả metric (trừ recall)

2. Hệ thống D (KG+4K) đạt Answer Relevance ≥ 0.85 với multi-hop questions

3. Statistical significance p < 0.05 trên ít nhất 3/4 metric

4. Effect size Cohen's d ≥ 0.5 (medium effect) giữa retrieval tốt và retrieval kém


❌ Giả định Cần Xét Lại Nếu:

1. Hệ thống A (Random+128K) cho recall cao bù đắp chất lượng kém

2. LLM "tự tế" được trong cửa sổ lớn vượt trội retrieval tốt

3. Không có sự khác biệt ý nghĩa thống kê giữa các hệ thống


4. Cài Đặt Môi Trường & Dependencies

4.1 Yêu Cầu Hệ Thống
Thành phần

Chi tiết

Python

3.10+ (khuyến nghị 3.11)

API Keys

ANTHROPIC_API_KEY, OPENAI_API_KEY (cho embeddings)

Vector DB

ChromaDB (local) hoặc Pinecone (cloud)

Graph DB

NetworkX (local test) hoặc Neo4j (production)

RAM

Ít nhất 8GB (16GB khi test KG lớn)


4.2 Cài Đặt Nhanh
# requirements.txt

anthropic>=0.25.0

sentence-transformers>=2.6.0

langchain>=0.2.0

langchain-openai>=0.1.0

chromadb>=0.5.0

networkx>=3.2

numpy>=1.26.0

pandas>=2.1.0

matplotlib>=3.8.0

scipy>=1.12.0  # for statistical tests


# Install:

pip install -r requirements.txt


4.3 Cấu Trúc Thư Mục Dự Án
memory_ai_test/

├── data/          # corpus test documents

├── scenarios/

│   ├── scenario_1_basic.py

│   ├── scenario_2_chunking.py

│   ├── scenario_3_scoring.py

│   └── scenario_4_rag_vs_kg.py

├── utils/

│   ├── metrics.py       # Answer Relevance, Precision, Recall, Faithfulness

│   ├── embeddings.py    # Embedding helpers

│   └── visualization.py # Plot results

├── results/          # JSON outputs

└── run_all.py        # chạy tất cả 4 scenarios



11/05/2026

KỊCH BẢN KIỂM THỬ GIẢ ĐỊNH
Giả định cần chứng minh
“Chất lượng câu trả lời giảm mạnh khi số lượng thông tin liên quan cần tổng hợp tăng lên (multi-context synthesis bottleneck).”


1. Mục tiêu test
Kiểm tra:

Khi số lượng fact cần combine tăng lên:

1 → 2 → 4 → 8 → 16

thì:

Accuracy giảm bao nhiêu?
Hallucination tăng bao nhiêu?
Model bỏ sót bao nhiêu facts?
Có điểm gãy (breaking point) ở đâu?

2. Thiết kế benchmark
Biến độc lập (Independent Variable)
Số lượng facts cần tổng hợp
N = [1, 2, 4, 8, 16]



Biến kiểm soát (Control Variables)
Giữ cố định:

Context length
Ví dụ:

100k tokens

để tránh ảnh hưởng bởi context window.


Noise ratio
Ví dụ:

95% irrelevant text

5% relevant facts



Fact complexity
Mỗi fact cùng độ khó:

Customer A bought Product X on Date Y.

Không fact nào khó hơn fact nào.


Prompt format
Giữ nguyên cùng một prompt template:

Analyze all relevant facts and answer the question.



3. Dataset sinh tự động
Step 1: tạo fact graph
Ví dụ tạo 16 facts có liên hệ:

F1: Alice manages Team Alpha

F2: Team Alpha owns Service Billing

F3: Billing depends on DB-01

F4: DB-01 had outage at 10:00

...

F16: Customer impact = Premium users



Step 2: chèn vào noise context
[random irrelevant paragraphs]

F1

[random irrelevant paragraphs]

F2

[random irrelevant paragraphs]

...

Mỗi fact bị phân tán xa nhau.


Step 3: tạo câu hỏi cần synthesis
Level 1 (1 fact)
Who manages Team Alpha?



Level 2 (2 facts)
Which service does Alice indirectly own?

(Cần F1 + F2)


Level 4 (4 facts)
What infrastructure issue affected Alice’s service?

(Cần F1 + F2 + F3 + F4)


Level 8
Describe the full outage chain.



Level 16
Summarize root cause, impacted users, and remediation.



4. Models đem test
Ví dụ:

GPT-5.5
Gemini 3.1 Pro
Claude Opus 4.7
DeepSeek V4

5. Metrics cần log
(1) Fact Recall
retrieved_relevant_facts / required_facts

Ví dụ:

6/8 = 75%



(2) Synthesis Accuracy
Model có kết luận đúng không?

correct_answer = 1/0



(3) Missing Fact Count
Bao nhiêu fact bị bỏ sót.


(4) Wrong Link Count
Nối sai quan hệ giữa facts.

Ví dụ:

DB-02 caused outage

trong khi đúng là:

DB-01



(5) Hallucination Rate
unsupported_claims / total_claims



(6) Latency
TTFT
total response time

6. Quy trình chạy test
For each model:
For N in [1,2,4,8,16]
for n in [1,2,4,8,16]:

    context = build_context(n)

    prompt = build_question(n)


    answer = model.generate(context, prompt)


    score(answer)

Chạy nhiều lần:

30 runs / level

để giảm variance.


7. Kết quả kỳ vọng
Ví dụ:

Facts cần tổng hợp

GPT-5.5

Gemini

Claude

1

96%

99%

94%

2

93%

97%

90%

4

84%

94%

78%

8

71%

88%

60%

16

48%

73%

39%


8. Cách chứng minh giả định đúng
Nếu xuất hiện đường cong:

Accuracy ↓ sharply as N ↑

và đặc biệt:

drop(8→16) > drop(1→2)

=> giảm phi tuyến (superlinear)

→ chứng minh:

Multi-context synthesis là bottleneck thật.


9. Insight hệ thống có thể rút ra
Nếu giả định đúng:

Không nên hỏi một prompt lớn
Thay vì:

Analyze everything.

nên:

Step 1: find relevant facts

Step 2: group facts

Step 3: synthesize per cluster

Step 4: final aggregation



Agent architecture tốt hơn
Nên dùng:

retrieval planner
decomposition
iterative reasoning
evidence graph
thay vì:

brute-force long context