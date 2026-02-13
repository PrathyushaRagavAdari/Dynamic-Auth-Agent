# Week 4 Deployment Readiness — Architecture Notes

## 🚀 Deployment Target
- **Hosting:** Streamlit Community Cloud (Free Tier)
- **Repo:** [https://github.com/PrathyushaRagavAdari/Dynamic-Auth-Agent](https://github.com/PrathyushaRagavAdari/Dynamic-Auth-Agent)
- **Live URL:** (Pending final push)

## 🏗️ System Architecture
1.  **Frontend (UI):** Streamlit (`app/main.py`)
    - Accepts user queries (Compliance Checks).
    - Displays "Evidence Packs" (PDF Citations).
    - Sidebar for Real-time Metrics.

2.  **Core Logic (Middleware):** Python (`src/`)
    - **Ingestion:** Fetches NIST 800-63B PDF.
    - **RAG Engine:** LlamaIndex + FAISS (Vector Store).
    - **PII Masking:** Local SHA-256 hashing before any LLM call.

3.  **Data Layer:**
    - **Vector Store:** Local FAISS index (for demo) / Pinecone (for production).
    - **Logs:** `logs/product_metrics.csv` (Persisted to GitHub or S3 in prod).
    - **Source of Truth:** Synthetic transaction data (Snowflake).

## 🔄 Data Flow
1.  **User** submits query via Streamlit.
2.  **App** hashes PII (if present) -> Sends to **RAG Engine**.
3.  **RAG Engine** retrieves top-3 chunks from NIST PDF.
4.  **LLM** (GPT-4o) generates "Pass/Fail" verdict.
5.  **App** logs result to CSV -> Updates Sidebar Metrics.

## 🛡️ Governance & Guardrails
- **Refusal Logic:** System rejects non-banking queries (e.g., "cake recipes") using keyword filtering and semantic routing.
- **Privacy:** No raw Account Numbers are ever sent to OpenAI; only hashed IDs.
- **Audit Trail:** Every single interaction is logged with a timestamp for regulatory review.

## 📈 Scaling Considerations
- **Current Bottleneck:** Local PDF parsing (OCR) is slow on CPU.
- **Production Fix:** Pre-process PDFs into embeddings *once* and store in Pinecone to reduce query latency from ~2s to ~200ms.
