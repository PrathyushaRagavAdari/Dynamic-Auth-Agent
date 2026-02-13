# Week 4 Failure & Risk Analysis

## 🚨 Primary Failure Scenario: "OCR Latency Spike"
**Scenario:**
During a high-traffic audit session, the OCR engine (Tesseract) used to parse complex charts (like the "AAL Requirements Table" in NIST documents) takes too long (>5 seconds) or times out.

**Trigger:**
User asks a question requiring visual extraction, e.g., *"What are the specific authenticator requirements for AAL3 in Table 5-1?"*

## 📉 Impact
- **Product:** The Compliance Officer receives a "Time Out" error or an incomplete answer missing the chart data.
- **Business:** Critical security policies might be misverified, leading to the approval of weak authentication methods (e.g., allowing SMS for AAL3 when hardware is required).

## 🔍 Detection Signals (Monitoring)
We detect this via **`logs/product_metrics.csv`**:
1.  **Latency Flag:** `Latency_Seconds > 3.0`
2.  **Evidence Flag:** `Evidence_Source == "None"` despite a "Success" status (indicates hallucination risk).

## 🛡️ Mitigation Strategy
1.  **Caching (Immediate):** Use `@st.cache_data` to store the parsed vector index. We parse the PDF *once* at startup, not per query.
2.  **Fallback (Code):** If OCR fails, fall back to the text-only captions we manually added to the vector store (Metadata: `source: manual_caption`).
3.  **Human-in-the-Loop:** If confidence is low (<0.7), the UI will flag the result with "⚠️ Verify with Source PDF" instead of a green checkmark.

## 🔄 Post-Mortem Plan (Next Sprint)
- **Action:** Move from local Tesseract OCR to a pre-computed Multi-Modal Vector Store (Pinecone) where images are already indexed.
- **Metric:** Track "OCR Timeout Rate" in the next weekly report.
