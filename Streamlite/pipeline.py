import os
import torch
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from transformers import pipeline
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document

# Global variable to cache the RAG chain
_RAG_CHAIN = None

def initialize_rag():
    """Sets up the RAG pipeline. Returns the chain."""
    global _RAG_CHAIN
    if _RAG_CHAIN is not None:
        return _RAG_CHAIN

    print("--- Initializing GraphGuard RAG Pipeline ---")

    # 1. Download Data if missing
    pdf_path = "./data/nist_guidelines.pdf"
    if not os.path.exists("./data"):
        os.makedirs("./data")
    if not os.path.exists(pdf_path):
        os.system(f"wget -q -O {pdf_path} https://nvlpubs.nist.gov/nistpubs/SpecialPublications/NIST.SP.800-63b.pdf")

    # 2. Ingest
    loader = PyMuPDFLoader(pdf_path)
    docs = loader.load()
    
    # Add Image Caption (Simulated)
    manual_image_caption = """
    [IMAGE CONTEXT: Table 5-1 Authenticator Assurance Level (AAL) Requirements]
    - AAL1: Requires single-factor authentication.
    - AAL2: Requires two distinct authentication factors (Password + OTP).
    - AAL3: Requires a hardware-based authenticator and cryptographic resistance.
    """
    docs.append(Document(page_content=manual_image_caption, metadata={"source": "aal_table.png"}))

    # 3. Split
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    splits = text_splitter.split_documents(docs)

    # 4. Retrievers
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(splits, embeddings)
    dense_retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    sparse_retriever = BM25Retriever.from_documents(splits)
    sparse_retriever.k = 4
    
    ensemble_retriever = EnsembleRetriever(
        retrievers=[dense_retriever, sparse_retriever],
        weights=[0.6, 0.4]
    )

    # 5. LLM
    device = 0 if torch.cuda.is_available() else -1
    pipe = pipeline(
        "text2text-generation",
        model="google/flan-t5-large",
        max_new_tokens=512,
        model_kwargs={"temperature": 0.1},
        device=device
    )
    llm = HuggingFacePipeline(pipeline=pipe)

    # 6. Chain
    template = """You are a Compliance Auditor. Answer based ONLY on the context.
    CONTEXT: {context}
    QUESTION: {question}
    ANSWER:"""
    prompt = PromptTemplate.from_template(template)

    def format_docs(docs):
        return "\n\n".join(f"[Source: {d.metadata.get('source', 'Doc')}] {d.page_content}" for d in docs)

    _RAG_CHAIN = (
        {"context": ensemble_retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return _RAG_CHAIN

def run_inference(query: str):
    """Entry point for the App to ask questions."""
    chain = initialize_rag()
    try:
        response = chain.invoke(query)
        # Return structured output for the App
        return {
            "answer": response,
            "status": "Success" if "not have enough evidence" not in response else "Refusal"
        }
    except Exception as e:
        return {"answer": f"Error: {str(e)}", "status": "Error"}
