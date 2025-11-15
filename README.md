# RAG Chatbot Terminal Application

## Objective
Terminal-based chatbot that uses Retrieval-Augmented Generation (RAG) to answer questions based on a PDF’s content, using only open-source tools.

## Features
- Extracts text from PDF via **pdfplumber**
- Converts text to embeddings with **Sentence Transformers (MiniLM)**
- Stores embeddings in **FAISS** and retrieves top-k relevant chunks
- Generates concise answers (50–100 words) with **open-source LLMs** (default: GPT-Neo)
- Fully open-source stack

## Why these tools and models
- **pdfplumber:** Reliable open-source PDF text extraction, handles varied layouts well.
- **Sentence Transformers (all-MiniLM-L6-v2):** Small, fast, high-quality semantic embeddings for accurate retrieval.
- **FAISS:** Open-source, high-performance similarity search over dense vectors; scales and is easy to use.
- **Hugging Face Transformers (GPT-Neo / FLAN-T5 / BART):** Open-source LLMs flexible for generation; GPT-Neo is a good default on CPUs/GPUs.

## Installation
```bash
git clone https://github.com/<your-username>/rag-chatbot-terminal.git
cd rag-chatbot-terminal

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
