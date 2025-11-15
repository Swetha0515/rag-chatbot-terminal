RAG Chatbot Terminal Application
🎯 Objective
A terminal-based chatbot that uses Retrieval-Augmented Generation (RAG) to answer questions based on the content of a provided PDF document, built entirely with open-source tools.

✨ Features
📄 Extracts text from PDF via pdfplumber

🔎 Converts text into embeddings with Sentence Transformers (MiniLM)

🗂️ Stores embeddings in FAISS and retrieves top‑k relevant chunks

🤖 Generates concise answers (50–100 words) with open-source LLMs (default: GPT‑Neo)

🛠️ Fully open-source stack, easy to extend or swap components

🛠️ Why these tools and models
pdfplumber: Reliable open-source PDF text extraction, handles varied layouts well.

Sentence Transformers (all‑MiniLM‑L6‑v2): Small, fast, high-quality semantic embeddings for accurate retrieval.

FAISS: Open-source, high-performance similarity search over dense vectors; scales easily.

Hugging Face Transformers (GPT‑Neo / FLAN‑T5 / BART): Open-source LLMs flexible for generation; GPT‑Neo is a good default on CPUs/GPUs.
