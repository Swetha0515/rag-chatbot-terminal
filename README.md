# RAG Chatbot Terminal Application

## 📌 Objective
A terminal-based chatbot that uses Retrieval-Augmented Generation (RAG) to answer user queries based on the content of a provided PDF document.

## ⚙️ Features
- Extracts text from PDF using **pdfplumber**.
- Embeds text with **Sentence Transformers (MiniLM)**.
- Stores embeddings in **FAISS** vector database.
- Retrieves relevant passages via similarity search.
- Generates concise answers (50–100 words) using **open-source LLMs** (default: GPT-Neo).
- Fully open-source stack.

## 🚀 Installation
```bash
git clone https://github.com/Swetha0515/rag-chatbot-terminal.git
cd rag-chatbot-terminal
python3 -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
pip install -r requirements.txt
