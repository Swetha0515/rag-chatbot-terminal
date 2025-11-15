
"""
Fixed RAG terminal chatbot (improved retrieval + safer generation)

- Default LLM: google/flan-t5-base (stronger than small)
- Embeddings: sentence-transformers/all-MiniLM-L6-v2
- Vector DB: FAISS (faiss-cpu, cosine similarity)
- PDF extraction: pdfplumber

Usage examples:
    source venv/bin/activate
    python3 rag_terminal_chatbot.py /path/to/SWETHA_RESUME.pdf
    python3 rag_terminal_chatbot.py /path/to/SWETHA_RESUME.pdf --llm distilgpt2
"""

import os
import sys
import argparse
import pickle
from typing import List

import pdfplumber
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoConfig,
)

# ---------------- Configuration ----------------
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CACHE_DIR = ".rag_cache"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 200
TOP_K = 10
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_CONTEXT_TOKENS = 800  # token-based truncation

PROMPT_TEMPLATE = """You are a precise assistant. 
Answer ONLY using the CONTEXT below. 
If the answer is not in CONTEXT, say: "No information found in the document."

Keep answers factual, concise, and between 50–100 words.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""

# ------------------------------------------------


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def extract_text_from_pdf(pdf_path: str) -> str:
    all_text = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                all_text.append(text)
    return "\n\n".join(all_text)


def chunk_text(text: str, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP) -> List[str]:
    chunks = []
    start = 0
    L = len(text)
    while start < L:
        end = min(start + chunk_size, L)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == L:
            break
        start = max(0, end - overlap)
    return chunks


def build_embeddings(passages: List[str], model_name: str, basename: str):
    ensure_dir(CACHE_DIR)
    cache_path = os.path.join(CACHE_DIR, f"embeddings_{basename}.pkl")
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "rb") as f:
                data = pickle.load(f)
            if "embeddings" in data:
                return data["embeddings"]
        except Exception:
            pass

    print("[info] Computing embeddings (sentence-transformers)...")
    model = SentenceTransformer(model_name, device="cpu")
    embeddings = model.encode(passages, show_progress_bar=True, convert_to_numpy=True)
    with open(cache_path, "wb") as f:
        pickle.dump({"embeddings": embeddings}, f)
    return embeddings


def build_faiss_index(embeddings: np.ndarray, basename: str):
    ensure_dir(CACHE_DIR)
    index_path = os.path.join(CACHE_DIR, f"faiss_{basename}.index")
    emb_dim = embeddings.shape[1]
    faiss.normalize_L2(embeddings)
    if os.path.exists(index_path):
        try:
            index = faiss.read_index(index_path)
            if index.d != emb_dim:
                raise RuntimeError("FAISS index dimension mismatch; rebuilding.")
            return index
        except Exception:
            print("[info] Existing FAISS index invalid or mismatched. Rebuilding...")
            try:
                os.remove(index_path)
            except Exception:
                pass
    print("[info] Creating FAISS index (IndexFlatL2)...")
    index = faiss.IndexFlatL2(emb_dim)
    index.add(embeddings)
    faiss.write_index(index, index_path)
    return index


def load_llm(model_name: str):
    print(f"[info] Loading model '{model_name}' on {DEVICE} ...")
    config = AutoConfig.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token if hasattr(tokenizer, "eos_token") else tokenizer.convert_tokens_to_ids(["<pad>"])[0]

    if getattr(config, "is_encoder_decoder", False):
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(DEVICE)
        model_type = "seq2seq"
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name).to(DEVICE)
        model_type = "causal"
    model.eval()
    return tokenizer, model, model_type


def truncate_context_by_tokens(tokenizer, context: str, max_tokens: int = MAX_CONTEXT_TOKENS) -> str:
    tokens = tokenizer.encode(context, truncation=True, max_length=max_tokens)
    return tokenizer.decode(tokens, skip_special_tokens=True)


def generate_answer(tokenizer, model, model_type: str, prompt: str, min_words=50, max_words=100):
    if model_type == "seq2seq":
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(DEVICE)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=256,
                num_beams=4,
                early_stopping=True,
                no_repeat_ngram_size=3,
                repetition_penalty=1.2,
            )
        gen = tokenizer.decode(out[0], skip_special_tokens=True).strip()
    else:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(DEVICE)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
                num_beams=4,
                no_repeat_ngram_size=3,
                repetition_penalty=1.2,
                early_stopping=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        gen = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    words = gen.split()
    if len(words) < min_words:
        gen = gen + " (answer too short, please refine question)."
    if len(words) > max_words:
        gen = " ".join(words[:max_words])
        if gen[-1] not in ".!?":
            gen = gen.rstrip(" ,;:") + "..."
    return gen


def prepare_context(passages: List[str], scores: List[float], tokenizer, top_k: int = TOP_K):
    parts = []
    for i, (p, s) in enumerate(zip(passages[:top_k], scores[:top_k])):
        parts.append(f"[Passage {i+1} | score={float(s):.4f}]\n{p}")
    combined = "\n\n".join(parts)
    return truncate_context_by_tokens(tokenizer, combined, MAX_CONTEXT_TOKENS)


def run_rag_interactive(pdf_path: str, llm_model_name: str = "google/flan-t5-base"):
    basename = os.path.splitext(os.path.basename(pdf_path))[0]
    print("[info] Extracting text from PDF...")
    text = extract_text_from_pdf(pdf_path)
    if not text.strip():
        print("[error] No text extracted from the PDF.")
        sys.exit(1)

    print("[info] Chunking document...")
    passages = chunk_text(text)
    print(f"[info] {len(passages)} passages created.")

    embeddings = build_embeddings(passages, EMBED_MODEL, basename)
    index = build_faiss_index(embeddings, basename)

    tokenizer, model, model_type = load_llm(llm_model_name)
    emb_model = SentenceTransformer(EMBED_MODEL, device="cpu")

    print("\n=== RAG chatbot ready ===")
    print("Ask a question related to the document. Type 'exit' to quit.\n")

    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[info] Exiting.")
            break
        if not question:
            continue
        if question.lower() in ("exit", "quit"):
            print("[info] Bye.")
            break

        q_emb = emb_model.encode([question], convert_to_numpy=True)
        faiss.normalize_L2(q_emb)
        D, I = index.search(q_emb, TOP_K)
        scores = D[0].tolist()
        idxs = I[0].tolist()
        retrieved = [passages[i] for i in idxs]

        context = prepare_context(retrieved, scores, tokenizer, TOP_K)
        prompt = PROMPT_TEMPLATE.format(context=context, question=question)

        try:
            answer = generate_answer(tokenizer, model, model_type, prompt)
        except Exception as e:
            answer = f"[error] generation failed: {e}"

        # Sanity check: avoid repetitive nonsense
        if answer and len(answer.split()) > 10:
            first_word = answer.split()[0]
            if answer.count(first_word) > 10:
                answer = "Sorry — I could not produce a concise answer from the document."

        print("\n[Answer]\n", answer)
        print("\n[Retrieved Passages]")
        for rank, (i, s) in enumerate(zip(idxs, scores), start=1):
            print(f"{rank}. passage #{i} (score={float(s):.4f})")
        print("\n---\n")


def main():
    parser = argparse.ArgumentParser(description="RAG terminal chatbot for PDF documents (improved).")
    parser.add_argument("pdf", help="Path to the PDF document")
    parser.add_argument(
        "--llm",
        default="google/flan-t5-base",
        help="Hugging Face model name (default: google/flan-t5-base). Use distilgpt2 or others if needed.",
    )
    args = parser.parse_args()

    pdf_path = args.pdf
    llm_name = args.llm

    if not os.path.exists(pdf_path):
        print(f"[error] PDF not found: {pdf_path}")
        sys.exit(1)

    run_rag_interactive(pdf_path, llm_name)


if __name__ == "__main__":
    main()
