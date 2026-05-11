# AI-pdf-chatbot
My first AI PDF chatbot using Streamlit + Ollama

Chat GPT:
# 📄 AI PDF Chatbot

My first AI chatbot project 🚀

A simple RAG-based PDF chatbot built with:
- Streamlit
- LangChain
- FAISS
- Ollama
- Qwen2.5
- bge-m3 embeddings

---

# Features

- Upload PDF files
- Ask questions about documents
- Semantic similarity search
- Streaming AI response
- Simple casual chat mode
- Local LLM using Ollama

---

# RAG Pipeline

```text
PDF
↓
Chunking
↓
Embedding
↓
FAISS Vector DB
↓
Similarity Search
↓
Context Injection
↓
LLM Response
```

---

# Run

## Install requirements

```bash
pip install -r requirements.txt
```

## Run Ollama

```bash
ollama run qwen2.5:7b
```

## Start Streamlit

```bash
streamlit run app.py
```

---

# Tech Stack

- Python
- Streamlit
- LangChain
- FAISS
- Ollama
- Qwen2.5
- bge-m3

---

# Screenshot

(Add screenshot here later 📸)

---

# Notes

This is my first AI/RAG chatbot project.
Built for learning how:
- embeddings work
- semantic search works
- vector databases work
- LLM + RAG architecture works
