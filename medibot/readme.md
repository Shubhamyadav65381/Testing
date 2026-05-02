# 🩺 Medibot - AI Health Assistant

**Medibot** is an AI-powered chatbot that answers medical questions using a retrieval-augmented generation (RAG) architecture. It combines a powerful language model (Mistral-7B-Instruct) with a FAISS-based vector search to provide contextual, concise, and reliable healthcare insights.

---

## 📌 Description

This application leverages Hugging Face's Mistral-7B language model and LangChain to deliver real-time answers to user queries. It retrieves relevant medical documents from a local vector database and uses a custom prompt template to ensure safe and informative responses.

---

## 🚀 Features

- 💬 Chat interface for natural interaction
- 🧠 Powered by Mistral-7B via Hugging Face Inference API
- 🔍 Retrieves relevant context from FAISS vector store
- ✅ Provides sourced and structured medical information
- 🛡️ Avoids hallucinations with a strict prompt format
- ⚙️ Fast and efficient retrieval with Sentence Transformers

---

## 🛠️ Tech Stack

- **Frontend:** Streamlit
- **Backend:** Python (asyncio, dotenv, os)
- **AI/ML:**  
  - `mistralai/Mistral-7B-Instruct-v0.3` (via Hugging Face)
  - Sentence Transformers: `all-MiniLM-L6-v2`
  - LangChain for chaining and retrieval
- **Vector Store:** FAISS
- **Deployment:** Streamlit-compatible


