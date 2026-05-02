import os
import asyncio
import nest_asyncio
import streamlit as st
from dotenv import load_dotenv, find_dotenv

# LangChain & Groq Imports
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from huggingface_hub import hf_hub_download

# Load environment variables
load_dotenv(find_dotenv())
nest_asyncio.apply()

# Sidebar
st.sidebar.markdown("<h2 style='color: #ffffff;'>📌 Description</h2>", unsafe_allow_html=True)
st.sidebar.image("utils/ph2.png", use_container_width=True)
st.sidebar.markdown("<p class='sidebar-text'>The LLM Medical Chatbot is an AI-powered assistant designed to provide instant, accurate, and reliable healthcare insights.</p>", unsafe_allow_html=True)

# Ensure async loop is running
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

# Constants
DB_FAISS_PATH = "vectorstore/db_faiss"
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
HF_REPO_ID = "Riteshkumarverma/medical-vectorstore"

if not GROQ_API_KEY:
    st.error("❌ GROQ_API_KEY is missing. Please set it in your environment.")
    st.stop()

@st.cache_resource
def load_vectorstore():
    try:
        # Download from HuggingFace if not exists locally
        os.makedirs(DB_FAISS_PATH, exist_ok=True)

        faiss_path = os.path.join(DB_FAISS_PATH, "index.faiss")
        pkl_path   = os.path.join(DB_FAISS_PATH, "index.pkl")

        if not os.path.exists(faiss_path):
            with st.spinner("📥 Downloading vectorstore (index.faiss)..."):
                hf_hub_download(
                    repo_id=HF_REPO_ID,
                    filename="vectorstore/db_faiss/index.faiss",
                    repo_type="dataset",
                    local_dir="."
                )

        if not os.path.exists(pkl_path):
            with st.spinner("📥 Downloading vectorstore (index.pkl)..."):
                hf_hub_download(
                    repo_id=HF_REPO_ID,
                    filename="vectorstore/db_faiss/index.pkl",
                    repo_type="dataset",
                    local_dir="."
                )

        embedding_model = HuggingFaceEmbeddings(
            model_name='sentence-transformers/all-MiniLM-L6-v2'
        )
        return FAISS.load_local(
            DB_FAISS_PATH,
            embedding_model,
            allow_dangerous_deserialization=True
        )

    except Exception as e:
        st.error(f"❌ Vectorstore load failed: {e}")
        return None

vectorstore = load_vectorstore()

def get_prompt_template():
    return PromptTemplate(
        template="""You are Medibot, an expert AI medical assistant trained on real medical books and guidelines.
Use the provided context to answer the user's question as accurately and helpfully as possible.
If the answer is not in the context, say "I don't have enough information on this, please consult a doctor."
Never make up medical information.

**Context:**
{context}

**Question:**
{question}

Provide a **clear, concise, and medically accurate response**.
Always remind the user to consult a licensed healthcare professional for diagnosis and treatment.
        """,
        input_variables=["context", "question"]
    )

def load_llm():
    os.environ["GROQ_API_KEY"] = GROQ_API_KEY
    return ChatGroq(
        temperature=0.5,
        model_name="llama-3.3-70b-versatile"
    )

def format_sources(source_documents):
    if not source_documents:
        return ""
    formatted_sources = "\n\n---\n**📚 Sources:**"
    seen = set()
    for doc in source_documents:
        source = doc.metadata.get('source', 'Unknown Source')
        # Show only filename not full path
        source_name = os.path.basename(source)
        if source_name not in seen:
            seen.add(source_name)
            formatted_sources += f"\n🔹 {source_name}"
    return formatted_sources

def main():
    st.title("💬 Medibot - AI Health Assistant")
    st.markdown("""
        **Ask any medical-related question, and I'll provide insights based on reliable information!**  
        🤖🩺 *Powered by Groq (Llama 3.3) + 7 Real Medical Books*
    """)

    with st.sidebar:
        st.markdown("""
        ### 🔍 About Medibot:
        - Uses **Meta-Llama 3.3 70B (Groq)** for answers
        - Trained on **7 real medical books** including:
          - 📘 KD Tripathi Pharmacology
          - 📘 Current Essentials of Medicine
          - 📘 Webster's Medical Dictionary
          - 📘 WHO Model Formulary
          - 📘 COVID-19 Clinical Guide (NIH)
          - 📘 Integrated Management of Childhood Illness (WHO)
          - 📘 Diseases of Ear, Nose & Throat
        - Provides **fast, reliable, book-backed responses**
        """)

        # Clear chat button
        if st.button("🗑️ Clear Chat"):
            st.session_state.messages = []
            st.rerun()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        st.chat_message(message["role"]).markdown(message["content"])

    user_query = st.chat_input("Ask your medical question...")

    if user_query:
        st.chat_message("user").markdown(f"**You:** {user_query}")
        st.session_state.messages.append({"role": "user", "content": user_query})

        with st.spinner("🤖 Medibot is thinking..."):
            try:
                if vectorstore is None:
                    st.error("❌ Error: Vector store failed to load.")
                    return

                qa_chain = RetrievalQA.from_chain_type(
                    llm=load_llm(),
                    chain_type="stuff",
                    retriever=vectorstore.as_retriever(search_kwargs={'k': 5}),
                    return_source_documents=True,
                    chain_type_kwargs={'prompt': get_prompt_template()}
                )

                response = qa_chain.invoke({'query': user_query})
                result  = response.get("result", "⚠️ No response generated.")
                sources = response.get("source_documents", [])

                formatted_response = f"**Medibot:** {result}{format_sources(sources)}"
                st.chat_message("assistant").markdown(formatted_response)
                st.session_state.messages.append({"role": "assistant", "content": formatted_response})

            except Exception as e:
                st.error(f"⚠️ Error: {str(e)}")

if __name__ == "__main__":
    main()
