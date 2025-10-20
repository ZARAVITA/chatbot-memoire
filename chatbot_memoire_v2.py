"""
Version améliorée avec historique de conversation
"""

import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate

load_dotenv()

st.set_page_config(
    page_title="🎓 Assistant IA - Mémoire",
    page_icon="📚",
    layout="wide"
)

# Configuration
PDF_PATH = "data/memoire.pdf"
VECTOR_STORE_PATH = "vector_store/faiss_index"
HF_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")
LLM_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

@st.cache_resource
def load_llm():
    if not HF_TOKEN:
        st.error("❌ Token Hugging Face manquant")
        st.stop()
    return HuggingFaceEndpoint(
        repo_id=LLM_MODEL,
        huggingfacehub_api_token=HF_TOKEN,
        temperature=0.3,
        max_new_tokens=512
    )

@st.cache_resource
def load_or_create_vector_store(_embeddings):
    if not os.path.exists(PDF_PATH):
        st.error(f"❌ PDF introuvable : {PDF_PATH}")
        st.stop()
    
    if os.path.exists(VECTOR_STORE_PATH):
        return FAISS.load_local(
            VECTOR_STORE_PATH, 
            _embeddings, 
            allow_dangerous_deserialization=True
        )
    
    loader = PyPDFLoader(PDF_PATH)
    documents = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_documents(documents)
    
    vector_store = FAISS.from_documents(chunks, _embeddings)
    os.makedirs(os.path.dirname(VECTOR_STORE_PATH), exist_ok=True)
    vector_store.save_local(VECTOR_STORE_PATH)
    
    return vector_store

def create_conversational_chain(llm, vector_store):
    """Chaîne avec mémoire conversationnelle."""
    
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )
    
    prompt_template = """Tu es un assistant académique expert. Utilise l'historique de conversation et le contexte pour répondre.

Contexte : {context}

Historique : {chat_history}

Question : {question}

Réponse détaillée :"""

    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "chat_history", "question"]
    )
    
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(search_kwargs={"k": 4}),
        memory=memory,
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": PROMPT}
    )
    
    return chain

def main():
    st.title("🎓 Assistant IA - Mémoire de Recherche")
    st.markdown("### 📚 Chat intelligent avec historique")
    
    # Initialisation de l'historique
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Chargement des ressources
    with st.spinner("⚙️ Initialisation..."):
        embeddings = load_embeddings()
        llm = load_llm()
        vector_store = load_or_create_vector_store(embeddings)
        
        if "chain" not in st.session_state:
            st.session_state.chain = create_conversational_chain(llm, vector_store)
    
    # Affichage de l'historique
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Input utilisateur
    if prompt := st.chat_input("Posez votre question..."):
        # Afficher la question
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Générer la réponse
        with st.chat_message("assistant"):
            with st.spinner("🤔 Réflexion..."):
                try:
                    result = st.session_state.chain({"question": prompt})
                    response = result["answer"]
                    
                    st.markdown(response)
                    
                    # Afficher les sources
                    with st.expander("📄 Sources"):
                        for i, doc in enumerate(result["source_documents"], 1):
                            st.caption(f"Source {i} (page {doc.metadata.get('page', 'N/A')})")
                            st.text(doc.page_content[:300] + "...")
                    
                    st.session_state.messages.append({"role": "assistant", "content": response})
                
                except Exception as e:
                    st.error(f"❌ Erreur : {str(e)}")
    
    # Bouton reset
    if st.sidebar.button("🔄 Réinitialiser la conversation"):
        st.session_state.messages = []
        st.session_state.pop("chain", None)
        st.rerun()
    
    # Statistiques
    st.sidebar.markdown("### 📊 Statistiques")
    st.sidebar.metric("Messages", len(st.session_state.messages))

if __name__ == "__main__":
    main()
