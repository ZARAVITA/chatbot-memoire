"""
Chatbot de Recherche Académique - Assistant IA pour Mémoire de Fin d'Études
==============================================================================
Projet : Approches intelligentes et mathématiques pour analyser le churn client
Auteur : Master Modélisation Mathématique et Science de Données
Technologies : LangChain, FAISS, Hugging Face, Streamlit
"""

import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain_community.vectorstores import FAISS
#from langchain.chains import RetrievalQA
#from langchain_community.chains import RetrievalQA
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.prompts import PromptTemplate

# Chargement des variables d'environnement
load_dotenv()

# Configuration de la page Streamlit
st.set_page_config(
    page_title="🎓 Assistant IA - Mémoire de Recherche",
    page_icon="📚",
    layout="wide"
)

# ==================== CONFIGURATION ====================

# Chemins des fichiers
PDF_PATH = "data/memoire.pdf"
VECTOR_STORE_PATH = "vector_store/faiss_index"

# Token Hugging Face (depuis .env)
HF_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")

# Modèle LLM et Embeddings
#LLM_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
LLM_MODEL = "meta-llama/Llama-3.1-8B"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Paramètres de découpage du texte
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


# ==================== FONCTIONS UTILITAIRES ====================

@st.cache_resource
def load_embeddings():
    """Charge le modèle d'embeddings."""
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )


@st.cache_resource
def load_llm():
    """Charge le modèle LLM depuis Hugging Face."""
    if not HF_TOKEN:
        st.error("❌ Token Hugging Face manquant. Vérifiez votre fichier .env")
        st.stop()
    
    return HuggingFaceEndpoint(
        repo_id=LLM_MODEL,
        huggingfacehub_api_token=HF_TOKEN,
        temperature=0.3,
        max_new_tokens=512,
        top_k=40,
        top_p=0.95
    )


@st.cache_resource
def load_or_create_vector_store(_embeddings):
    """
    Charge la base vectorielle FAISS existante ou la crée depuis le PDF.
    Le préfixe _ dans _embeddings évite le hashing par Streamlit.
    """
    # Vérification de l'existence du PDF
    if not os.path.exists(PDF_PATH):
        st.error(f"❌ Fichier PDF introuvable : {PDF_PATH}")
        st.info("📝 Placez votre mémoire dans le dossier `data/` avec le nom `memoire.pdf`")
        st.stop()
    
    # Chargement de la base vectorielle existante
    if os.path.exists(VECTOR_STORE_PATH):
        st.info("📂 Chargement de la base vectorielle existante...")
        return FAISS.load_local(
            VECTOR_STORE_PATH, 
            _embeddings, 
            allow_dangerous_deserialization=True
        )
    
    # Création de la base vectorielle depuis le PDF
    st.info("🔄 Première exécution : création de la base vectorielle...")
    
    # Chargement du PDF
    loader = PyPDFLoader(PDF_PATH)
    documents = loader.load()
    
    # Découpage en chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len
    )
    chunks = text_splitter.split_documents(documents)
    
    st.info(f"📄 Document découpé en {len(chunks)} segments")
    
    # Création des embeddings et de la base FAISS
    vector_store = FAISS.from_documents(chunks, _embeddings)
    
    # Sauvegarde locale
    os.makedirs(os.path.dirname(VECTOR_STORE_PATH), exist_ok=True)
    vector_store.save_local(VECTOR_STORE_PATH)
    
    st.success("✅ Base vectorielle créée et sauvegardée !")
    
    return vector_store


def create_qa_chain(llm, vector_store):
    """Crée la chaîne de question-réponse avec RetrievalQA."""
    
    # Template de prompt personnalisé
    prompt_template = """Tu es un assistant académique expert spécialisé dans l'analyse du churn client en e-commerce.
Tu réponds à des questions sur un mémoire de recherche en Data Science.

Contexte extrait du mémoire :
{context}

Question : {question}

Instructions :
- Réponds de manière précise et structurée
- Cite les méthodes, modèles ou résultats mentionnés dans le contexte
- Si l'information n'est pas dans le contexte, indique-le clairement
- Utilise un ton académique mais accessible

Réponse :"""

    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )
    
    # Création de la chaîne RetrievalQA
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 4}
        ),
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    
    return qa_chain


# ==================== INTERFACE STREAMLIT ====================

def main():
    """Fonction principale de l'application."""
    
    # En-tête
    st.title("🎓 Assistant IA - Mémoire de Recherche")
    st.markdown("""
    ### 📚 Approches intelligentes et mathématiques pour analyser le churn client dans le e-commerce
    
    Posez vos questions sur :
    - Les modèles de survie (CoxPH, Weibull AFT, DeepSurv, DeepHit)
    - Les méthodes d'analyse du churn
    - Les résultats et conclusions du mémoire
    - Les techniques de feature engineering
    """)
    
    st.divider()
    
    # Chargement des ressources
    with st.spinner("⚙️ Initialisation du système..."):
        embeddings = load_embeddings()
        llm = load_llm()
        vector_store = load_or_create_vector_store(embeddings)
        qa_chain = create_qa_chain(llm, vector_store)
    
    st.success("✅ Système prêt !")
    
    # Zone de saisie de la question
    st.subheader("💬 Posez votre question")
    
    question = st.text_area(
        "Question :",
        placeholder="Exemple : Quels sont les principaux modèles de survie utilisés dans ce mémoire ?",
        height=100
    )
    
    # Bouton de soumission
    if st.button("🔍 Rechercher la réponse", type="primary"):
        if question.strip():
            with st.spinner("🤔 Analyse en cours..."):
                try:
                    # Exécution de la requête
                    result = qa_chain.invoke({"query": question})
                    
                    # Affichage de la réponse
                    st.subheader("💡 Réponse :")
                    st.markdown(result["result"])
                    
                    # Affichage des sources (optionnel)
                    with st.expander("📄 Sources extraites du mémoire"):
                        for i, doc in enumerate(result["source_documents"], 1):
                            st.markdown(f"**Extrait {i} (page {doc.metadata.get('page', 'N/A')}) :**")
                            st.text(doc.page_content[:500] + "...")
                            st.divider()
                
                except Exception as e:
                    st.error(f"❌ Erreur lors de la recherche : {str(e)}")
        else:
            st.warning("⚠️ Veuillez saisir une question.")
    
    # Footer
    st.divider()
    st.markdown("""
    <div style='text-align: center; color: gray; font-size: 0.9em;'>
        <p>🚀 Projet développé avec LangChain, FAISS et Streamlit</p>
        <p>📖 Master Modélisation Mathématique et Science de Données</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
