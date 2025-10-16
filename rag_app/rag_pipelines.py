# Fichier: rag_app/rag_pipelines.py
# Ce module contient la logique pure pour construire différentes chaînes RAG.
# Pour chaque nouvelle version, nous ajouterons une nouvelle fonction 'create_rag_pipeline_vX'.

from langchain import hub
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, PyMuPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# Importation de notre configuration centralisée
# ... (le début du fichier reste le même) ...
# Importation de notre configuration centralisée
import config # <-- AJUSTEMENT : On utilise un import absolu


# ... (le reste du fichier reste identique) ...

# --- Initialisation des composants lourds (chargés une seule fois au démarrage) ---
llm = ChatOpenAI(model_name=config.LLM_MODEL, temperature=0)
embeddings = OpenAIEmbeddings(model=config.EMBEDDING_MODEL)

def create_rag_pipeline_v1():
    """
    Construit et retourne la chaîne RAG de base (Version 1).
    Cette version est simple : Load -> Split -> Embed -> Retrieve -> Generate.
    """
    print("--- Initialisation du Pipeline RAG v1 (Base) ---")

    # 1. Chargement des documents PDF depuis le dossier spécifié dans la config.
    print(f"Chargement des documents depuis : {config.DOCS_FOLDER}")
    loader = DirectoryLoader(config.DOCS_FOLDER, glob="**/*.pdf", loader_cls=PyMuPDFLoader, show_progress=True)
    docs = loader.load()
    print(f"{len(docs)} pages chargées.")

    # 2. Découpage des documents en morceaux (chunks).
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP)
    splits = text_splitter.split_documents(docs)
    print(f"{len(splits)} morceaux créés.")

    # 3. Création de la base de données vectorielle en mémoire (Chroma).
    # Elle sera reconstruite à chaque redémarrage du conteneur.
    print("Création du vectorstore en mémoire...")
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    retriever = vectorstore.as_retriever()

    # 4. Création de la chaîne avec LangChain Expression Language (LCEL).
    # Utilise un prompt standard du LangChain Hub.
    prompt = hub.pull("rlm/rag-prompt")

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    print("--- Pipeline RAG v1 (Base) Prêt ---")
    return rag_chain