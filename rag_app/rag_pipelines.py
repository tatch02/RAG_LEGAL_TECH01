# Fichier: rag_app/rag_pipelines.py
# Ce module contient la logique pure pour construire différentes chaînes RAG.
# Pour chaque nouvelle version, nous ajouterons une nouvelle fonction 'create_rag_pipeline_vX'.
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, PyMuPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# version 2
from langchain.load import dumps, loads
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from operator import itemgetter
#############
# Importation de notre configuration centralisée
# ... (le début du fichier reste le même) ...
# Importation de notre configuration centralisée
import config # <-- AJUSTEMENT : On utilise un import absolu

RAG_PROMPT_TEMPLATE = """Vous êtes un assistant IA spécialisé dans la **Loi de Finances de la République du Cameroun pour l'exercice 2025** (document fourni dans le contexte). Votre mission est de répondre aux questions des utilisateurs (entreprises, particuliers, comptables) qui ne sont pas forcément experts en fiscalité ou en droit.

Utilisez exclusivement les extraits de contexte suivants pour formuler votre réponse. Ne vous basez sur aucune connaissance extérieure.

Instructions impératives :

1.  **Réponse Basée sur le Contexte : Votre réponse doit provenir UNIQUEMENT des informations présentes dans les extraits de `Context` fournis.
2.  [cite_start] Citation des Sources Obligatoire : Pour chaque information que vous donnez, vous **DEVEZ** indiquer la source précise d'où elle provient. Apposez la citation immédiatement après la phrase ou l'information concernée, en utilisant le format ``, où 'x' est le numéro de la source indiqué dans le contexte (par exemple, `[cite: 1459]`). [cite_start]Si une phrase combine des informations de plusieurs sources, citez-les toutes (par exemple, `[cite: 1459, 1460]`). Si une information est répétée dans plusieurs sources, citez la source la plus pertinente ou la première occurrence.
3.  Niveau de Détail : Fournissez une réponse aussi complète et détaillée que le contexte le permet pour répondre à la question. **Évitez les réponses trop courtes** si le contexte contient des informations pertinentes supplémentaires (définitions, conditions, exceptions, taux, procédures, sanctions, etc.). Extrayez tous les détails utiles.
4.  Clarté pour Tous : Expliquez les termes techniques ou les concepts juridiques/fiscaux de manière **simple et compréhensible** pour un non-expert, *tout en restant fidèle au texte source*. Si le contexte donne une définition ou une explication, utilisez-la.
5.  Honnêteté : Si les extraits de contexte fournis ne contiennent **absolument pas** l'information nécessaire pour répondre à la question, dites explicitement que l'information n'est pas disponible dans les documents fournis. Ne spéculez pas et n'inventez pas de réponse.
6.  Langue : Répondez en français.

Question : {question}

Context :
{context}

Réponse :
"""

RAG_PROMPT = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)

def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)


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

    rag_chain = (
        {"context": retriever | format_docs, "question": itemgetter("question")}
        | RAG_PROMPT
        | llm
        | StrOutputParser()
    )

    print("--- Pipeline RAG v1 (Base) Prêt ---")
    return rag_chain

def reciprocal_rank_fusion(results: list[list], k=60):
    """ Fusion RRF qui prend plusieurs listes de documents classés. """
    fused_scores = {}
    for docs in results:
        for rank, doc in enumerate(docs):
            doc_str = dumps(doc)
            if doc_str not in fused_scores:
                fused_scores[doc_str] = 0
            previous_score = fused_scores[doc_str]
            fused_scores[doc_str] += 1 / (rank + k)

    reranked_results = [
        (loads(doc), score)
        for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    ]
    # Extrait uniquement les documents, pas les scores
    return [doc for doc, score in reranked_results]

# --- Pipeline V2 : RAG-Fusion ---

def create_rag_pipeline_v2_fusion():
    """
    Construit et retourne la chaîne RAG V2 (RAG-Fusion).
    Load -> Split -> Embed -> Generate Queries -> Retrieve (x4) -> Fuse -> Generate.
    """
    print("--- Initialisation du Pipeline RAG v2 (RAG-Fusion) ---")

    # 1. Chargement et Découpage (identique à V1)
    print(f"Chargement des documents depuis : {config.DOCS_FOLDER}")
    loader = DirectoryLoader(config.DOCS_FOLDER, glob="**/*.pdf", loader_cls=PyMuPDFLoader, show_progress=False)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP)
    splits = text_splitter.split_documents(docs)
    
    # 2. Vectorstore et Retriever (identique à V1)
    print("Création du vectorstore en mémoire (V2)...")
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    retriever = vectorstore.as_retriever()

    # 3. Définition des chaînes RAG-Fusion (logique du Notebook Part 6)
    
    # Prompt RAG standard (pour la réponse finale)

    # Prompt RAG-Fusion pour générer les requêtes alternatives
    # Note : config.RAG_FUSION_TEMPLATE doit être défini dans config.py
    prompt_rag_fusion = ChatPromptTemplate.from_template(config.RAG_FUSION_TEMPLATE)
    
    # Chaîne de génération de requêtes : (Question -> LLM -> 4 requêtes)
    generate_queries_chain = (
        itemgetter("question")
        | prompt_rag_fusion 
        | llm 
        | StrOutputParser() 
        | (lambda x: x.split("\n")) # Sépare les requêtes en une liste
    )
    
    # Chaîne de recherche fusionnée : (Liste de requêtes -> 4x Recherche -> Fusion RRF)
    retrieval_chain_fusion = generate_queries_chain | retriever.map() | RunnableLambda(reciprocal_rank_fusion)

    # 4. Chaîne RAG finale (V2)
    # (Recherche fusionnée -> Contexte formaté) + Question -> Réponse finale
    rag_chain_v2 = (
        # La V2 attend aussi un dict {"question": "..."}
        {"context": retrieval_chain_fusion | format_docs, 
         "question": itemgetter("question")} 
        | RAG_PROMPT
        | llm
        | StrOutputParser()
    )

    print("--- Pipeline RAG v2 (RAG-Fusion) Prêt ---")
    return rag_chain_v2

