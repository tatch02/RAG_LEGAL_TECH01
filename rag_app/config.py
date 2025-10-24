# Fichier: rag_app/config.py
# Ce module charge et expose toutes les variables de configuration
# depuis le fichier .env, en fournissant des valeurs par défaut.

import os
from dotenv import load_dotenv

# Charge les variables d'environnement depuis le fichier .env situé à la racine du projet
load_dotenv()

# --- Configuration LangSmith (pour le suivi des expériences) ---
os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGCHAIN_TRACING_V2", "true")
os.environ["LANGCHAIN_ENDPOINT"] = os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT", "Default Project")

# --- Clé API OpenAI ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("La variable d'environnement OPENAI_API_KEY est manquante dans votre fichier .env")

# --- Configuration des Modèles ---
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-3.5-turbo")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

# --- Configuration du RAG ---
# Chemin vers les documents à l'intérieur du conteneur Docker
DOCS_FOLDER = "/app/data/knowledge_base"
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 1000))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 200))
# --- Configuration RAG-Fusion (V2) ---
RAG_FUSION_TEMPLATE = """Vous êtes un assistant IA serviable. Votre tâche est de générer quatre
requêtes de recherche différentes basées sur une seule requête d'entrée.

Générez plusieurs requêtes de recherche liées à : {question}

Sortie (4 requêtes, séparées par des sauts de ligne) :"""