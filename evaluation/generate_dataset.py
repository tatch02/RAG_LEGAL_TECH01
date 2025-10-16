# evaluation/generate_dataset.py
import os
import pandas as pd
import giskard
from datetime import datetime
from langchain_community.document_loaders import PyMuPDFLoader, DirectoryLoader
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

# Charge les variables d'environnement (notamment OPENAI_API_KEY) depuis la racine du projet
load_dotenv("../.env")

# =====================================================================================
# --- SECTION DE CONFIGURATION ---
# =====================================================================================
GENERATION_METHOD = 'giskard'  # Choisissez: 'giskard' ou 'langchain'
DOCS_PATH = "../data/knowledge_base/"
NUM_QUESTIONS_TO_GENERATE = 20
GENERATION_LLM = "gpt-3.5-turbo"

LANGCHAIN_PROMPT_TEMPLATE = """
À partir du CONTEXTE ci-dessous, veuillez générer une question pertinente et sa réponse exacte.
La réponse doit se trouver exclusivement dans le texte fourni.

CONTEXTE:
{context}

QUESTION:
RÉPONSE:
"""

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M")
OUTPUT_NAME = f"dataset_{GENERATION_METHOD}_{NUM_QUESTIONS_TO_GENERATE}q_{TIMESTAMP}"
OUTPUT_GISKARD_PATH = f"datasets/{OUTPUT_NAME}.jsonl"
OUTPUT_CSV_PATH = f"datasets/{OUTPUT_NAME}.csv"
# =====================================================================================

def save_outputs(df: pd.DataFrame, giskard_name: str):
    print(f"Sauvegarde du fichier CSV dans : {OUTPUT_CSV_PATH}")
    df.to_csv(OUTPUT_CSV_PATH, index=False)
    print(f"Sauvegarde du dataset Giskard dans : {OUTPUT_GISKARD_PATH}")
    giskard_dataset = giskard.Dataset(df=df, target="ground_truth", name=giskard_name)
    giskard_dataset.save(OUTPUT_GISKARD_PATH)
    print("✅ Sauvegardes terminées.")

def generate_with_langchain(documents):
    print("--- Lancement de la génération avec la méthode LangChain ---")
    llm = ChatOpenAI(model_name=GENERATION_LLM, temperature=0.1)
    prompt = PromptTemplate(template=LANGCHAIN_PROMPT_TEMPLATE, input_variables=["context"])
    chain = LLMChain(llm=llm, prompt=prompt)
    selected_docs = documents[:NUM_QUESTIONS_TO_GENERATE]
    print(f"Génération de {len(selected_docs)} paires Q&R...")
    results = chain.batch([{"context": doc.page_content} for doc in selected_docs])
    questions, answers = [], []
    for result in results:
        try:
            parts = result['text'].strip().split("QUESTION:")
            qa_part = parts[1].split("RÉPONSE:")
            questions.append(qa_part[0].strip())
            answers.append(qa_part[1].strip())
        except IndexError:
            continue
    df = pd.DataFrame({"question": questions, "ground_truth": answers})
    save_outputs(df, giskard_name=f"RAG QA - LangChain - {TIMESTAMP}")

def generate_with_giskard(documents):
    print("--- Lancement de la génération avec la méthode Giskard ---")
    df_knowledge = pd.DataFrame([doc.dict() for doc in documents])
    knowledge_base = giskard.KnowledgeBase(df_knowledge)
    print(f"Génération de {NUM_QUESTIONS_TO_GENERATE} questions via Giskard...")
    testset = giskard.rag.generate_testset(
        knowledge_base=knowledge_base,
        num_questions=NUM_QUESTIONS_TO_GENERATE,
        language='fr',
        llm=GENERATION_LLM,
        agent_description="Un assistant IA qui répond à des questions sur des documents."
    )
    df = testset.to_pandas()
    save_outputs(df, giskard_name=f"RAG QA - Giskard - {TIMESTAMP}")

def main():
    print("--- Démarrage du script de génération de dataset ---")
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY est manquante. Veuillez la définir dans votre fichier .env")
    loader = DirectoryLoader(DOCS_PATH, glob="**/*.pdf", loader_cls=PyMuPDFLoader)
    documents = loader.load()
    if not documents:
        raise FileNotFoundError(f"Aucun document trouvé dans {DOCS_PATH}")
    if GENERATION_METHOD == 'langchain':
        generate_with_langchain(documents)
    elif GENERATION_METHOD == 'giskard':
        generate_with_giskard(documents)
    else:
        raise ValueError(f"Méthode '{GENERATION_METHOD}' non reconnue.")
    print("--- Script terminé ---")

if __name__ == "__main__":
    main()