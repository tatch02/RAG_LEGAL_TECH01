# run_evaluation.py
import os
import requests
import pandas as pd
import giskard
from datetime import datetime
from dotenv import load_dotenv

# giskard LLM client (comme dans le notebook)
from giskard.llm.client.openai import OpenAIClient
import giskard.llm

# giskard.rag helpers
from giskard.rag import evaluate, KnowledgeBase
try:
    from giskard.rag import QATestset
except Exception:
    QATestset = None

# Charger les variables d'environnement
load_dotenv("../.env")

# ============================
# === CONFIGURATION ===
# ============================
RAG_APP_URL = "http://localhost:5002/ask"      # URL de ton agent RAG
AGENT_ID = "RAG_v1_Base"
DATASET_PATH = "datasets/testset.jsonl"        # dataset à évaluer
EVALUATION_LLM = "gpt-4o-mini"                 # modèle juge
DOCUMENTS_FOLDER_PATH = os.path.join("..", "data", "knowledge_base")
REPORTS_FOLDER = "reports"
os.makedirs(REPORTS_FOLDER, exist_ok=True)


def answer_fn(question: str, history=None, timeout=60):
    """
    Fonction wrapper : prend une question (string) et renvoie la réponse de l'agent RAG via l'API HTTP.
    """
    payload = {"question": question}
    if history:
        payload["conversation_history"] = history

    try:
        resp = requests.post(RAG_APP_URL, json=payload, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        return data.get("answer", data.get("response", ""))
    except requests.RequestException as e:
        print(f"⚠️ Erreur appel API pour question '{question}' : {e}")
        return f"ERROR_CALL_API: {e}"


def load_and_normalize_dataset(path: str) -> pd.DataFrame:
    """Lit le fichier JSONL et normalise les colonnes attendues par Giskard."""
    df = pd.read_json(path, lines=True)

    if "answer" in df.columns and "reference_answer" not in df.columns:
        df.rename(columns={"answer": "reference_answer"}, inplace=True)
    if "ground_truth" in df.columns and "reference_answer" not in df.columns:
        df.rename(columns={"ground_truth": "reference_answer"}, inplace=True)

    if "metadata" in df.columns and "reference_context" not in df.columns:
        def _extract_page_content(x):
            if isinstance(x, dict):
                return x.get("page_content", "")
            return ""
        df["reference_context"] = df["metadata"].apply(_extract_page_content)

    if "question" not in df.columns:
        raise ValueError("Le dataset doit contenir une colonne 'question'.")

    return df


def maybe_build_knowledge_base(documents_folder: str):
    """Construit une KnowledgeBase à partir de PDFs si un dossier est fourni."""
    if not documents_folder:
        return None

    try:
        from langchain.document_loaders import DirectoryLoader, PyMuPDFLoader
    except Exception:
        print("⚠️ Installe 'langchain' et 'pymupdf' pour activer la KnowledgeBase.")
        return None

    if not os.path.isdir(documents_folder):
        print(f"⚠️ DOCUMENTS_FOLDER_PATH='{documents_folder}' introuvable.")
        return None

    print(f"Chargement des documents depuis : {documents_folder}")
    loader = DirectoryLoader(
        documents_folder,
        glob="**/*.pdf",
        loader_cls=PyMuPDFLoader,
        show_progress=True
    )
    documents = loader.load()
    df_pages = [{"text": d.page_content, **(d.metadata or {})} for d in documents]

    if not df_pages:
        print("⚠️ Aucun document trouvé pour la KnowledgeBase.")
        return None

    kb = KnowledgeBase(pd.DataFrame(df_pages))
    print(f"KnowledgeBase créée ({len(df_pages)} pages).")
    return kb


def main():
    print(f"--- Lancement de l'évaluation RAG : {AGENT_ID} ---")
    print(f"Chargement du dataset : {DATASET_PATH}")
    df = load_and_normalize_dataset(DATASET_PATH)

    # Configure le juge LLM
    print(f"Configuration du juge LLM (Giskard) : {EVALUATION_LLM}")
    giskard.llm.set_llm_api("openai")
    oc = OpenAIClient(model=EVALUATION_LLM)
    giskard.llm.set_default_client(oc)

    # Tente de charger un QATestset si possible
    testset_obj = None
    if QATestset is not None:
        try:
            testset_obj = QATestset.load(DATASET_PATH)
            print("✅ Testset chargé via QATestset.load()")
        except Exception:
            testset_obj = None

    testset_for_eval = testset_obj if testset_obj else df

    # KnowledgeBase optionnelle
    knowledge_base = maybe_build_knowledge_base(DOCUMENTS_FOLDER_PATH)

    print("Déclenchement de l'évaluation (evaluate)...")
    if knowledge_base:
        report = evaluate(answer_fn, testset=testset_for_eval, knowledge_base=knowledge_base)
    else:
        report = evaluate(answer_fn, testset=testset_for_eval)

    # Sauvegarde du rapport
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(REPORTS_FOLDER, f"giskard_RAG_report_{AGENT_ID}_{timestamp}.html")
    try:
        report.to_html(report_path)
        print(f"✅ Rapport HTML sauvegardé : {report_path}")
    except Exception:
        report.save(os.path.join(REPORTS_FOLDER, f"report_{AGENT_ID}_{timestamp}"))
        print(f"✅ Rapport sauvegardé dans '{REPORTS_FOLDER}'")

    # Export des échecs si dispo
    try:
        failures_df = getattr(report, "failures", None)
        if isinstance(failures_df, pd.DataFrame) and len(failures_df) > 0:
            csv_filename = os.path.join(REPORTS_FOLDER, f"evaluation_failures_{AGENT_ID}_{timestamp}.csv")
            failures_df.to_csv(csv_filename, index=False, encoding="utf-8-sig")
            print(f"✅ Échecs exportés dans : {csv_filename}")
        else:
            print("Aucun échec détecté.")
    except Exception as e:
        print(f"⚠️ Erreur export échecs : {e}")

    print("🎉 Évaluation terminée !")


if __name__ == "__main__":
    main()
