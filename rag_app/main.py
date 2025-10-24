# Fichier: rag_app/main.py
from flask import Flask, request, jsonify
# Importe les DEUX fonctions de création de pipeline
from rag_pipelines import create_rag_pipeline_v1, create_rag_pipeline_v2_fusion

# --- Initialisation Globale ---
# Nous chargeons les deux pipelines au démarrage
print("Chargement du pipeline RAG V1...")
rag_chain_v1 = create_rag_pipeline_v1()

print("Chargement du pipeline RAG V2 (Fusion)...")
rag_chain_v2 = create_rag_pipeline_v2_fusion()

# --- Création de l'application Flask ---
app = Flask(__name__)

@app.route("/ask_v1", methods=["POST"])
def ask_v1():
    """ Endpoint pour le RAG de base V1 """
    payload = request.json or {}
    question = payload.get("question", "")
    if not question:
        return jsonify({"error": "Aucune question fournie"}), 400

    try:
        # Les deux chaînes attendent maintenant un dictionnaire
        answer = rag_chain_v1.invoke({"question": question})
        return jsonify({"answer": answer})
        
    except Exception as e:
        print(f"Erreur durant la chaîne RAG V1 : {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/ask_v2", methods=["POST"])
def ask_v2():
    """ Endpoint pour le RAG V2 (RAG-Fusion) """
    payload = request.json or {}
    question = payload.get("question", "")
    if not question:
        return jsonify({"error": "Aucune question fournie"}), 400

    try:
        # Les deux chaînes attendent maintenant un dictionnaire
        answer = rag_chain_v2.invoke({"question": question})
        return jsonify({"answer": answer})
        
    except Exception as e:
        print(f"Erreur durant la chaîne RAG V2 : {e}")
        return jsonify({"error": str(e)}), 500

# Le bloc "if __name__ == '__main__':" a été supprimé.
# Gunicorn lancera directement l'objet 'app'.