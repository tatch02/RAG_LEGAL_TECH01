# Fichier: rag_app/main.py
from flask import Flask, request, jsonify
from rag_pipelines import create_rag_pipeline_v1

# --- Initialisation Globale ---
rag_chain = create_rag_pipeline_v1()

# --- Création de l'application Flask ---
app = Flask(__name__)

@app.route("/ask", methods=["POST"])
def ask():
    payload = request.json or {}
    question = payload.get("question", "")
    if not question:
        return jsonify({"error": "Aucune question fournie"}), 400

    try:
        answer = rag_chain.invoke(question)
        return jsonify({"answer": answer})
        
    except Exception as e:
        print(f"Erreur durant la chaîne RAG : {e}")
        return jsonify({"error": str(e)}), 500

# Le bloc "if __name__ == '__main__':" a été supprimé.
# Gunicorn lancera directement l'objet 'app'.