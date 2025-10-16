# rag_app/api.py
from flask import Blueprint, request, jsonify
import time
import os
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# Importation de la configuration et des objets partagés
from . import config
from . import rag_pipeline, llm # On importera les objets créés dans main.py

api_blueprint = Blueprint('api', __name__)

def build_context_for_llm(docs: list) -> str:
    # ... (La même fonction que dans votre app.py) ...
    context_items = []
    for i, doc in enumerate(docs, start=1):
        source_name = os.path.basename(doc.metadata.get("source", "N/A"))
        page_num = doc.metadata.get("page", "N/A")
        page_num_display = page_num + 1 if isinstance(page_num, int) else "N/A"
        context_items.append(f"Source [{i}]: {source_name} (Page {page_num_display})\n---\n{doc.page_content}")
    return "\n\n".join(context_items)

def answer_question(question: str, context: str, docs: list) -> dict:
    # ... (La même fonction que dans votre app.py, utilisant les objets LangChain) ...
    system_prompt = "..." # Le même prompt système
    human_prompt = f"CONTEXTE:\n{context}\n\nQUESTION:\n{question}\n\nRÉPONSE:"
    
    messages = [SystemMessage(content=system_prompt), HumanMessage(content=human_prompt)]
    response = llm.invoke(messages)
    answer = response.content
    sources = [...] # La même logique pour créer les sources
    return {"answer": answer.strip(), "sources": sources}

@api_blueprint.route("/ask", methods=["POST"])
def ask():
    payload = request.json or {}
    question = payload.get("question", "")
    chat_history_raw = payload.get("chat_history", [])
    
    chat_history = [HumanMessage(content=m["content"]) if m["role"] == "user" else AIMessage(content=m["content"]) for m in chat_history_raw]

    start_time = time.time()
    try:
        reranked_docs = rag_pipeline.invoke({"input": question, "chat_history": chat_history})
        
        if not reranked_docs or reranked_docs[0].metadata.get('rerank_score', 0) < config.RERANKING_CONFIDENCE_THRESHOLD:
            # ... (Logique d'abstention) ...
            return jsonify(...)

        context_str = build_context_for_llm(reranked_docs)
        answer_payload = answer_question(question, context_str, reranked_docs)
        
        # ... (Formatage de la réponse finale) ...
        return jsonify(result)
        
    except Exception as e:
        # ... (Gestion d'erreur) ...
        return jsonify({"error": str(e)}), 500