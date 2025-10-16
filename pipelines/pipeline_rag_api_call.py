"""
title: Agent RAG RDUE V1
author: VotreNom
date: 2025-08-27
version: 4.0
license: MIT
description: Un pipeline qui appelle notre serveur RAG V4 en transmettant l'historique de la conversation.
requirements: requests
"""
import requests
import os

class Pipeline:
    def __init__(self):
        # --- MODIFICATION CLÉ ---
        # On utilise le nom du service défini dans docker-compose.yml ('rag_server_v1')
        # et son port INTERNE (toujours 6000, sauf si on l'a changé dans le Dockerfile).
        self.rag_server_url = "http://rag_server_v1:6000/ask"
        print(f"Pipeline V1 initialisé. Il appellera {self.rag_server_url}")

    def pipe(self, user_message: str, chat_history: list = [], **kwargs) -> str:
        """
        Intercepte le message de l'utilisateur et le transmet au serveur RAG.
        La V1 ne gère pas encore l'historique de conversation.
        """
        print(f"Pipeline V1 a reçu : '{user_message}'")

        payload = {"question": user_message}

        try:
            response = requests.post(self.rag_server_url, json=payload)
            response.raise_for_status() # Lève une erreur si le statut est 4xx ou 5xx

            data = response.json()
            answer = data.get("answer", "Le serveur n'a pas renvoyé de réponse.")
            
            # La V1 ne renvoie pas de sources structurées, on retourne juste la réponse.
            return answer

        except requests.exceptions.RequestException as e:
            error_message = f"Erreur de communication avec le serveur RAG : {e}"
            print(error_message)
            return error_message



"""
title: RAG API Caller Pipeline
author: Gemini
date: 2025-06-18
version: 1.0
license: MIT
description: A simple pipeline that calls our dedicated RAG server.
requirements: requests
"""
"""
import requests

class Pipeline:
    def __init__(self):
        # L'URL de notre serveur RAG. On utilise le nom du service docker-compose.
        self.rag_server_url = "http://rag_server:5000/ask_rag"
        print(f"Pipeline 'RAG API Caller' initialisé. Il appellera {self.rag_server_url}")

    def pipe(self, user_message: str, **kwargs) -> str:
        print(f"Pipeline a reçu '{user_message}'. Transfert au serveur RAG.")

        try:
            # On envoie une requête POST avec la question en JSON
            response = requests.post(self.rag_server_url, json={"question": user_message})
            response.raise_for_status() # Lève une erreur si le statut n'est pas 200

            # On extrait la réponse du JSON retourné par le serveur
            answer = response.json().get("answer", "Le serveur n'a pas renvoyé de réponse.")
            return answer

        except requests.exceptions.RequestException as e:
            print(f"Erreur de communication avec le serveur RAG : {e}")
            return "Erreur : Impossible de contacter le serveur RAG."
 """       
        
