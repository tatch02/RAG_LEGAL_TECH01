"""
title: Agent RAG Fusion V2
author: VotreNom
date: 2025-09-18
version: 2.0
license: MIT
description: Un pipeline qui appelle notre serveur RAG V2 (RAG-Fusion).
requirements: requests
"""
import requests
import os

class Pipeline:
    def __init__(self):
        # --- MODIFICATION CLÉ ---
        # Il appelle le MÊME service docker, mais le NOUVEL endpoint /ask_v2
        self.rag_server_url = "http://rag_server_v1:6000/ask_v2"
        print(f"Pipeline V2 initialisé. Il appellera {self.rag_server_url}")

    def pipe(self, user_message: str, chat_history: list = [], **kwargs) -> str:
        """
        Intercepte le message de l'utilisateur et le transmet au serveur RAG V2.
        """
        print(f"Pipeline V2 a reçu : '{user_message}'")

        payload = {"question": user_message}

        try:
            response = requests.post(self.rag_server_url, json=payload)
            response.raise_for_status() 

            data = response.json()
            answer = data.get("answer", "Le serveur V2 n'a pas renvoyé de réponse.")

            return answer

        except requests.exceptions.RequestException as e:
            error_message = f"Erreur de communication avec le serveur RAG V2 : {e}"
            print(error_message)
            return error_message