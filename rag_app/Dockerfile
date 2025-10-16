# Fichier: rag_app/Dockerfile
# Instructions pour construire l'image de notre serveur RAG.

# Utilise une image Python légère comme base.
FROM python:3.11-slim

# Définit le répertoire de travail à l'intérieur du conteneur.
WORKDIR /app

ENV PYTHONPATH /app

# Copie le fichier des dépendances.
COPY requirements.txt .

# Installe les dépendances Python.
# --no-cache-dir est une bonne pratique pour garder les images légères.
RUN pip install --no-cache-dir -r requirements.txt

# Copie tout le code de l'application (main.py, api.py, etc.)
COPY . .

# Expose le port 5000 pour que le service Flask soit accessible.
EXPOSE 6000

# Commande pour lancer le serveur lorsque le conteneur démarre.
# On utilise le point d'entrée main.py.
CMD ["gunicorn", "--bind", "0.0.0.0:6000", "--workers", "1", "main:app"]
