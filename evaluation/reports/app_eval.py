import streamlit as st
import pandas as pd

# Charger le dataset
df = pd.read_csv("evaluation_failures_RAG_v1_Base_20250918_162012.csv")

# Affichage stylé
st.title("📊 Aperçu du Dataset")

# Limiter le nombre de lignes pour éviter surcharge
st.dataframe(df.head(84), use_container_width=True)

# Pour voir la taille du dataset
st.write(f"Nombre total de lignes : {len(df)}")
