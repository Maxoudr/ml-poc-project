"""Streamlit app - Prédiction de résultats de matchs de football"""

from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np
import streamlit as st
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from config import MODEL_METRICS_FILE, MODELS_DIR, PLOTS_DIR

BASE_DIR = Path(__file__).resolve().parent.parent

def build_app() -> None:
    st.set_page_config(page_title="Football Match Predictor", layout="wide")

    menu = st.sidebar.selectbox(
        "Navigation",
        ["🏠 Projet", "📊 Modèles", "⚽ Démo"]
    )

    if menu == "🏠 Projet":
        st.title("⚽ Prédiction de résultats de matchs de football")
        
        st.header("Objectif business")
        st.write("""
        Ce projet vise à prédire le résultat d'un match de football international 
        (victoire domicile, nul, victoire extérieur) en utilisant des données historiques.
        
        **Applications :**
        - Aide à la décision pour analystes sportifs
        - Support pour paris sportifs
        - Analyse tactique pour staffs techniques
        """)

        st.header("Dataset")
        st.write("""
        - **Source** : Kaggle - International Football Results
        - **Période** : 1990 - 2024
        - **Matchs** : 32 101 matchs internationaux
        - **Features** : 12 variables dont ranking FIFA, forme récente, head-to-head
        """)

        st.header("Variable cible")
        col1, col2, col3 = st.columns(3)
        col1.metric("Home Win", "48%")
        col2.metric("Away Win", "28%")
        col3.metric("Draw", "23%")

    elif menu == "📊 Modèles":
        st.title("📊 Comparaison des modèles")

        st.header("Métriques")
        st.write("""
        - **Métrique principale** : F1-score macro (traite chaque classe également)
        - **Métrique secondaire** : Accuracy
        """)

        st.header("Résultats")
        results = pd.DataFrame({
            'Modèle': ['Logistic Regression', 'Random Forest', 'Gradient Boosting'],
            'Accuracy': [0.53, 0.55, 0.57],
            'F1 macro': [0.49, 0.49, 0.48]
        })
        st.dataframe(results, use_container_width=True)

        # Graphique comparaison
        fig, ax = plt.subplots(figsize=(8, 4))
        x = range(len(results))
        ax.bar([i - 0.2 for i in x], results['Accuracy'], width=0.4, label='Accuracy', color='steelblue')
        ax.bar([i + 0.2 for i in x], results['F1 macro'], width=0.4, label='F1 macro', color='orange')
        ax.set_xticks(x)
        ax.set_xticklabels(results['Modèle'])
        ax.legend()
        ax.set_title('Comparaison des modèles')
        st.pyplot(fig)

        st.header("Protocole")
        st.write("""
        - Train/Test split : 80% / 20%
        - SMOTE pour rééquilibrer les classes
        - random_state=42 pour la reproductibilité
        """)

    elif menu == "⚽ Démo":
        st.title("⚽ Prédire le résultat d'un match")

        # Charger le meilleur modèle
        model_path = MODELS_DIR / "gradient_boosting.joblib"
        model = joblib.load(model_path)

        st.sidebar.header("Paramètres du match")
        
        home_team = st.sidebar.text_input("Équipe domicile", "France")
        away_team = st.sidebar.text_input("Équipe extérieure", "Brazil")
        home_rank = st.sidebar.slider("Ranking FIFA domicile", 1, 200, 2)
        away_rank = st.sidebar.slider("Ranking FIFA extérieur", 1, 200, 5)
        home_form = st.sidebar.slider("Forme domicile (% victoires)", 0.0, 1.0, 0.6)
        away_form = st.sidebar.slider("Forme extérieur (% victoires)", 0.0, 1.0, 0.5)
        neutral = st.sidebar.checkbox("Terrain neutre", False)
        is_friendly = st.sidebar.checkbox("Match amical", False)

        if st.button("Prédire le résultat"):
            # Créer les features
            X = np.array([[
                0,  # home_team_enc
                1,  # away_team_enc
                0,  # tournament_enc
                int(neutral),
                2024,  # year
                6,  # month
                home_form,
                away_form,
                int(is_friendly),
                0.5,  # h2h
                home_rank,
                away_rank
            ]])

            pred = model.predict(X)[0]
            proba = model.predict_proba(X)[0]

            labels = ['Away Win', 'Draw', 'Home Win']
            result = labels[pred]

            st.success(f"Résultat prédit : **{result}**")

            col1, col2, col3 = st.columns(3)
            col1.metric("Away Win", f"{proba[0]:.1%}")
            col2.metric("Draw", f"{proba[1]:.1%}")
            col3.metric("Home Win", f"{proba[2]:.1%}")


if __name__ == "__main__":
    build_app()