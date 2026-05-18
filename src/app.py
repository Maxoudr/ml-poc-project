"""Streamlit app - Prédiction de résultats de matchs de football"""

from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np
import streamlit as st
import joblib
import plotly.graph_objects as go

from config import MODELS_DIR,PLOTS_DIR

# Équipes avec drapeaux et rankings FIFA réels
TEAMS = {
    "France": {"flag": "🇫🇷", "rank": 2},
    "Spain": {"flag": "🇪🇸", "rank": 6},
    "England": {"flag": "🏴󠁧󠁢󠁥󠁮󠁧󠁿", "rank": 5},
    "Germany": {"flag": "🇩🇪", "rank": 12},
    "Brazil": {"flag": "🇧🇷", "rank": 5},
    "Argentina": {"flag": "🇦🇷", "rank": 1},
    "Portugal": {"flag": "🇵🇹", "rank": 6},
    "Italy": {"flag": "🇮🇹", "rank": 9},
    "Netherlands": {"flag": "🇳🇱", "rank": 7},
    "Belgium": {"flag": "🇧🇪", "rank": 3},
    "Croatia": {"flag": "🇭🇷", "rank": 10},
    "Uruguay": {"flag": "🇺🇾", "rank": 16},
    "Morocco": {"flag": "🇲🇦", "rank": 14},
    "Senegal": {"flag": "🇸🇳", "rank": 18},
    "Japan": {"flag": "🇯🇵", "rank": 17},
    "South Korea": {"flag": "🇰🇷", "rank": 22},
    "United States": {"flag": "🇺🇸", "rank": 13},
    "Mexico": {"flag": "🇲🇽", "rank": 15},
    "Colombia": {"flag": "🇨🇴", "rank": 19},
    "Denmark": {"flag": "🇩🇰", "rank": 21},
    "Switzerland": {"flag": "🇨🇭", "rank": 20},
    "Poland": {"flag": "🇵🇱", "rank": 26},
    "Sweden": {"flag": "🇸🇪", "rank": 23},
    "Australia": {"flag": "🇦🇺", "rank": 27},
    "Nigeria": {"flag": "🇳🇬", "rank": 39},
    "Ghana": {"flag": "🇬🇭", "rank": 60},
    "Cameroon": {"flag": "🇨🇲", "rank": 42},
    "Algeria": {"flag": "🇩🇿", "rank": 35},
    "Tunisia": {"flag": "🇹🇳", "rank": 32},
    "Egypt": {"flag": "🇪🇬", "rank": 34},
    "Turkey": {"flag": "🇹🇷", "rank": 42},
    "Iran": {"flag": "🇮🇷", "rank": 22},
    "Saudi Arabia": {"flag": "🇸🇦", "rank": 56},
    "San Marino": {"flag": "🇸🇲", "rank": 210},
}

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@400;600;700&family=Roboto:wght@300;400;700&display=swap');

* { font-family: 'Roboto', sans-serif; }

/* Background */
.stApp {
    background: linear-gradient(135deg, #0a0a0a 0%, #0d1b0e 50%, #0a0a0a 100%);
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1b0e 0%, #0a1a0b 100%);
    border-right: 2px solid #00ff44;
}

/* Titre principal */
.main-title {
    font-family: 'Rajdhani', sans-serif;
    font-size: 3.5em;
    font-weight: 700;
    text-align: center;
    background: linear-gradient(90deg, #00ff44, #00cc33, #ffffff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-shadow: none;
    margin-bottom: 5px;
    letter-spacing: 2px;
}

.subtitle {
    text-align: center;
    color: #888;
    font-size: 1.1em;
    margin-bottom: 30px;
    letter-spacing: 1px;
}

/* Cards équipes */
.team-card {
    background: linear-gradient(135deg, #0d2b0f, #1a3a1c);
    border: 2px solid #00ff44;
    border-radius: 20px;
    padding: 25px 15px;
    text-align: center;
    box-shadow: 0 0 20px rgba(0, 255, 68, 0.2);
    transition: all 0.3s ease;
    margin: 10px 0;
}

.team-card:hover {
    box-shadow: 0 0 40px rgba(0, 255, 68, 0.4);
    transform: translateY(-3px);
}

.team-flag {
    font-size: 4em;
    display: block;
    margin-bottom: 10px;
}

.team-name {
    font-family: 'Rajdhani', sans-serif;
    font-size: 1.8em;
    font-weight: 700;
    color: white;
    letter-spacing: 1px;
}

.team-rank {
    color: #00ff44;
    font-size: 0.9em;
    margin-top: 5px;
}

/* VS */
.vs-container {
    display: flex;
    align-items: center;
    justify-content: center;
    height: 100%;
}

.vs-text {
    font-family: 'Rajdhani', sans-serif;
    font-size: 4em;
    font-weight: 700;
    color: #00ff44;
    text-shadow: 0 0 20px rgba(0, 255, 68, 0.5);
    animation: pulse 2s infinite;
}

@keyframes pulse {
    0% { text-shadow: 0 0 20px rgba(0, 255, 68, 0.5); }
    50% { text-shadow: 0 0 40px rgba(0, 255, 68, 1); }
    100% { text-shadow: 0 0 20px rgba(0, 255, 68, 0.5); }
}

/* Bouton prédire */
.stButton > button {
    background: linear-gradient(90deg, #00ff44, #00cc33) !important;
    color: black !important;
    font-family: 'Rajdhani', sans-serif !important;
    font-size: 1.3em !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 50px !important;
    padding: 15px 40px !important;
    letter-spacing: 2px !important;
    box-shadow: 0 0 30px rgba(0, 255, 68, 0.4) !important;
    transition: all 0.3s ease !important;
    width: 100% !important;
}

.stButton > button:hover {
    background: linear-gradient(90deg, #00cc33, #009922) !important;
    box-shadow: 0 0 50px rgba(0, 255, 68, 0.7) !important;
    transform: scale(1.02) !important;
}

/* Résultat */
.result-win {
    background: linear-gradient(135deg, #0d2b0f, #1a4a1c);
    border: 2px solid #00ff44;
    border-radius: 20px;
    padding: 30px;
    text-align: center;
    box-shadow: 0 0 40px rgba(0, 255, 68, 0.3);
    margin: 20px 0;
}

.result-draw {
    background: linear-gradient(135deg, #2b2b0d, #4a4a1c);
    border: 2px solid #ffaa00;
    border-radius: 20px;
    padding: 30px;
    text-align: center;
    box-shadow: 0 0 40px rgba(255, 170, 0, 0.3);
    margin: 20px 0;
}

.result-away {
    background: linear-gradient(135deg, #2b0d0d, #4a1c1c);
    border: 2px solid #ff4444;
    border-radius: 20px;
    padding: 30px;
    text-align: center;
    box-shadow: 0 0 40px rgba(255, 68, 68, 0.3);
    margin: 20px 0;
}

.result-text {
    font-family: 'Rajdhani', sans-serif;
    font-size: 2.5em;
    font-weight: 700;
    letter-spacing: 2px;
}

/* Metric cards */
.metric-box {
    background: linear-gradient(135deg, #0d1b0e, #1a2a1c);
    border: 1px solid #00ff4422;
    border-radius: 15px;
    padding: 20px;
    text-align: center;
    margin: 5px;
}

.metric-value {
    font-family: 'Rajdhani', sans-serif;
    font-size: 2.5em;
    font-weight: 700;
}

.metric-label {
    color: #888;
    font-size: 0.85em;
    text-transform: uppercase;
    letter-spacing: 1px;
}

/* Pitch divider */
.pitch-divider {
    text-align: center;
    font-size: 3em;
    margin: 20px 0;
    filter: drop-shadow(0 0 10px rgba(0,255,68,0.5));
}

/* Stats bar */
.stat-bar-container {
    background: #1a1a1a;
    border-radius: 10px;
    padding: 15px 20px;
    margin: 8px 0;
    display: flex;
    justify-content: space-between;
    align-items: center;
    border: 1px solid #222;
}

/* Navigation sidebar */
[data-testid="stSidebar"] .stRadio label {
    color: white !important;
    font-size: 1.1em !important;
}

/* Scrollbar */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #0a0a0a; }
::-webkit-scrollbar-thumb { background: #00ff44; border-radius: 3px; }
</style>
"""

def build_app() -> None:
    st.set_page_config(
        page_title="⚽ Football Predictor",
        page_icon="⚽",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.markdown(CSS, unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.markdown("""
        <div style='text-align:center; padding: 20px 0;'>
            <span style='font-size:3em'>⚽</span>
            <h2 style='color:#00ff44; font-family:Rajdhani; letter-spacing:2px; margin:0'>FOOTBALL</h2>
            <p style='color:#888; margin:0'>PREDICTOR</p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("---")
        
        menu = st.sidebar.radio(
    "",
    ["🏠  Projet", "📊  Modèles", "⚽  Démo", "💰 Simulation  Paris"],
        )
        
        st.markdown("---")
        st.markdown("""
        <div style='text-align:center; color:#444; font-size:0.8em; padding:10px'>
            <p>Machine Learning Project</p>
            <p style='color:#00ff44'>v1.0.0</p>
        </div>
        """, unsafe_allow_html=True)

    # ==================== PAGE PROJET ====================
    if "Projet" in menu:
        st.markdown('<div class="main-title">⚽ FOOTBALL MATCH PREDICTOR</div>', unsafe_allow_html=True)
        st.markdown('<div class="subtitle">Prédiction de résultats de matchs internationaux par Machine Learning</div>', unsafe_allow_html=True)
        st.markdown('<div class="pitch-divider">🏟️</div>', unsafe_allow_html=True)

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown('<div class="metric-box"><div class="metric-value" style="color:#00ff44">32K</div><div class="metric-label">Matchs analysés</div></div>', unsafe_allow_html=True)
        with col2:
            st.markdown('<div class="metric-box"><div class="metric-value" style="color:#00aaff">12</div><div class="metric-label">Features</div></div>', unsafe_allow_html=True)
        with col3:
            st.markdown('<div class="metric-box"><div class="metric-value" style="color:#ffaa00">3</div><div class="metric-label">Modèles ML</div></div>', unsafe_allow_html=True)
        with col4:
            st.markdown('<div class="metric-box"><div class="metric-value" style="color:#ff4444">57%</div><div class="metric-label">Accuracy</div></div>', unsafe_allow_html=True)

        st.markdown("---")
        
        col1, col2 = st.columns([3, 2])
        with col1:
            st.markdown("### 🎯 Objectif Business")
            st.markdown("""
            Ce projet prédit le résultat d'un match de football international 
            en utilisant le Machine Learning sur des données historiques depuis 1990.
            
            **Applications concrètes :**
            - 📈 Aide à la décision pour **analystes sportifs**
            - 🎰 Support pour **paris sportifs**
            - 🏋️ Analyse tactique pour **staffs techniques**
            - 📺 Contenu prédictif pour **médias sportifs**
            """)

        with col2:
            st.markdown("### 📊 Dataset")
            st.markdown("""
            | Caractéristique | Valeur |
            |---|---|
            | Source | Kaggle |
            | Période | 1990 - 2024 |
            | Matchs | 32 101 |
            | Pays | 200+ |
            | Features | 12 |
            """)

        st.markdown("---")
        st.markdown("### 🎯 Distribution de la variable cible")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown('<div class="metric-box"><span style="font-size:2em">🏠</span><div class="metric-value" style="color:#00ff44">48%</div><div class="metric-label">Home Win</div></div>', unsafe_allow_html=True)
        with col2:
            st.markdown('<div class="metric-box"><span style="font-size:2em">🤝</span><div class="metric-value" style="color:#ffaa00">23%</div><div class="metric-label">Draw</div></div>', unsafe_allow_html=True)
        with col3:
            st.markdown('<div class="metric-box"><span style="font-size:2em">✈️</span><div class="metric-value" style="color:#ff4444">28%</div><div class="metric-label">Away Win</div></div>', unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("### 🔑 Features utilisées")
        
        features_data = {
            "Feature": ["🏆 home_fifa_rank", "🏆 away_fifa_rank", "📈 home_form", "📈 away_form",
                       "⚔️ h2h", "🏟️ tournament", "⚖️ neutral", "🤝 is_friendly",
                       "👕 home_team", "👕 away_team", "📅 year", "📅 month"],
            "Description": [
                "Ranking FIFA équipe domicile", "Ranking FIFA équipe extérieure",
                "Forme récente domicile (% victoires)", "Forme récente extérieure (% victoires)",
                "Historique head-to-head", "Type de compétition",
                "Terrain neutre (oui/non)", "Match amical (oui/non)",
                "Équipe domicile encodée", "Équipe extérieure encodée",
                "Année du match", "Mois du match"
            ],
            "Importance": ["🔥🔥🔥", "🔥🔥🔥", "🔥🔥🔥", "🔥🔥🔥",
                          "🔥🔥", "🔥🔥", "🔥🔥", "🔥🔥",
                          "🔥", "🔥", "🔥", "🔥"]
        }
        st.dataframe(pd.DataFrame(features_data), use_container_width=True, hide_index=True)

        st.markdown("---")
        st.markdown("### 📈 Analyse exploratoire")
        st.image(str(PLOTS_DIR / "eda_results_by_year.png"), use_container_width=True, caption="Evolution des résultats par année")

    # ==================== PAGE MODÈLES ====================
    elif "Modèles" in menu:
        st.markdown('<div class="main-title">📊 COMPARAISON DES MODÈLES</div>', unsafe_allow_html=True)
        st.markdown('<div class="subtitle">Évaluation et comparaison de 3 algorithmes de Machine Learning</div>', unsafe_allow_html=True)
        st.markdown("---")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown('<div class="metric-box"><span style="font-size:2em">🥉</span><h3 style="color:white">Logistic Regression</h3><div class="metric-value" style="color:#ffaa00">53%</div><div class="metric-label">Accuracy</div><br><div class="metric-value" style="color:#ffaa00">0.49</div><div class="metric-label">F1 Macro</div></div>', unsafe_allow_html=True)
        with col2:
            st.markdown('<div class="metric-box"><span style="font-size:2em">🥈</span><h3 style="color:white">Random Forest</h3><div class="metric-value" style="color:#00aaff">55%</div><div class="metric-label">Accuracy</div><br><div class="metric-value" style="color:#00aaff">0.49</div><div class="metric-label">F1 Macro</div></div>', unsafe_allow_html=True)
        with col3:
            st.markdown('<div class="metric-box" style="border-color:#00ff44"><span style="font-size:2em">🥇</span><h3 style="color:#00ff44">Gradient Boosting</h3><div class="metric-value" style="color:#00ff44">57%</div><div class="metric-label">Accuracy</div><br><div class="metric-value" style="color:#00ff44">0.48</div><div class="metric-label">F1 Macro</div></div>', unsafe_allow_html=True)

        st.markdown("---")

        # Graphique Plotly
        fig = go.Figure()
        models = ['Logistic\nRegression', 'Random\nForest', 'Gradient\nBoosting']
        fig.add_trace(go.Bar(
            name='Accuracy', x=models, y=[0.53, 0.55, 0.57],
            marker=dict(color=['#ffaa00', '#00aaff', '#00ff44'],
                       line=dict(color='white', width=1)),
            text=['53%', '55%', '57%'], textposition='auto',
        ))
        fig.add_trace(go.Bar(
            name='F1 Macro', x=models, y=[0.49, 0.49, 0.48],
            marker=dict(color=['#cc8800', '#0088cc', '#00cc33'],
                       line=dict(color='white', width=1), opacity=0.7),
            text=['0.49', '0.49', '0.48'], textposition='auto',
        ))
        fig.update_layout(
            title=dict(text='Performance des modèles', font=dict(size=20, color='white')),
            barmode='group',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            legend=dict(bgcolor='rgba(0,0,0,0)', bordercolor='#333'),
            xaxis=dict(gridcolor='#222'),
            yaxis=dict(gridcolor='#222', range=[0, 0.7]),
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        st.markdown("### 📊 Courbes ROC")
        st.image(str(PLOTS_DIR / "roc_curves.png"), use_container_width=True, caption="Courbes ROC des 3 modèles")

        st.markdown("### 🎯 Matrice de confusion - Gradient Boosting")
        st.image(str(PLOTS_DIR / "confusion_matrix.png"), use_container_width=True, caption="Matrice de confusion")

        st.markdown("---")
        st.markdown("### ℹ️ Détails des modèles")

        col1, col2, col3 = st.columns(3)
        with col1:
            with st.expander("📘 Logistic Regression"):
                st.markdown("""
                **Hypothèses :** relation linéaire entre features et cible
                
                ✅ Simple et interprétable  
                ✅ Rapide à entraîner  
                ❌ Ne capture pas les non-linéarités  
                ❌ Moins performant sur données complexes
                """)
        with col2:
            with st.expander("🌲 Random Forest"):
                st.markdown("""
                **Hypothèses :** combinaison de 50 arbres de décision
                
                ✅ Robuste aux outliers  
                ✅ Gère bien les features catégorielles  
                ❌ Moins performant que le boosting  
                ❌ Fichier modèle volumineux
                """)
        with col3:
            with st.expander("🚀 Gradient Boosting ⭐"):
                st.markdown("""
                **Hypothèses :** arbres entraînés séquentiellement
                
                ✅ Meilleure accuracy (57%)  
                ✅ Performant sur données tabulaires  
                ❌ Plus long à entraîner  
                ❌ Risque d'overfitting
                """)

        st.markdown("---")
        st.markdown("### ⚠️ Limites identifiées")
        col1, col2 = st.columns(2)
        with col1:
            st.warning("**Class imbalance** : Draw très difficile à prédire (seulement 23% des matchs)")
        with col2:
            st.info("**Limite naturelle** : Les bookmakers professionnels atteignent ~60% max sur le foot")

    # ==================== PAGE DÉMO ====================
    elif "Démo" in menu:
        st.markdown('<div class="main-title">⚽ MATCH PREDICTOR</div>', unsafe_allow_html=True)
        st.markdown('<div class="subtitle">Sélectionnez deux équipes et découvrez le résultat prédit</div>', unsafe_allow_html=True)
        st.markdown("---")

        model = joblib.load(MODELS_DIR / "gradient_boosting.joblib")

        col1, col_vs, col2 = st.columns([5, 2, 5])

        with col1:
            st.markdown("### 🏠 Équipe Domicile")
            home_team = st.selectbox("", list(TEAMS.keys()), key="home", label_visibility="collapsed")
            home_info = TEAMS[home_team]
            st.markdown(f"""
            <div class="team-card">
                <span class="team-flag">{home_info['flag']}</span>
                <div class="team-name">{home_team}</div>
                <div class="team-rank">🏆 Ranking FIFA : #{home_info['rank']}</div>
            </div>
            """, unsafe_allow_html=True)
            home_rank = st.slider("🏆 Ranking FIFA", 1, 210, home_info['rank'], key="home_rank")
            home_form = st.slider("📈 Forme récente", 0.0, 1.0, 0.6, 0.05, key="home_form",
                                 help="% de victoires sur les 10 derniers matchs")

        with col_vs:
            st.markdown("<br><br><br><br><br>", unsafe_allow_html=True)
            st.markdown('<div class="vs-container"><div class="vs-text">VS</div></div>', unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            neutral = st.checkbox("⚖️ Terrain neutre", help="Match joué dans un pays tiers")
            is_friendly = st.checkbox("🤝 Match amical")

        with col2:
            st.markdown("### ✈️ Équipe Extérieure")
            away_team = st.selectbox("", list(TEAMS.keys()), index=1, key="away", label_visibility="collapsed")
            away_info = TEAMS[away_team]
            st.markdown(f"""
            <div class="team-card">
                <span class="team-flag">{away_info['flag']}</span>
                <div class="team-name">{away_team}</div>
                <div class="team-rank">🏆 Ranking FIFA : #{away_info['rank']}</div>
            </div>
            """, unsafe_allow_html=True)
            away_rank = st.slider("🏆 Ranking FIFA", 1, 210, away_info['rank'], key="away_rank")
            away_form = st.slider("📈 Forme récente", 0.0, 1.0, 0.5, 0.05, key="away_form",
                                 help="% de victoires sur les 10 derniers matchs")

        st.markdown("---")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            predict_btn = st.button("🔮 PRÉDIRE LE RÉSULTAT", use_container_width=True)

        if predict_btn:
            X = np.array([[
                0, 1, 0,
                int(neutral), 2024, 6,
                home_form, away_form,
                int(is_friendly), 0.5,
                home_rank, away_rank
            ]])

            pred = model.predict(X)[0]
            proba = model.predict_proba(X)[0]
            labels = ['Away Win', 'Draw', 'Home Win']
            result = labels[pred]

            st.markdown("<br>", unsafe_allow_html=True)

            if result == 'Home Win':
                css_class = "result-win"
                color = "#00ff44"
                emoji = "🏠"
                winner = f"{home_info['flag']} {home_team}"
            elif result == 'Draw':
                css_class = "result-draw"
                color = "#ffaa00"
                emoji = "🤝"
                winner = "Match nul"
            else:
                css_class = "result-away"
                color = "#ff4444"
                emoji = "✈️"
                winner = f"{away_info['flag']} {away_team}"

            st.markdown(f"""
            <div class="{css_class}">
                <div style="font-size:3em">{emoji}</div>
                <div class="result-text" style="color:{color}">{result}</div>
                <div style="color:white; font-size:1.3em; margin-top:10px">{winner}</div>
                <div style="color:#888; margin-top:5px">{home_info['flag']} {home_team} vs {away_info['flag']} {away_team}</div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Graphique probabilités
            fig = go.Figure(go.Bar(
                x=[f"{home_info['flag']} {home_team}", "🤝 Draw", f"{away_info['flag']} {away_team}"],
                y=[proba[2], proba[1], proba[0]],
                marker=dict(
                    color=[
                        f'rgba(0,255,68,{0.9 if result=="Home Win" else 0.4})',
                        f'rgba(255,170,0,{0.9 if result=="Draw" else 0.4})',
                        f'rgba(255,68,68,{0.9 if result=="Away Win" else 0.4})'
                    ],
                    line=dict(color='white', width=1)
                ),
                text=[f"{p:.1%}" for p in [proba[2], proba[1], proba[0]]],
                textposition='auto',
                textfont=dict(size=16, color='white')
            ))
            fig.update_layout(
                title=dict(text='Probabilités de chaque résultat', font=dict(size=18, color='white')),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                yaxis=dict(tickformat='.0%', range=[0, 1], gridcolor='#222'),
                xaxis=dict(gridcolor='#222'),
                showlegend=False,
                height=350
            )
            st.plotly_chart(fig, use_container_width=True)


             # ==================== PAGE PARIS ====================

    elif "Paris" in menu:
        st.markdown('<div class="main-title">💰 SIMULATION PARIS SPORTIFS</div>', unsafe_allow_html=True)
        st.markdown('<div class="subtitle">Parcours du Brésil à la coupe du monde 2014</div>', unsafe_allow_html=True)
        st.markdown("---")

        # Matchs du parcours du Brésil
        parcours = [
    {"phase": "Groupe", "home": "Brésil", "away": "Croatie",
     "home_rank": 4, "away_rank": 18, "home_form": 0.8, "away_form": 0.6,
     "neutral": False, "is_friendly": False,
     "vrai_resultat": "Home Win",
     "cote_home": 1.45, "cote_draw": 4.2, "cote_away": 7.5},
    {"phase": "Groupe", "home": "Brésil", "away": "Mexique",
     "home_rank": 4, "away_rank": 20, "home_form": 0.8, "away_form": 0.6,
     "neutral": False, "is_friendly": False,
     "vrai_resultat": "Draw",
     "cote_home": 1.60, "cote_draw": 3.8, "cote_away": 6.0},
    {"phase": "Groupe", "home": "Brésil", "away": "Cameroun",
     "home_rank": 4, "away_rank": 50, "home_form": 0.8, "away_form": 0.4,
     "neutral": False, "is_friendly": False,
     "vrai_resultat": "Home Win",
     "cote_home": 1.25, "cote_draw": 6.0, "cote_away": 12.0},
    {"phase": "8èmes", "home": "Brésil", "away": "Chili",
     "home_rank": 4, "away_rank": 14, "home_form": 0.8, "away_form": 0.7,
     "neutral": False, "is_friendly": False,
     "vrai_resultat": "Home Win",
     "cote_home": 1.65, "cote_draw": 3.6, "cote_away": 5.5},
    {"phase": "Quarts", "home": "Brésil", "away": "Colombie",
     "home_rank": 4, "away_rank": 8, "home_form": 0.8, "away_form": 0.7,
     "neutral": False, "is_friendly": False,
     "vrai_resultat": "Home Win",
     "cote_home": 1.70, "cote_draw": 3.5, "cote_away": 5.0},
    {"phase": "Demis", "home": "Brésil", "away": "Allemagne",
     "home_rank": 4, "away_rank": 2, "home_form": 0.7, "away_form": 0.8,
     "neutral": False, "is_friendly": False,
     "vrai_resultat": "Away Win",
     "cote_home": 2.10, "cote_draw": 3.4, "cote_away": 3.4},
    {"phase": "3ème place", "home": "Brésil", "away": "Pays-Bas",
     "home_rank": 4, "away_rank": 5, "home_form": 0.5, "away_form": 0.7,
     "neutral": False, "is_friendly": False,
     "vrai_resultat": "Away Win",
     "cote_home": 2.20, "cote_draw": 3.3, "cote_away": 3.3},
]

        model = joblib.load(MODELS_DIR / "gradient_boosting.joblib")
        mise = 10  # €

        st.markdown(f"### 💵 Mise par match : **{mise}€**")
        st.markdown("---")

        total_mise = 0
        total_gains = 0
        results_table = []

        labels = ['Away Win', 'Draw', 'Home Win']

        for match in parcours:
            X = np.array([[
                0, 1, 0,
                int(match["neutral"]), 2022, 11,
                match["home_form"], match["away_form"],
                int(match["is_friendly"]), 0.5,
                match["home_rank"], match["away_rank"]
            ]])

            pred = model.predict(X)[0]
            prediction = labels[pred]
            vrai = match["vrai_resultat"]

            # Cote correspondant à la prédiction
            if prediction == "Home Win":
                cote_pariee = match["cote_home"]
            elif prediction == "Draw":
                cote_pariee = match["cote_draw"]
            else:
                cote_pariee = match["cote_away"]

            # Calcul gain/perte
            if prediction == vrai:
                gain = round(mise * cote_pariee - mise, 2)
                correct = "✅"
            else:
                gain = -mise
                correct = "❌"

            total_mise += mise
            total_gains += gain

            results_table.append({
                "Phase": match["phase"],
                "Match": f"{match['home']} vs {match['away']}",
                "Prédiction": prediction,
                "Résultat réel": vrai,
                "Cote": cote_pariee,
                "Gain/Perte": f"{'+' if gain > 0 else ''}{gain}€",
                "": correct
            })

        # Afficher le tableau
        df_results = pd.DataFrame(results_table)
        st.dataframe(df_results, use_container_width=True, hide_index=True)

        st.markdown("---")

        # Bilan final
        bilan = round(total_gains, 2)
        col1, col2, col3 = st.columns(3)
        col1.markdown(f'<div class="metric-box"><div class="metric-value" style="color:#ff4444">{total_mise}€</div><div class="metric-label">Total misé</div></div>', unsafe_allow_html=True)
        col2.markdown(f'<div class="metric-box"><div class="metric-value" style="color:{"#00ff44" if bilan > 0 else "#ff4444"}">{("+" if bilan > 0 else "")}{bilan}€</div><div class="metric-label">Bilan net</div></div>', unsafe_allow_html=True)
        col3.markdown(f'<div class="metric-box"><div class="metric-value" style="color:{"#00ff44" if bilan > 0 else "#ff4444"}">{round((bilan/total_mise)*100, 1)}%</div><div class="metric-label">ROI</div></div>', unsafe_allow_html=True)

        # Graphique évolution
        gains_cumules = []
        cumul = 0
        for match in results_table:
            val = float(match["Gain/Perte"].replace("€","").replace("+",""))
            cumul += val
            gains_cumules.append(cumul)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=[m["Phase"] + " - " + m["Match"].split(" vs ")[1] for m in results_table],
            y=gains_cumules,
            mode='lines+markers',
            line=dict(color='#00ff44', width=3),
            marker=dict(size=10, color=['#00ff44' if g >= 0 else '#ff4444' for g in gains_cumules]),
            fill='tozeroy',
            fillcolor='rgba(0,255,68,0.1)'
        ))
        fig.add_hline(y=0, line_dash="dash", line_color="white", opacity=0.5)
        fig.update_layout(
            title=dict(text='Evolution des gains cumulés', font=dict(size=18, color='white')),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            xaxis=dict(gridcolor='#222'),
            yaxis=dict(gridcolor='#222', ticksuffix='€'),
        )
        st.plotly_chart(fig, use_container_width=True)

        if bilan > 0:
            st.success(f"🎉 En suivant les prédictions du modèle sur le parcours du Brésil, vous auriez gagné **{bilan}€** pour {total_mise}€ misés !")
        else:
            st.error(f"😔 En suivant les prédictions du modèle sur le parcours du Brésil, vous auriez perdu **{abs(bilan)}€** pour {total_mise}€ misés.")

if __name__ == "__main__":
    build_app()