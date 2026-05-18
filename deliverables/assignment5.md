# Assignment 5 - Application Streamlit

## Description de l'application :
Dashboard interactif de prédiction de résultats de matchs de football international.

## Objectif de l'interface :
- **Exploration** : comprendre les données et le problème
- **Comparaison** : évaluer les 3 modèles ML
- **Prédiction** : simuler le résultat d'un match en temps réel

## Structure de l'application (3 pages) :

### 🏠 Page Projet
- Présentation du problème et objectif business
- Statistiques clés du dataset (32 101 matchs, 12 features)
- Distribution de la variable cible
- Tableau des features avec leur importance

### 📊 Page Modèles
- Comparaison des 3 modèles (LR, RF, GB)
- Graphique interactif Plotly des métriques
- Détails de chaque modèle (avantages, limites)
- Protocole d'évaluation

### ⚽ Page Démo
- Sélection de deux équipes avec drapeaux
- Sliders pour ranking FIFA et forme récente
- Options terrain neutre / match amical
- Prédiction en temps réel avec probabilités

## Inputs utilisateurs :
- Équipe domicile (liste déroulante)
- Équipe extérieure (liste déroulante)
- Ranking FIFA domicile (slider 1-210)
- Ranking FIFA extérieur (slider 1-210)
- Forme récente domicile (slider 0-1)
- Forme récente extérieure (slider 0-1)
- Terrain neutre (checkbox)
- Match amical (checkbox)

## Outputs affichés :
- Résultat prédit (Home Win / Draw / Away Win)
- Probabilités de chaque résultat (graphique Plotly)

## Comment lancer l'application :
```bash
cd ml-poc-project
python3 scripts/main.py
```
L'application est accessible sur http://localhost:8501

## Modèle utilisé :
Gradient Boosting (meilleur modèle, 57% accuracy)
Fichier : `models/gradient_boosting.joblib`