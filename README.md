# ⚽ Football Match Predictor

Prédiction de résultats de matchs de football international par Machine Learning.

## Description
Ce projet prédit le résultat d'un match de football international (Home Win / Draw / Away Win) 
en utilisant des données historiques depuis 1990 et 3 modèles de Machine Learning.

## Installation

### 1. Cloner le repo
```bash
git clone git@github.com:Maxoudr/ml-poc-project.git
cd ml-poc-project
```

### 2. Créer un environnement virtuel
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Installer les dépendances
```bash
pip install -r requirements.txt
```

### 4. Ajouter les données
Télécharger les datasets depuis Kaggle :
- https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017
- https://www.kaggle.com/datasets/cashncarry/fifaworldranking

Placer les fichiers CSV dans le dossier `data/`

## Exécution

### Lancer le projet complet
```bash
python3 scripts/main.py
```

Cela va :
1. Évaluer les 3 modèles
2. Sauvegarder les résultats dans `results/model_metrics.csv`
3. Lancer l'application Streamlit sur http://localhost:8501

## Structure du projet

ml-poc-project/
├── data/               # Données (ignorées par git)
├── deliverables/       # Assignments rendus
├── models/             # Modèles entraînés
├── notebooks/          # Notebooks d'exploration et modélisation
├── plots/              # Visualisations sauvegardées
├── results/            # Résultats des modèles
├── src/                # Code source
│   ├── app.py          # Application Streamlit
│   ├── config.py       # Configuration
│   ├── data.py         # Chargement des données
│   └── metrics.py      # Métriques d'évaluation
└── scripts/
└── main.py         # Point d'entrée principal

## Modèles
| Modèle | Accuracy | F1 Macro |
|--------|----------|----------|
| Logistic Regression | 53% | 0.49 |
| Random Forest | 55% | 0.49 |
| **Gradient Boosting** | **57%** | **0.48** |