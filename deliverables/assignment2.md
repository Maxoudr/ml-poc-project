# Assignment 2 - Feature Engineering

## Nettoyage des données :
- Suppression des 72 lignes avec scores manquants (0.15% du dataset)
- Filtrage des matchs avant 1990 (données trop bruitées)
- Résultat : 32 101 matchs utilisables

## Nouvelles features créées :
- `year` : année extraite de la date
- `month` : mois extrait de la date
- `result` : variable cible (Home Win / Draw / Away Win)

## Features supprimées :
- `home_score` / `away_score` : utilisées uniquement pour créer la cible
- `city` / `country` : trop de modalités, peu d'impact attendu
- `date` : remplacée par year et month

## Transformations appliquées :
- LabelEncoder sur home_team, away_team, tournament, result
- Conversion de neutral (bool -> int)
- StandardScaler sur toutes les features

## Train/Test split :
- 80% train : 25 680 matchs
- 20% test : 6 421 matchs
- random_state=42

## Alternatives non retenues :
- One-Hot Encoding sur les équipes : trop de colonnes (300+ équipes)
- PCA : La variance est répartie uniformément entre les 6 composantes (~17% chacune). Il faut 5 composantes pour expliquer 86% de la variance. Le PCA n'apporte pas de réduction utile ici. On conserve les 6 features originales sans PCA
- Données avant 1990 : trop bruitées

## Impact attendu :
- LabelEncoding simple et efficace pour les équipes
- StandardScaler nécessaire pour la Logistic Regression
- Class imbalance à surveiller (Home Win 48%, Draw 23%, Away Win 28%)

## Datasets transformés :
- Localisés dans `data/` (ignoré par git)
- Fichiers : X_train.npy, X_test.npy, y_train.npy, y_test.npy
- Chargement via `src/data.py` avec la fonction `load_dataset_split()`

## Notebook :
- `notebooks/data_exploration.ipynb`
- Exécuter toutes les cellules dans l'ordre