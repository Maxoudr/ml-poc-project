# Assignment 1 - Proposition de projet ML

## Mon projet :
Prédire le résultat d'un match de football international (victoire domicile, nul, victoire extérieur)

## Le business case :
Aider les analystes sportifs et parieurs à anticiper les résultats 
des matchs internationaux pour optimiser leurs décisions stratégiques.

## Description du dataset :
- Nom : International football results from 1872 to 2024
- Source : Kaggle - https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017
- Fichiers : results.csv, goalscorers.csv, shootouts.csv, former_names.csv
- Nombre de matchs : 49 215 (après nettoyage)
- Période couverte : 1872 à 2024 (on utilisera 1990-2024 soit 32 101 matchs)

## Type de données :
- Variables catégorielles : home_team, away_team, tournament, city, country
- Variables numériques : home_score, away_score
- Variable booléenne : neutral (terrain neutre ou non)

## Méthode de collecte :
Téléchargement manuel depuis Kaggle

## Justification du choix :
Dataset propre, bien documenté, peu de valeurs manquantes (0.15%), 
et suffisamment grand pour entraîner un modèle de classification.

## Variable cible :
result : Home Win / Draw / Away Win

## Limites éventuelles :
- Class imbalance : Home Win 48%, Away Win 28%, Draw 23%
- Matchs amicaux (Friendly) très nombreux, moins représentatifs
- Peu de features disponibles directement

## Objectif ML :
Problème de classification à 3 classes

## Métrique d'évaluation :
- Accuracy
- F1-score (weighted) pour tenir compte du class imbalance

## Description des features disponibles :
- `home_team` : équipe qui joue à domicile
- `away_team` : équipe qui joue à l'extérieur
- `tournament` : type de compétition (Coupe du Monde, amical, qualifications...)
- `city` : ville où se joue le match
- `country` : pays hôte
- `neutral` : booléen, True si le match se joue sur terrain neutre
- `date` : date du match (permettra d'extraire l'année, le mois...)
- `home_score` / `away_score` : scores (utilisés uniquement pour créer la variable cible)

## Contexte machine learning :
Problème de classification supervisée à 3 classes (Home Win / Draw / Away Win).
Les modèles envisagés sont :
- Logistic Regression (baseline)
- Random Forest
- XGBoost

## Comment obtenir les données :
Dataset téléchargé manuellement depuis Kaggle :
https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017
Les fichiers CSV sont à placer dans le dossier `data/` du repository.
Ce dossier est ignoré par Git (.gitignore) pour éviter de pusher des fichiers lourds.

## Comment exécuter le notebook :
1. Placer les fichiers CSV dans le dossier `data/`
2. Ouvrir `notebooks/data_exploration.ipynb` dans VS Code ou Jupyter
3. Mettre à jour le chemin des fichiers si nécessaire
4. Exécuter toutes les cellules dans l'ordre