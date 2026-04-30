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