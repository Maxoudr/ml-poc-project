# Assignment 3 - Modélisation

## Définition du problème ML :
Classification supervisée à 3 classes : Home Win / Draw / Away Win

## Métrique d'évaluation :
- F1-score macro (traite chaque classe également, adapté au class imbalance)
- Accuracy (pour comparaison)

## Protocole d'évaluation :
- Train/Test split : 80% / 20% (25 680 / 6 421 matchs)
- SMOTE appliqué sur le train set pour rééquilibrer les classes
- random_state=42 pour la reproductibilité

## Les 3 modèles :

### 1. Logistic Regression (baseline)
- **Hypothèses** : relation linéaire entre les features et la cible
- **Avantages** : simple, rapide, interprétable
- **Limites** : ne capture pas les relations non-linéaires
- **Résultats** : Accuracy 0.53 | F1 macro 0.49

### 2. Random Forest
- **Hypothèses** : combinaison de plusieurs arbres de décision
- **Avantages** : robuste, gère bien les features catégorielles
- **Limites** : moins performant que le boosting sur données tabulaires
- **Résultats** : Accuracy 0.55 | F1 macro 0.49

### 3. Gradient Boosting
- **Hypothèses** : arbres entraînés séquentiellement pour corriger les erreurs
- **Avantages** : meilleure performance sur données tabulaires
- **Limites** : plus long à entraîner, risque d'overfitting
- **Résultats** : Accuracy 0.57 | F1 macro 0.48

## Justification des choix :
- LR comme baseline simple et interprétable
- RF pour sa robustesse sur les features catégorielles
- GB car généralement le meilleur sur données tabulaires

## Limites identifiées :
- Class imbalance (Draw très difficile à prédire)
- Le foot est intrinsèquement difficile à prédire (~60% max pour les bookmakers)
- Ranking FIFA manquant pour certaines équipes (valeur par défaut 150)

## Notebooks :
- `notebooks/first_model.ipynb` : entraînement et comparaison des 3 modèles
- Relancer toutes les cellules dans l'ordre pour reproduire les résultats
- Les modèles sont sauvegardés dans `models/`