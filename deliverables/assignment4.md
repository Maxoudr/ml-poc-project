# Assignment 4 - Visualisations

## Plot 1 - EDA : Evolution des résultats par année
- **Fichier** : `plots/eda_results_by_year.png`
- **Objectif** : Visualiser l'évolution des résultats depuis 1990
- **Type** : Courbe temporelle (line chart)
- **Interprétation** : Home Win stable à ~48%, Away Win en légère hausse, Draw en baisse

## Plot 2 - Performances : Courbes ROC
- **Fichier** : `plots/roc_curves.png`
- **Objectif** : Comparer les 3 modèles sur chaque classe
- **Type** : ROC curves avec AUC
- **Interprétation** : Gradient Boosting meilleur AUC sur Home Win et Away Win

## Plot 3 - Résultats : Matrice de confusion
- **Fichier** : `plots/confusion_matrix.png`
- **Objectif** : Montrer les vrais/faux positifs du meilleur modèle
- **Type** : Heatmap
- **Interprétation** : Le modèle prédit bien Home Win mais a du mal avec Draw

## Notebooks :
- `notebooks/data_exploration.ipynb` → plots EDA
- `notebooks/first_model.ipynb` → plots modèles
- Relancer toutes les cellules pour régénérer les plots