# Football Match Predictor

A machine learning project that predicts the outcome of international football matches (Home Win / Draw / Away Win) using historical data from 1990 to 2024.

## Project Overview

- **Problem type**: Multi-class classification (3 classes)
- **Best model**: Gradient Boosting (57% accuracy)
- **Features**: 12 variables including FIFA ranking, recent form, head-to-head history
- **Dataset**: 32,101 international matches

## Repository Structure
ml-poc-project/
├── data/                    # Data files (not tracked by git)
├── deliverables/            # Assignment markdown files
│   ├── assignment1.md       # Project proposal
│   ├── assignment2.md       # Feature engineering
│   ├── assignment3.md       # Model selection
│   ├── assignment4.md       # Visualizations
│   └── assignment5.md       # Streamlit app
├── models/                  # Trained ML models (.joblib)
├── notebooks/               # Jupyter notebooks
│   ├── data_exploration.ipynb
│   └── first_model.ipynb
├── plots/                   # Saved visualizations
├── results/                 # Model evaluation results
├── scripts/
│   └── main.py              # Main entry point
└── src/
├── app.py               # Streamlit application
├── config.py            # Project configuration
├── data.py              # Data loading function
└── metrics.py           # Evaluation metrics

## Installation

### 1. Clone the repository
```bash
git clone git@github.com:Maxoudr/ml-poc-project.git
cd ml-poc-project
```

### 2. Create and activate virtual environment
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

## How to Get the Data

### Dataset 1 - International Football Results
1. Go to https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017
2. Download the dataset
3. Place the following files in the `data/` folder:
   - `results.csv`
   - `goalscorers.csv`
   - `shootouts.csv`
   - `former_names.csv`

### Dataset 2 - FIFA World Ranking
1. Go to https://www.kaggle.com/datasets/cashncarry/fifaworldranking
2. Download the dataset
3. Place `fifa_ranking-2023-07-20.csv` in the `data/` folder

### Retrain the models
After adding the data, run the notebooks in order:
```bash
# 1. Open and run all cells in notebooks/data_exploration.ipynb
# 2. Open and run all cells in notebooks/first_model.ipynb
```

## Run the Project

```bash
python3 scripts/main.py
```

This will:
1. Evaluate the 3 trained models
2. Save results to `results/model_metrics.csv`
3. Launch the Streamlit app at http://localhost:8501

## Models

| Model | Accuracy | F1 Macro |
|-------|----------|----------|
| Logistic Regression | 53% | 0.49 |
| Random Forest | 55% | 0.49 |
| Gradient Boosting | **57%** | **0.48** |

## Features

| Feature | Description |
|---------|-------------|
| home_fifa_rank | FIFA ranking of home team |
| away_fifa_rank | FIFA ranking of away team |
| home_form | Home team recent win rate (last 10 matches) |
| away_form | Away team recent win rate (last 10 matches) |
| h2h | Head-to-head historical win rate |
| tournament_enc | Competition type (encoded) |
| neutral | Neutral venue (0/1) |
| is_friendly | Friendly match (0/1) |
| year | Match year |
| month | Match month |