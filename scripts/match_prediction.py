import pandas as pd
from joblib import load
import os

def load_data(file_path):
    return pd.read_csv(file_path)

def get_match_features(df, home_team, away_team):
    match = df[(df['Home'] == home_team) & (df['Away'] == away_team)]
    if match.empty:
        raise ValueError("Match not found.")
    return match

if __name__ == "__main__":
    model = load('data/voting_classifier.pkl')
    df = load_data('data/fe_combined_matches.csv')

    # Assume teams are passed via environment variables
    home_team = os.getenv("HOME_TEAM", "liverpool").lower()
    away_team = os.getenv("AWAY_TEAM", "manchester city").lower()

    features = get_match_features(df, home_team, away_team)

    # Predict the outcome
    prediction = model.predict(features)
    print(f"Predicted result: {prediction}")
