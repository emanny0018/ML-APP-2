import pandas as pd
from joblib import load
import os

def load_data(file_path):
    """Load the dataset from the given file path."""
    print(f"Loading data from {file_path}...")
    return pd.read_csv(file_path)

def get_match_features(df, home_team, away_team):
    """
    Extract match features for a specific home and away team.
    If match is not found, or required columns are missing, raise an exception.
    """
    if 'Home' not in df.columns or 'Away' not in df.columns:
        raise ValueError("Required columns 'Home' and 'Away' are missing from the dataset.")

    # Find the match for the specified home and away team
    match = df[(df['Home'].str.lower() == home_team.lower()) & (df['Away'].str.lower() == away_team.lower())]
    
    if match.empty:
        raise ValueError(f"Match between {home_team} and {away_team} not found in the dataset.")
    
    print(f"Match found between {home_team} and {away_team}.")
    return match

def predict_match_result(model, features):
    """Make a prediction for the given match features using the loaded model."""
    if features.empty:
        raise ValueError("No valid match features to make a prediction.")
    
    # Prediction
    prediction = model.predict(features)
    return prediction

if __name__ == "__main__":
    # Step 1: Load the trained model
    model_path = 'data/voting_classifier.pkl'
    print(f"Loading trained model from {model_path}...")
    model = load(model_path)

    # Step 2: Load the dataset containing match features
    data_path = 'data/fe_combined_matches.csv'
    df = load_data(data_path)

    # Step 3: Get home and away team from environment variables (or defaults)
    home_team = os.getenv("HOME_TEAM", "liverpool")
    away_team = os.getenv("AWAY_TEAM", "manchester city")

    # Step 4: Extract the features for the specified match
    try:
        match_features = get_match_features(df, home_team, away_team)
        
        # Step 5: Ensure we only pass the available predictors to the model
        predictors = [
            "Venue_Code", "Opp_Code", "Day_Code", "Rolling_HomeGoals", "Rolling_AwayGoals", 
            "Venue_Opp_Interaction", "Decayed_Rolling_HomeGoals", "Decayed_Rolling_AwayGoals", 
            "Home_Advantage", "Home_Streak_Wins", "Away_Streak_Losses"
        ]

        available_predictors = [col for col in predictors if col in match_features.columns]
        
        if not available_predictors:
            raise ValueError("No valid predictors available for the model to make predictions.")

        print(f"Available predictors for match prediction: {available_predictors}")

        # Step 6: Make the prediction using the available features
        prediction = predict_match_result(model, match_features[available_predictors])
        print(f"Predicted result: {prediction[0]}")

    except Exception as e:
        print(f"Error: {e}")

