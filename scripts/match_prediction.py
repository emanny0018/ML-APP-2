import os
import pandas as pd
from joblib import load

def load_data(file_path):
    """Load dataset from a file."""
    return pd.read_csv(file_path)

def get_match_features(df, home_team, away_team):
    """Extract features for a specific match based on engineered features."""
    match = df[(df['Home'] == home_team) & (df['Away'] == away_team)]
    
    if match.empty:
        raise ValueError(f"Match between {home_team} and {away_team} not found in the dataset.")
    
    return match.iloc[0]

def display_available_teams(df):
    """Display available teams in the dataset to avoid input errors."""
    teams = sorted(df['Home'].unique())
    print("\nAvailable teams in the dataset:")
    for team in teams:
        print(f"  - {team}")

def calculate_predicted_scores(predicted_proba):
    """Calculate predicted scores based on probabilities."""
    home_goals = int(predicted_proba[0][0] * 3)
    away_goals = int(predicted_proba[0][1] * 3)
    return home_goals, away_goals

if __name__ == "__main__":
    # Load the trained model
    model_path = 'data/voting_classifier.pkl'
    model = load(model_path)

    # Load the engineered data
    df = load_data('data/fe_combined_data.csv')

    # Display available teams
    display_available_teams(df)

    # Get inputs from environment variables or set them manually for testing
    home_team = os.getenv("HOME_TEAM", "liverpool").strip().lower()
    away_team = os.getenv("AWAY_TEAM", "manchester city").strip().lower()

    # Extract features for the prediction based on the available data
    features = get_match_features(df, home_team, away_team)

    # Define predictors based on your feature engineering process
    advanced_predictors = [
        "Venue_Code", "Opp_Code", "Day_Code", "Rolling_HomeGoals", "Rolling_AwayGoals", 
        "Venue_Opp_Interaction", "Decayed_Rolling_HomeGoals", "Decayed_Rolling_AwayGoals", 
        "Home_Advantage", "Home_Streak_Wins", "Away_Streak_Losses"
    ]

    # Select the relevant features from the match data
    features = features[advanced_predictors].to_frame().T

    # Predict the outcome
    prediction = model.predict(features)
    predicted_proba = model.predict_proba(features)

    # Calculate predicted scores
    home_goals, away_goals = calculate_predicted_scores(predicted_proba)

    # Output the predicted result and scores
    result_map = {0: "Home Win", 1: "Away Win", 2: "Draw"}
    predicted_result = result_map.get(prediction[0], "Unknown Result")
    
    print(f"\nPredicted Outcome: {home_team.capitalize()} vs {away_team.capitalize()}")
    print(f"Predicted Result: {predicted_result}")
    print(f"Predicted Score: {home_team.capitalize()} {home_goals}-{away_goals} {away_team.capitalize()}")
    print(f"Predicted Score Probability: Home Win {predicted_proba[0][0]:.2f}, Away Win {predicted_proba[0][1]:.2f}, Draw {predicted_proba[0][2]:.2f}")

    # Save the prediction results to a file
    results_path = "data/prediction_results.txt"
    with open(results_path, "w") as f:
        f.write(f"Predicted Outcome: {home_team.capitalize()} vs {away_team.capitalize()}\n")
        f.write(f"Predicted Result: {predicted_result}\n")
        f.write(f"Predicted Score: {home_team.capitalize()} {home_goals}-{away_goals} {away_team.capitalize()}\n")
        f.write(f"Predicted Score Probability: Home Win {predicted_proba[0][0]:.2f}, Away Win {predicted_proba[0][1]:.2f}, Draw {predicted_proba[0][2]:.2f}\n")
    
    print(f"\nPrediction results saved to {results_path}.")

