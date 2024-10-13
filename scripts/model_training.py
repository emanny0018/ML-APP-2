import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

def train_model():
    # Step 1: Load the feature-engineered dataset
    data = pd.read_csv('data/fe_combined_matches.csv')

    # Step 2: Define possible predictors
    predictors = [
        "Venue_Code", "Opp_Code", "Day_Code", "Rolling_HomeGoals", "Rolling_AwayGoals", 
        "Venue_Opp_Interaction", "Decayed_Rolling_HomeGoals", "Decayed_Rolling_AwayGoals", 
        "Home_Advantage", "Home_Streak_Wins", "Away_Streak_Losses"
    ]

    # Step 3: Filter available predictors
    available_predictors = [col for col in predictors if col in data.columns]
    if not available_predictors:
        raise ValueError("No valid predictors found for model training.")
    
    print(f"Available predictors for training: {available_predictors}")

    # Step 4: Define target
    target = "Target"
    if target not in data.columns:
        raise ValueError(f"Target column '{target}' is missing in the dataset.")

    # Step 5: Split the data into train and test sets
    train, test = train_test_split(data, test_size=0.2, random_state=42, stratify=data[target])

    # Step 6: Train Voting Classifier (XGBoost and RandomForest)
    xgb = XGBClassifier(random_state=42)
    rf = RandomForestClassifier(random_state=42)

    model = VotingClassifier(estimators=[('xgb', xgb), ('rf', rf)], voting='soft')
    model.fit(train[available_predictors], train[target])

    # Step 7: Save the trained model
    joblib.dump(model, 'data/voting_classifier.pkl')
    print("Model training completed and saved as 'data/voting_classifier.pkl'.")

if __name__ == "__main__":
    train_model()

