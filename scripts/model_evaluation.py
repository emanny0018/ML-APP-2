import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from joblib import load
import os

def evaluate_model():
    """Load the test data and the trained model, evaluate and save results."""
    
    # Step 1: Load the feature-engineered dataset
    print("Loading feature-engineered dataset...")
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
        raise ValueError("No valid predictors found for model evaluation.")
    
    print(f"Available predictors for evaluation: {available_predictors}")

    # Step 4: Define target
    target = "Target"
    if target not in data.columns:
        raise ValueError(f"Target column '{target}' is missing in the dataset.")

    # Step 5: Split the data into training and test sets
    _, test = train_test_split(data, test_size=0.2, random_state=42, stratify=data[target])

    # Step 6: Load the trained model
    print("Loading trained model from 'data/voting_classifier.pkl'...")
    model = load('data/voting_classifier.pkl')

    # Step 7: Make predictions on the test set
    print("Making predictions on the test set...")
    test_predictions = model.predict(test[available_predictors])

    # Step 8: Evaluate the model performance
    accuracy = accuracy_score(test[target], test_predictions) * 100
    conf_matrix = confusion_matrix(test[target], test_predictions)
    class_report = classification_report(test[target], test_predictions)

    # Step 9: Print results
    print(f"Accuracy: {accuracy:.2f}%")
    print("Confusion Matrix:")
    print(conf_matrix)
    print("Classification Report:")
    print(class_report)

if __name__ == "__main__":
    evaluate_model()

