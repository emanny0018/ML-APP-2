import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from joblib import load
import os

def evaluate_model():
    """Load the test data and the trained model, evaluate and save results."""
    
    # Load the feature-engineered dataset
    print("Loading feature-engineered dataset...")
    data = pd.read_csv('data/fe_combined_matches.csv')

    # Define predictors and target variable
    predictors = [
        "Venue_Code", "Opp_Code", "Day_Code", "Rolling_HomeGoals", "Rolling_AwayGoals", 
        "Venue_Opp_Interaction", "Decayed_Rolling_HomeGoals", "Decayed_Rolling_AwayGoals", 
        "Home_Advantage", "Home_Streak_Wins", "Away_Streak_Losses"
    ]
    target = "Target"

    # Split the data into training and test sets
    _, test = train_test_split(data, test_size=0.2, random_state=42, stratify=data[target])

    # Load the trained model
    print("Loading trained model from 'data/voting_classifier.pkl'...")
    model = load('data/voting_classifier.pkl')

    # Make predictions on the test set
    print("Making predictions on the test set...")
    test_predictions = model.predict(test[predictors])

    # Evaluate the model performance
    accuracy = accuracy_score(test[target], test_predictions) * 100
    conf_matrix = confusion_matrix(test[target], test_predictions)
    class_report = classification_report(test[target], test_predictions)

    # Print results
    print(f"Accuracy: {accuracy:.2f}%")
    print("Confusion Matrix:")
    print(conf_matrix)
    print("Classification Report:")
    print(class_report)

    # Save evaluation results to a file
    evaluation_results = {
        "accuracy": accuracy,
        "confusion_matrix": conf_matrix.tolist(),
        "classification_report": class_report
    }

    results_df = pd.DataFrame({
        "Accuracy": [accuracy],
        "Confusion_Matrix": [conf_matrix.tolist()],
        "Classification_Report": [class_report]
    })

    evaluation_file = 'data/evaluation_results.csv'
    print(f"Saving evaluation results to {evaluation_file}...")
    results_df.to_csv(evaluation_file, index=False)

if __name__ == "__main__":
    evaluate_model()
