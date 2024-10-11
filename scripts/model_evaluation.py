import os
import pandas as pd
from joblib import load
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def load_data(file_path):
    """Load the dataset from a CSV file."""
    return pd.read_csv(file_path)

def evaluate_model(model, test_data, predictors):
    """Evaluate the model using accuracy, confusion matrix, and classification report."""
    
    # Predict on the test data
    test_predictions = model.predict(test_data[predictors])
    
    # Calculate accuracy
    accuracy = accuracy_score(test_data["Target"], test_predictions)
    accuracy_percentage = accuracy * 100

    # Generate confusion matrix and classification report
    conf_matrix = confusion_matrix(test_data["Target"], test_predictions)
    class_report = classification_report(test_data["Target"], test_predictions)

    # Output results to console
    print(f"Test Set Accuracy: {accuracy_percentage:.2f}%")
    print("Confusion Matrix:")
    print(conf_matrix)
    print("Classification Report:")
    print(class_report)

    # Save results to a file
    results_path = "data/evaluation_results.txt"
    with open(results_path, "w") as f:
        f.write(f"Test Set Accuracy: {accuracy_percentage:.2f}%\n")
        f.write("Confusion Matrix:\n")
        f.write(f"{conf_matrix}\n")
        f.write("Classification Report:\n")
        f.write(f"{class_report}\n")

    print(f"Evaluation results saved to {results_path}.")

if __name__ == "__main__":
    # Load the trained model
    model_path = 'data/voting_classifier.pkl'
    model = load(model_path)

    # Load the feature-engineered data
    df = load_data('data/fe_combined_data.csv')

    # Split the data into train and test sets
    _, test = train_test_split(df, test_size=0.2, random_state=42, stratify=df["Target"])

    # Define predictors based on your feature engineering
    advanced_predictors = [
        "Venue_Code", "Opp_Code", "Day_Code", "Rolling_HomeGoals", "Rolling_AwayGoals", 
        "Venue_Opp_Interaction", "Decayed_Rolling_HomeGoals", "Decayed_Rolling_AwayGoals", 
        "Home_Advantage", "Home_Streak_Wins", "Away_Streak_Losses"
    ]

    # Evaluate the model on the test set
    evaluate_model(model, test, advanced_predictors)

