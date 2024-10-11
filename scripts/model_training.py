import os
import pandas as pd
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.impute import SimpleImputer
import joblib

def train_model(input_path, model_output_path):
    """Train the model using the feature-engineered data and save the trained model."""
    
    # Step 1: Load the feature-engineered data
    print(f"Loading feature-engineered data from {input_path}...")
    data = pd.read_csv(input_path)

    # Define predictors based on the feature-engineering process
    predictors = [
        "Venue_Code", "Opp_Code", "Day_Code", "Rolling_HomeGoals", "Rolling_AwayGoals", 
        "Venue_Opp_Interaction", "Decayed_Rolling_HomeGoals", "Decayed_Rolling_AwayGoals", 
        "Home_Advantage", "Home_Streak_Wins", "Away_Streak_Losses"
    ]

    # Step 2: Split data into training and test sets
    train, _ = train_test_split(data, test_size=0.2, random_state=42, stratify=data["Target"])

    # Step 3: Define pipelines for both XGBoost and RandomForest
    imputer = SimpleImputer(strategy='mean')

    # XGBoost pipeline
    xgb_pipeline = Pipeline(steps=[
        ('imputer', imputer),
        ('classifier', XGBClassifier(random_state=42))
    ])

    # RandomForest pipeline
    rf_pipeline = Pipeline(steps=[
        ('imputer', imputer),
        ('classifier', RandomForestClassifier(random_state=42, n_estimators=200, max_depth=7))
    ])

    # Step 4: Ensemble using Voting Classifier
    voting_classifier = VotingClassifier(
        estimators=[('xgb', xgb_pipeline), ('rf', rf_pipeline)],
        voting='soft'
    )

    # Step 5: Train the ensemble model
    print("Training the voting classifier...")
    voting_classifier.fit(train[predictors], train["Target"])

    # Step 6: Cross-validation for accuracy
    cv_scores = cross_val_score(voting_classifier, train[predictors], train["Target"], cv=5, scoring='accuracy')
    print(f"Cross-Validation Accuracy Scores: {cv_scores}")
    print(f"Mean Cross-Validation Accuracy: {cv_scores.mean()}")

    # Step 7: Save the trained model
    joblib.dump(voting_classifier, model_output_path)
    print(f"Model saved to {model_output_path}")

if __name__ == "__main__":
    # Paths to input feature-engineered data and output model
    input_file = 'data/fe_combined_data.csv'
    model_output_file = 'data/voting_classifier.pkl'

    # Train the model and save it
    train_model(input_file, model_output_file)

