import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingClassifier
from xgboost import XGBClassifier
<<<<<<< HEAD
from sklearn.ensemble import RandomForestClassifier
=======
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
>>>>>>> 5bb6ce4 (updated scripts)
import joblib
import os

<<<<<<< HEAD
def train_model():
    # Load the feature-engineered dataset
    data = pd.read_csv('data/fe_combined_matches.csv')

    # Define predictors and target variable
    predictors = ["Venue_Code", "Opp_Code", "Day_Code", "Rolling_HomeGoals", "Rolling_AwayGoals", 
                  "Venue_Opp_Interaction", "Decayed_Rolling_HomeGoals", "Decayed_Rolling_AwayGoals", 
                  "Home_Advantage", "Home_Streak_Wins", "Away_Streak_Losses"]
    target = "Target"

    # Split the data into train and test sets
    train, test = train_test_split(data, test_size=0.2, random_state=42, stratify=data[target])

    # Train Voting Classifier (XGBoost and RandomForest)
    xgb = XGBClassifier(random_state=42)
    rf = RandomForestClassifier(random_state=42)

    model = VotingClassifier(estimators=[('xgb', xgb), ('rf', rf)], voting='soft')
    model.fit(train[predictors], train[target])

    # Save the trained model
    joblib.dump(model, 'data/voting_classifier.pkl')

    print("Model training completed and saved as 'data/voting_classifier.pkl'.")

if __name__ == "__main__":
    train_model()
=======
# Load the feature-engineered dataset
data = pd.read_csv('data/fe_combined_matches.csv')

# Define the predictors based on feature engineering
predictors = [
    "Venue_Code", "Opp_Code", "Day_Code", "Rolling_HomeGoals", "Rolling_AwayGoals", 
    "Venue_Opp_Interaction", "Decayed_Rolling_HomeGoals", "Decayed_Rolling_AwayGoals", 
    "Home_Advantage", "Home_Streak_Wins", "Away_Streak_Losses"
]

# Split the data into train and test sets
train, _ = train_test_split(data, test_size=0.2, random_state=42, stratify=data["Target"])

# Define an imputer to handle NaN values
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

# Ensemble using Voting Classifier
voting_classifier = VotingClassifier(
    estimators=[('xgb', xgb_pipeline), ('rf', rf_pipeline)],
    voting='soft'
)

# Fit the ensemble model
voting_classifier.fit(train[predictors], train["Target"])

# Save the model to a file
if not os.path.exists('data'):
    os.makedirs('data')
model_path = 'data/voting_classifier.pkl'
joblib.dump(voting_classifier, model_path)
print(f"Model saved to {model_path}")

>>>>>>> 5bb6ce4 (updated scripts)
