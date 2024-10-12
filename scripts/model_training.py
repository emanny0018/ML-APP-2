import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
import joblib

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
