import pandas as pd
import os

def add_features(df):
    """Add features based on your feature engineering process."""
    
    # Check if necessary columns exist before proceeding
    required_columns = ['Home', 'Away', 'Date', 'HomeGoals', 'AwayGoals', 'Target']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        raise KeyError(f"Missing columns in dataset: {missing_columns}. Please check the dataset.")

    # Add features based on your feature engineering process
    df["Venue_Code"] = df["Home"].astype("category").cat.codes
    df["Opp_Code"] = df["Away"].astype("category").cat.codes
    df["Day_Code"] = pd.to_datetime(df["Date"]).dt.dayofweek

    # Rolling averages for goals
    df["Rolling_HomeGoals"] = df.groupby("Home")["HomeGoals"].transform(lambda x: x.rolling(3, min_periods=1).mean())
    df["Rolling_AwayGoals"] = df.groupby("Away")["AwayGoals"].transform(lambda x: x.rolling(3, min_periods=1).mean())
    
    # Interaction between venue and opponent
    df["Venue_Opp_Interaction"] = df["Venue_Code"] * df["Opp_Code"]

    # Exponentially weighted mean for decayed goals
    df["Decayed_Rolling_HomeGoals"] = df.groupby('Home')['HomeGoals'].transform(lambda x: x.ewm(alpha=0.9).mean())
    df["Decayed_Rolling_AwayGoals"] = df.groupby('Away')['AwayGoals'].transform(lambda x: x.ewm(alpha=0.9).mean())

    # Home Advantage: Difference in rolling averages between home and away
    df["Home_Advantage"] = df.groupby('Home')['HomeGoals'].transform(lambda x: x.rolling(5, min_periods=1).mean()) - df.groupby('Away')['AwayGoals'].transform(lambda x: x.rolling(5, min_periods=1).mean())

    # Streaks for wins and losses
    df['Home_Streak_Wins'] = df.groupby('Home')['Target'].transform(lambda x: (x == 0).rolling(window=5, min_periods=1).sum())
    df['Away_Streak_Losses'] = df.groupby('Away')['Target'].transform(lambda x: (x == 1).rolling(window=5, min_periods=1).sum())

    return df

def apply_feature_engineering(input_path, output_path):
    """Load the preprocessed data, apply feature engineering, and save the output."""
    
    # Step 1: Load the preprocessed data
    print(f"Loading preprocessed data from {input_path}...")
<<<<<<< HEAD
    df = pd.read_csv(input_path)
    print(f"Columns in the dataset: {df.columns.tolist()}")
=======
    df = pd.read_csv(input_path, low_memory=False)
>>>>>>> 5bb6ce4 (updated scripts)
    
    # Step 2: Check if the necessary columns are present
    print("Applying feature engineering...")
    
    # Step 3: Add engineered features
    df = add_features(df)

    # Step 4: Save the feature-engineered data
    print(f"Saving feature-engineered data to {output_path}...")
    df.to_csv(output_path, index=False)

    # Step 5: Show a preview of the feature-engineered data
    print("\nFirst few rows of feature-engineered data:")
    print(df.head())

if __name__ == "__main__":
    # Define input and output paths
    input_file = os.getenv('INPUT_DATA', 'data/preprocessed_data.csv')  # Preprocessed data file path
    output_file = os.getenv('OUTPUT_DATA', 'data/fe_combined_matches.csv')  # Output file for feature-engineered data

    # Apply feature engineering
    apply_feature_engineering(input_file, output_file)
    print(f"Feature engineering completed. Saved to {output_file}.")
<<<<<<< HEAD
=======

>>>>>>> 5bb6ce4 (updated scripts)
