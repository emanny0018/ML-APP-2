import pandas as pd
import os

def add_features(df):
    """Add features dynamically based on available columns."""
    
    # Step 1: Add features only if columns are available
    if 'Home' in df.columns and 'Away' in df.columns:
        df["Venue_Code"] = df["Home"].astype("category").cat.codes
        df["Opp_Code"] = df["Away"].astype("category").cat.codes
    else:
        print("Warning: 'Home' or 'Away' columns missing. Skipping related features.")

    if 'Date' in df.columns:
        df["Day_Code"] = pd.to_datetime(df["Date"]).dt.dayofweek
    else:
        print("Warning: 'Date' column missing. Skipping day-based features.")

    if 'HomeGoals' in df.columns:
        df["Rolling_HomeGoals"] = df.groupby("Home")["HomeGoals"].transform(lambda x: x.rolling(3, min_periods=1).mean())
        df["Decayed_Rolling_HomeGoals"] = df.groupby('Home')["HomeGoals"].transform(lambda x: x.ewm(alpha=0.9).mean())
    else:
        print("Warning: 'HomeGoals' column missing. Skipping goal-related features.")

    if 'AwayGoals' in df.columns:
        df["Rolling_AwayGoals"] = df.groupby("Away")["AwayGoals"].transform(lambda x: x.rolling(3, min_periods=1).mean())
        df["Decayed_Rolling_AwayGoals"] = df.groupby('Away')["AwayGoals"].transform(lambda x: x.ewm(alpha=0.9).mean())
    else:
        print("Warning: 'AwayGoals' column missing. Skipping goal-related features.")

    # Optional: Home advantage (difference between rolling averages of home and away goals)
    if 'HomeGoals' in df.columns and 'AwayGoals' in df.columns:
        df["Home_Advantage"] = df["Rolling_HomeGoals"] - df["Rolling_AwayGoals"]
    else:
        print("Warning: 'HomeGoals' or 'AwayGoals' columns missing. Skipping Home Advantage feature.")

    return df

def apply_feature_engineering(input_path, output_path):
    """Load the preprocessed data, apply feature engineering, and save the output."""
    
    # Step 1: Load the preprocessed data
    print(f"Loading preprocessed data from {input_path}...")
    df = pd.read_csv(input_path)

    # Step 2: Apply feature engineering
    df = add_features(df)

    # Step 3: Save the feature-engineered data
    print(f"Saving feature-engineered data to {output_path}...")
    df.to_csv(output_path, index=False)

    # Step 4: Show a preview of the feature-engineered data
    print("\nFirst few rows of feature-engineered data:")
    print(df.head())

if __name__ == "__main__":
    # Define input and output paths
    input_file = os.getenv('INPUT_DATA', 'data/preprocessed_data.csv')  # Preprocessed data file path
    output_file = os.getenv('OUTPUT_DATA', 'data/fe_combined_matches.csv')  # Output file for feature-engineered data

    # Apply feature engineering
    apply_feature_engineering(input_file, output_file)

