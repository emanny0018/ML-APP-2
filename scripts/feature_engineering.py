import pandas as pd
import os

def add_features(df):
    """Feature engineering: Add new features to the preprocessed data."""
    df["Venue_Code"] = df["Home"].astype("category").cat.codes
    df["Opp_Code"] = df["Away"].astype("category").cat.codes
    df["Day_Code"] = pd.to_datetime(df["Date"]).dt.dayofweek
    df["Rolling_HomeGoals"] = df.groupby("Home")["GF"].transform(lambda x: x.rolling(3, min_periods=1).mean())
    df["Rolling_AwayGoals"] = df.groupby("Away")["GA"].transform(lambda x: x.rolling(3, min_periods=1).mean())
    df["Venue_Opp_Interaction"] = df["Venue_Code"] * df["Opp_Code"]
    df["Decayed_Rolling_HomeGoals"] = df.groupby('Home')['GF'].transform(lambda x: x.ewm(alpha=0.9).mean())
    df["Decayed_Rolling_AwayGoals"] = df.groupby('Away')['GA'].transform(lambda x: x.ewm(alpha=0.9).mean())
    df["Home_Advantage"] = df.groupby('Home')['GF'].transform(lambda x: x.rolling(5, min_periods=1).mean()) - df.groupby('Away')['GA'].transform(lambda x: x.rolling(5, min_periods=1).mean())
    
    return df

def apply_feature_engineering(input_path, output_path):
    """Load preprocessed data, apply feature engineering, and save the result."""
    # Step 1: Load the preprocessed data
    print(f"Loading preprocessed data from {input_path}...")
    df = pd.read_csv(input_path)

    # Step 2: Add features to the data
    df = add_features(df)

    # Step 3: Save the feature-engineered data
    print(f"Saving feature-engineered data to {output_path}...")
    df.to_csv(output_path, index=False)

    # Step 4: Print first few rows of the feature-engineered data
    print("First few rows of feature-engineered data:")
    print(df.head())

if __name__ == "__main__":
    # Paths to input preprocessed data and output feature-engineered data
    input_file = 'data/preprocessed_data.csv'
    output_file = 'data/fe_combined_data.csv'

    # Apply feature engineering
    apply_feature_engineering(input_file, output_file)

