import pandas as pd
import os

def add_features(df):
    """Add new features to the dataset."""
    df["Venue_Code"] = df["Home"].astype("category").cat.codes
    df["Opp_Code"] = df["Away"].astype("category").cat.codes
    df["Day_Code"] = pd.to_datetime(df["Date"]).dt.dayofweek
    df["Rolling_HomeGoals"] = df.groupby("Home")["GF"].transform(lambda x: x.rolling(3, min_periods=1).mean())
    df["Rolling_AwayGoals"] = df.groupby("Away")["GA"].transform(lambda x: x.rolling(3, min_periods=1).mean())
    df["Venue_Opp_Interaction"] = df["Venue_Code"] * df["Opp_Code"]
    df["Decayed_Rolling_HomeGoals"] = df.groupby('Home')['GF'].transform(lambda x: x.ewm(alpha=0.9).mean())
    df["Decayed_Rolling_AwayGoals"] = df.groupby('Away')['GA'].transform(lambda x: x.ewm(alpha=0.9).mean())
    df["Home_Advantage"] = df.groupby('Home')['GF'].transform(lambda x: x.rolling(5, min_periods=1).mean()) - df.groupby('Away')['GA'].transform(lambda x: x.rolling(5, min_periods=1).mean())
    df['Home_Streak_Wins'] = df.groupby('Home')['W'].transform(lambda x: (x == 1).rolling(window=5, min_periods=1).sum())
    df['Away_Streak_Losses'] = df.groupby('Away')['L'].transform(lambda x: (x == 1).rolling(window=5, min_periods=1).sum())

    return df

def apply_feature_engineering(input_path, output_path):
    """Apply feature engineering and save the feature-engineered dataset."""
    print(f"Loading preprocessed data from {input_path}...")
    df = pd.read_csv(input_path)

    print("Applying feature engineering...")
    df = add_features(df)

    print(f"Saving feature-engineered data to {output_path}...")
    df.to_csv(output_path, index=False)

    print("First few rows of feature-engineered data:")
    print(df.head())

if __name__ == "__main__":
    input_file = 'data/preprocessed_data.csv'
    output_file = 'data/fe_combined_matches.csv'
    apply_feature_engineering(input_file, output_file)
