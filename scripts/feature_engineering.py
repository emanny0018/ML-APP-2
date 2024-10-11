import pandas as pd
import numpy as np

def add_features(df):
    df["Venue_Code"] = df["Team"].astype("category").cat.codes
    df["Opp_Code"] = df["Opposition"].astype("category").cat.codes
    df["Day_Code"] = pd.to_datetime(df["Date"]).dt.dayofweek
    df["Rolling_GF"] = df.groupby("Team")["GF"].transform(lambda x: x.rolling(3, min_periods=1).mean())
    df["Rolling_GA"] = df.groupby("Team")["GA"].transform(lambda x: x.rolling(3, min_periods=1).mean())
    df["Venue_Opp_Interaction"] = df["Venue_Code"] * df["Opp_Code"]
    df["Decayed_Rolling_GF"] = df.groupby('Team')['GF'].transform(lambda x: x.ewm(alpha=0.9).mean())
    df["Decayed_Rolling_GA"] = df.groupby('Team')['GA'].transform(lambda x: x.ewm(alpha=0.9).mean())
    df["Home_Advantage"] = df.groupby("Team")["GF"].transform(lambda x: x.rolling(5, min_periods=1).mean()) - \
                           df.groupby("Opposition")["GA"].transform(lambda x: x.rolling(5, min_periods=1).mean())

    return df

def apply_feature_engineering():
    df = pd.read_csv('data/combined_matches.csv')
    df = add_features(df)
    df.to_csv('data/fe_combined_matches.csv', index=False)
    print("Feature engineering completed and file saved.")

if __name__ == "__main__":
    apply_feature_engineering()

