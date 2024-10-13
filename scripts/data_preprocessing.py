import pandas as pd
import os

def preprocess_data(input_path, output_path):
    """Load the raw data, clean it, handle missing values, and save the preprocessed data."""
    
    # Step 1: Load the raw data
    print(f"Loading raw data from {input_path}...")
    df = pd.read_csv(input_path, low_memory=False)  # low_memory=False to handle mixed types warning

    # Step 2: Print available columns
    available_columns = df.columns.tolist()
    print(f"\nAvailable columns in the dataset:\n{available_columns}")

    # Step 3: Define columns required for feature engineering
    required_for_features = ['Home', 'Away', 'Date', 'HomeGoals', 'AwayGoals']
    available_for_features = [col for col in required_for_features if col in df.columns]

    # Step 4: Print which columns will be used for feature engineering
    if available_for_features:
        print(f"\nColumns that will be used for feature engineering:\n{available_for_features}")
    else:
        print("\nWarning: None of the required columns for feature engineering are present in the dataset.")
    
    missing_columns = [col for col in required_for_features if col not in df.columns]
    if missing_columns:
        print(f"\nWarning: The following required columns for feature engineering are missing:\n{missing_columns}")
    
    # Step 5: Drop unnecessary or incomplete columns (if present)
    unnecessary_columns = ['Odds_Draw', 'Odds_Away_Win']
    df.drop(columns=[col for col in unnecessary_columns if col in df.columns], inplace=True, errors='ignore')

    # Step 6: Handle missing values (filling missing values with 0 as a placeholder)
    df.fillna(0, inplace=True)

    # Step 7: Convert 'Date' column to datetime format if present
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        print("\n'Date' column converted to datetime format.")

    # Step 8: Save the preprocessed data
    print(f"\nSaving preprocessed data to {output_path}...")
    df.to_csv(output_path, index=False)

    # Step 9: Print a preview of the preprocessed data
    print("\nFirst few rows of preprocessed data:")
    print(df.head())

if __name__ == "__main__":
    # Define paths to input and output data
    input_file = os.getenv('INPUT_DATA', 'data/epl_combined_data.csv')  # Path to raw data
    output_file = 'data/preprocessed_data.csv'  # Path to save preprocessed data

    # Run preprocessing
    preprocess_data(input_file, output_file)

