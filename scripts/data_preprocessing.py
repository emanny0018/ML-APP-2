import pandas as pd
import os

def preprocess_data(input_path, output_path):
    """Load the raw data, clean it, handle missing values, and save the preprocessed data."""
    
    # Step 1: Load the raw data
    print(f"Loading raw data from {input_path}...")
    df = pd.read_csv(input_path, low_memory=False)  # Added low_memory=False to handle mixed types warning

    # Step 2: Check for required columns
    required_columns = ['Home', 'Away', 'Date', 'HomeGoals', 'AwayGoals', 'Target']
    print(f"Columns in dataset: {list(df.columns)}")  # Print all available columns in the dataset
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        raise KeyError(f"Missing required columns: {missing_columns}")

    # Step 3: Drop unnecessary or incomplete columns (if needed)
    unnecessary_columns = ['Odds_Draw', 'Odds_Away_Win']
    df.drop(columns=[col for col in unnecessary_columns if col in df.columns], inplace=True, errors='ignore')

    # Step 4: Handle missing values (You can customize how to handle missing data)
    df.fillna(0, inplace=True)

    # Step 5: Convert 'Date' column to datetime format
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

    # Step 6: Save the preprocessed data
    print(f"Saving preprocessed data to {output_path}...")
    df.to_csv(output_path, index=False)

    # Step 7: Print a preview of the preprocessed data
    print("First few rows of preprocessed data:")
    print(df.head())

if __name__ == "__main__":
    # Define paths to input and output data
    input_file = os.getenv('INPUT_DATA', 'data/epl_combined_data.csv')  # Path to raw data
    output_file = 'data/preprocessed_data.csv'  # Path to save preprocessed data

    # Run preprocessing
    preprocess_data(input_file, output_file)

