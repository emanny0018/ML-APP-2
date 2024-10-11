import pandas as pd
import os

def preprocess_data(input_path, output_path):
    """Load the raw data, clean it, handle missing values, and save the preprocessed data."""
    
    # Step 1: Load the raw data
    print(f"Loading raw data from {input_path}...")
    df = pd.read_csv(input_path)

    # Step 2: Drop unnecessary or incomplete columns
    # Drop columns such as Odds_Draw, Odds_Away_Win (customize based on your needs)
    df.drop(columns=['Odds_Draw', 'Odds_Away_Win'], inplace=True, errors='ignore')

    # Step 3: Handle missing values (You can also customize how to handle missing data)
    df.fillna(0, inplace=True)

    # Step 4: Convert 'Date' column to datetime format
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

    # Step 5: Save the preprocessed data
    print(f"Saving preprocessed data to {output_path}...")
    df.to_csv(output_path, index=False)

    # Step 6: Print a preview of the preprocessed data
    print("First few rows of preprocessed data:")
    print(df.head())

if __name__ == "__main__":
    # Define paths to input and output data
    input_file = os.getenv('INPUT_DATA', 'data/epl_combined_data.csv')  # Path to raw data
    output_file = 'data/preprocessed_data.csv'  # Path to save preprocessed data

    # Run preprocessing
    preprocess_data(input_file, output_file)

