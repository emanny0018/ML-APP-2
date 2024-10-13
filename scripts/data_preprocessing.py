import os
import pandas as pd

def preprocess_data(input_path, output_path):
    """Load the raw data, clean it, handle missing values, and save the preprocessed data."""
    
    # Step 1: Load the raw data
    print(f"Loading raw data from {input_path}...")
    df = pd.read_csv(input_path, low_memory=False)

<<<<<<< HEAD
    # Step 2: Check if the required columns exist
    required_columns = ['Home', 'Away', 'Date', 'HomeGoals', 'AwayGoals', 'Target']
    
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise KeyError(f"Missing required columns: {missing_columns}")

    # Step 3: Handle missing values
=======
    # Step 2: Ensure required columns are present
    required_columns = ['Home', 'Away', 'Date', 'HomeGoals', 'AwayGoals', 'Target']
    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        raise KeyError(f"Missing required columns: {missing_columns}")

    # Step 3: Drop unnecessary or incomplete columns
    df.drop(columns=['Odds_Draw', 'Odds_Away_Win'], inplace=True, errors='ignore')

    # Step 4: Handle missing values (You can also customize how to handle missing data)
>>>>>>> 5bb6ce4 (updated scripts)
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
    input_file = os.getenv('INPUT_DATA', 'data/epl_combined_data.csv')  # Path to raw data
    output_file = 'data/preprocessed_data.csv'  # Path to save preprocessed data

    # Run preprocessing
    preprocess_data(input_file, output_file)
<<<<<<< HEAD
    print(f"Preprocessing completed. Saved to {output_file}.")
=======
    print(f"Preprocessing completed. Data saved to {output_file}.")

>>>>>>> 5bb6ce4 (updated scripts)
