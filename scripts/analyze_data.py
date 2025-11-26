import pandas as pd
import json
import numpy as np
from datetime import datetime

# Configuration
INPUT_FILE = "data/raw_data/julio_amigo_dos_games_full.json"

def analyze_chess_data():
    print(f"--- Loading data from {INPUT_FILE} ---")
    try:
        with open(INPUT_FILE, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print("File not found. Please check the path.")
        return

    # Flatten the nested JSON structure
    # We specifically target nested fields that are critical for Tilt Detection
    df = pd.json_normalize(
        data, 
        sep='_'
    )
    
    print(f"Total Games Loaded: {len(df)}")
    print(f"Total Raw Columns: {len(df.columns)}")
    
    # --- 1. Completeness Analysis ---
    print("\n[1] Missing Value Analysis (Critical Columns)")
    
    # Define columns that MUST be present for the model to work
    critical_cols = [
        'id', 'rated', 'variant', 'speed', 'createdAt', 'status', 
        'players_white_rating', 'players_black_rating', 'moves'
    ]
    
    # Define columns that are valuable but might be missing (e.g., engine analysis)
    optional_cols = [
        'players_white_analysis_acpl', 'players_black_analysis_acpl',
        'opening_eco', 'clock_totalTime'
    ]
    
    all_cols_to_check = critical_cols + optional_cols
    
    # Check which of these actually exist in the dataframe
    existing_cols = [c for c in all_cols_to_check if c in df.columns]
    missing_cols = [c for c in all_cols_to_check if c not in df.columns]
    
    if missing_cols:
        print(f"WARNING: The following expected columns are completely missing from the data: {missing_cols}")

    # Calculate missing percentages for existing columns
    missing_counts = df[existing_cols].isnull().sum()
    missing_pct = (missing_counts / len(df)) * 100
    
    missing_df = pd.DataFrame({'Missing Count': missing_counts, 'Percentage': missing_pct})
    print(missing_df[missing_df['Missing Count'] > 0].sort_values('Percentage', ascending=False).to_string())
    
    if missing_df['Missing Count'].sum() == 0:
        print("Perfect! No missing values in critical/optional columns found.")

    # --- 2. Data Integrity & Edge Cases ---
    print("\n[2] Integrity Checks")
    
    # Check A: Empty Moves
    empty_moves = df[df['moves'] == ''].shape[0]
    print(f"- Games with empty move list (aborted/immediate resign): {empty_moves}")
    
    # Check B: Timestamps
    # Ensure createdAt is reasonable (not in the future, not before Lichess existed)
    # Lichess uses milliseconds
    dates = pd.to_datetime(df['createdAt'], unit='ms')
    print(f"- Date Range: {dates.min()} to {dates.max()}")
    
    if dates.isnull().any():
        print("  CRITICAL: Some dates could not be parsed.")
        
    # Check C: Ratings
    # Ensure ratings are positive integers
    if 'players_white_rating' in df.columns:
        invalid_ratings = df[ (df['players_white_rating'] < 0) | (df['players_white_rating'] > 4000) ]
        print(f"- Games with suspicious White ratings (<0 or >4000): {len(invalid_ratings)}")
        
    # --- 3. Variant Distribution ---
    print("\n[3] Variant Distribution")
    print(df['variant'].value_counts().to_string())
    
    # --- 4. Speed Distribution ---
    print("\n[4] Speed Distribution")
    print(df['speed'].value_counts().to_string())

    # --- 5. Engine Analysis Availability ---
    # For tilt detection, ACPL (Average Centipawn Loss) is a huge feature.
    # We need to know how many games actually have this data.
    print("\n[5] Engine Analysis Availability")
    if 'players_white_analysis_acpl' in df.columns:
        has_acpl = df['players_white_analysis_acpl'].notnull().sum()
        print(f"Games with White ACPL: {has_acpl} ({has_acpl/len(df)*100:.2f}%)")
    else:
        print("No ACPL data found for White.")

if __name__ == "__main__":
    analyze_chess_data()