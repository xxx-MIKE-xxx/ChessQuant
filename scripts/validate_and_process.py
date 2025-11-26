import json
import pandas as pd
import numpy as np
from pathlib import Path

# --- CONFIG ---
# Adjust these paths to match your Colab/Repo structure
INPUT_JSON = "julio_amigo_dos_games_full.json" 
OUTPUT_CSV = "games_enriched_clean.csv"
HERO_USER = "julio_amigo_dos"
SESSION_GAP_MINUTES = 30

def validate_and_process():
    print(f"--- 1. Loading Raw Data from {INPUT_JSON} ---")
    with open(INPUT_JSON, 'r') as f:
        data = json.load(f)
    
    # Flatten JSON
    df = pd.json_normalize(data, sep='_')
    print(f"Loaded {len(df)} raw games.")

    # --- 2. Feature Extraction ---
    print("\n--- 2. Extracting Features ---")
    
    # A. Identify Hero Color & Context
    df['user_color'] = np.where(df['players_white_user_name'] == HERO_USER, 'white', 'black')
    
    # NEW: Extract Metadata (Rating, Opening, etc.)
    # ---------------------------------------------------------
    df['rated'] = df.get('rated', False)
    df['speed'] = df.get('speed', 'unknown')
    df['perf'] = df.get('perf', 'unknown')
    df['source'] = df.get('source', 'unknown')
    df['opening_eco'] = df.get('opening_eco', 'unknown')
    df['opening_name'] = df.get('opening_name', 'unknown')
    
    # Ratings
    df['my_rating'] = np.where(df['user_color'] == 'white', 
                               df.get('players_white_rating', np.nan), 
                               df.get('players_black_rating', np.nan))
    
    df['opp_rating'] = np.where(df['user_color'] == 'white', 
                                df.get('players_black_rating', np.nan), 
                                df.get('players_white_rating', np.nan))
                                
    df['rating_diff'] = np.where(df['user_color'] == 'white', 
                                 df.get('players_white_ratingDiff', 0), 
                                 df.get('players_black_ratingDiff', 0))
    # ---------------------------------------------------------

    # B. Extract Analysis
    df['my_acpl'] = np.where(df['user_color'] == 'white', 
                             df.get('players_white_analysis_acpl', np.nan), 
                             df.get('players_black_analysis_acpl', np.nan))
    
    df['my_blunder_count'] = np.where(df['user_color'] == 'white', 
                                      df.get('players_white_analysis_blunder', np.nan), 
                                      df.get('players_black_analysis_blunder', np.nan))

    # C. Time Calculation
    df['created_at'] = pd.to_datetime(df['createdAt'], unit='ms')
    df['last_move_at'] = pd.to_datetime(df['lastMoveAt'], unit='ms')
    df['game_duration_sec'] = (df['last_move_at'] - df['created_at']).dt.total_seconds()
    
    df['moves_list'] = df['moves'].fillna("").apply(lambda x: x.split(" "))
    df['move_count'] = df['moves_list'].apply(lambda x: len(x) // 2)
    df['my_avg_secs_per_move'] = df['game_duration_sec'] / df['move_count'].replace(0, 1)

    # D. Result
    conditions = [
        df['winner'] == df['user_color'],
        df['winner'].isna()
    ]
    choices = [1.0, 0.5]
    df['result'] = np.select(conditions, choices, default=0.0)

    # --- 3. Session Features ---
    print("\n--- 3. Calculating Session Features ---")
    df = df.sort_values('created_at').reset_index(drop=True)
    df['time_diff'] = df['created_at'].diff()
    df['is_new_session'] = (df['time_diff'] > pd.Timedelta(minutes=SESSION_GAP_MINUTES)) | (df['time_diff'].isna())
    df['session_id'] = df['is_new_session'].cumsum()
    df['games_played'] = df.groupby('session_id').cumcount() + 1
    
    session_start_speed = df.groupby('session_id')['my_avg_secs_per_move'].transform('first')
    df['speed_vs_start'] = df['my_avg_secs_per_move'] / (session_start_speed + 0.001)

    # --- 4. Cleaning ---
    print("\n--- 4. Final Cleaning ---")
    # Drop missing analysis
    df_clean = df.dropna(subset=['my_acpl', 'my_blunder_count'])
    
    # Drop short/empty games
    df_clean = df_clean[df_clean['move_count'] >= 3]
    
    # SELECT FINAL COLUMNS (Updated to include Metadata)
    final_cols = [
        'id', 'created_at', 'session_id', 'games_played', 'result',
        'my_acpl', 'my_blunder_count', 'my_avg_secs_per_move', 'speed_vs_start',
        # NEW FIELDS
        'my_rating', 'opp_rating', 'rating_diff', 
        'opening_eco', 'opening_name', 'rated', 'speed', 'perf'
    ]
    
    # Ensure all exist before saving (handle missing gracefully)
    existing_cols = [c for c in final_cols if c in df_clean.columns]
    
    df_clean[existing_cols].to_csv(OUTPUT_CSV, index=False)
    print(f"\n✅ SUCCESS! Saved {len(df_clean)} validated games to {OUTPUT_CSV}")
    print(f"Columns included: {existing_cols}")

if __name__ == "__main__":
    validate_and_process()