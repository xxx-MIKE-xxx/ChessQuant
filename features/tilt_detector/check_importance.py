import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    roc_auc_score, 
    roc_curve,
    accuracy_score
)

# --- CONFIG ---
INPUT_FILE = "data/formatted_data/games_enriched_full.csv"
RANDOM_STATE = 42

def check_feature_importance():
    print(f"--- Loading Data from {INPUT_FILE} ---")
    try:
        df = pd.read_csv(INPUT_FILE)
    except FileNotFoundError:
        print(f"Error: {INPUT_FILE} not found.")
        return

    # -------------------------------------------------------
    # 1. FIX: ROBUST COLUMN HANDLING
    # -------------------------------------------------------
    # The error "KeyError: result" happens if the column is named 'pl' or missing.
    
    if 'result' not in df.columns:
        print("⚠️ 'result' column missing. Attempting to derive...")
        
        if 'pl' in df.columns:
            # Assumption: 'pl' is Rating Change or Score. 
            # If pl > 0 it's likely a Win.
            print("   -> Found 'pl' column. Deriving 'result' from it.")
            df['result'] = np.where(df['pl'] > 0, 1.0, 0.0)
            
        elif 'winner' in df.columns and 'user_color' in df.columns:
            print("   -> Found 'winner'/'user_color'. Deriving 'result'.")
            conditions = [
                df['winner'] == df['user_color'],
                df['winner'].isna()
            ]
            choices = [1.0, 0.5]
            df['result'] = np.select(conditions, choices, default=0.0)
            
        else:
            # Last resort: Check if 'rating_diff' exists
            candidates = [c for c in df.columns if 'rating' in c and 'diff' in c]
            if candidates:
                print(f"   -> Using '{candidates[0]}' as proxy for result.")
                df['result'] = np.where(df[candidates[0]] > 0, 1.0, 0.0)
            else:
                raise KeyError("Could not find 'result', 'pl', or 'winner' columns. Cannot determine target.")

    print(f"Data Loaded. Shape: {df.shape}")

    # -------------------------------------------------------
    # 2. REPLICATE PREPROCESSING (Including the 'Smoking Gun')
    # -------------------------------------------------------
    
    # Ensure Session ID exists
    if 'session_id' not in df.columns:
        if 'created_at' in df.columns:
            df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce')
            df = df.sort_values('created_at').reset_index(drop=True)
            df['time_diff'] = df['created_at'].diff()
            # 30 min gap = new session
            df['new_session'] = (df['time_diff'] > pd.Timedelta(minutes=30)).astype(int)
            df['session_id'] = df['new_session'].cumsum()
        else:
            # Fallback for no time data
            df['session_id'] = 0

    df = df.sort_values(['session_id']).reset_index(drop=True)
    df['game_in_session'] = df.groupby('session_id').cumcount()
    
    # --- THE SMOKING GUN (Preserved) ---
    # Recreating the exact environment of the production model
    if 'my_acpl' not in df.columns: df['my_acpl'] = np.nan
    if 'my_blunder_count' not in df.columns: df['my_blunder_count'] = np.nan
    if 'my_avg_secs_per_move' not in df.columns: 
        # Fallback proxy
        if 'game_duration_sec' in df.columns:
             df['my_avg_secs_per_move'] = df['game_duration_sec'] / 40
        else:
             df['my_avg_secs_per_move'] = 10.0 # Default dummy

    # >> The problematic fills <<
    df['my_acpl'] = df['my_acpl'].fillna(50) 
    df['my_blunder_count'] = df['my_blunder_count'].fillna(0)
    df['my_avg_secs_per_move'] = df['my_avg_secs_per_move'].fillna(df['my_avg_secs_per_move'].mean())
    
    # Feature Engineering
    window_sizes = [3, 5]
    grp = df.groupby('session_id')
    for w in window_sizes:
        df[f'roll_{w}_acpl_mean'] = grp['my_acpl'].transform(lambda x: x.rolling(w).mean())
        df[f'roll_{w}_time_per_move'] = grp['my_avg_secs_per_move'].transform(lambda x: x.rolling(w).mean())
        
    first_speed = grp['my_avg_secs_per_move'].transform('first') + 1e-5
    df['speed_vs_start'] = df['my_avg_secs_per_move'] / first_speed
    df['games_played'] = df['game_in_session'] + 1
    
    # Clean NaNs
    numeric_cols = [c for c in df.columns if 'roll_' in c] + ['speed_vs_start']
    df[numeric_cols] = df[numeric_cols].fillna(0)

    # Features List
    features = [
        'my_acpl', 
        'my_blunder_count', 
        'my_avg_secs_per_move', 
        'speed_vs_start', 
        'games_played'
    ] + numeric_cols
    
    # Target: Next Game is a Loss (Tilt Risk)
    # Shift result backwards by 1. If Next Result is 0 (Loss), Target = 1 (Risk)
    df['target'] = (df['result'] == 0).shift(-1).fillna(0).astype(int)
    
    # Drop the last game of every session (since we don't know the next result)
    df = df[df['session_id'] == df['session_id'].shift(-1)]

    # -------------------------------------------------------
    # 3. TRAIN & METRICS
    # -------------------------------------------------------
    X = df[features]
    y = df['target']
    
    print(f"\nTraining on {len(X)} samples...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)
    
    model = HistGradientBoostingClassifier(
        max_iter=100, 
        max_depth=5, 
        random_state=RANDOM_STATE
    )
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # --- STANDARD METRICS ---
    print("\n" + "="*40)
    print("      MODEL PERFORMANCE METRICS")
    print("="*40)
    
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    
    print(f"Accuracy: {acc:.4f}")
    print(f"ROC AUC : {auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Safe', 'Tilt Risk']))
    
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # -------------------------------------------------------
    # 4. FEATURE IMPORTANCE
    # -------------------------------------------------------
    print("\n" + "="*40)
    print("      FEATURE IMPORTANCE ANALYSIS")
    print("="*40)
    
    result = permutation_importance(
        model, X_test, y_test, n_repeats=10, random_state=RANDOM_STATE, n_jobs=-1
    )
    
    sorted_idx = result.importances_mean.argsort()
    
    print(f"{'Feature':<30} | {'Importance':<10} | {'Std Dev'}")
    print("-" * 55)
    for i in sorted_idx:
        name = features[i]
        score = result.importances_mean[i]
        std = result.importances_std[i]
        print(f"{name:<30} | {score:.5f}    | {std:.5f}")
        
    top_feature = features[sorted_idx[-1]]
    print("\n[CONCLUSION]")
    if 'acpl' in top_feature and result.importances_mean[sorted_idx[-1]] > 0.01:
        print("⚠️  ACPL appears important. Ensure 'fillna(50)' isn't masking real signals only for the 3% subset.")
    else:
        print(f"✅ ACPL impact is low or negligible.")
        print(f"   The model is primarily driven by '{top_feature}'.") 
        print("   This confirms the Behavioral Hypothesis (Speed/Fatigue) is doing the heavy lifting.")

if __name__ == "__main__":
    check_feature_importance()