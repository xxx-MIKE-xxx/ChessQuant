import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import HistGradientBoostingClassifier
from pathlib import Path

# --- CONFIG ---
RANDOM_STATE = 42
MIN_SESSION_LENGTH = 3
# Model B used "Pure Tilt" features. We stick to that strict list.
WINDOW_SIZES = [3, 5]

OUTPUT_MODEL_PATH = Path("assets/tilt_model_production.joblib")
OUTPUT_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------
# 1. DATA LOADING
# ---------------------------------------------------------
def find_data_file(filename):
    base = Path(__file__).resolve().parent
    candidates = [
        base / filename,
        base / "data" / "formatted_data" / filename,
        base.parent.parent / "data" / "formatted_data" / filename, # from features/tilt_detector
        Path.cwd() / "data" / "formatted_data" / filename
    ]
    for p in candidates:
        if p.exists(): return p
    return None

def load_and_preprocess(csv_path):
    print(f"Loading training data from: {csv_path}")
    df = pd.read_csv(csv_path)
    df['created_at'] = pd.to_datetime(df['created_at'], format='mixed', utc=True)
    
    # Ensure Sessions exist
    if 'session_id' not in df.columns:
        df = df.sort_values('created_at').reset_index(drop=True)
        df['time_diff'] = df['created_at'].diff()
        df['new_session'] = (df['time_diff'] > pd.Timedelta(hours=1)).astype(int)
        df['session_id'] = df['new_session'].cumsum()
        
    df = df.sort_values(['session_id', 'created_at']).reset_index(drop=True)
    df['game_in_session'] = df.groupby('session_id').cumcount()
    df['session_cum_pl'] = df.groupby('session_id')['pl'].cumsum()
    

    return df

# ---------------------------------------------------------
# 2. FEATURE ENGINEERING (PURE TILT ONLY)
# ---------------------------------------------------------
def add_features(df):
    df_feat = df.copy()
    grp = df_feat.groupby('session_id')
    
    for w in WINDOW_SIZES:
        # Note: We purposely EXCLUDE 'roll_pl_sum' to keep it blind
        df_feat[f'roll_{w}_acpl_mean'] = grp['my_acpl'].transform(lambda x: x.rolling(w).mean())
        df_feat[f'roll_{w}_time_per_move'] = grp['my_avg_secs_per_move'].transform(lambda x: x.rolling(w).mean())

    # Fatigue / Speed
    first_game_speed = grp['my_avg_secs_per_move'].transform('first')
    df_feat['speed_vs_start'] = df_feat['my_avg_secs_per_move'] / (first_game_speed + 1e-5)
    
    df_feat['games_played'] = df_feat['game_in_session'] + 1
    
    # Clean NaNs
    numeric_cols = [c for c in df_feat.columns if 'roll_' in c] + ['speed_vs_start']
    df_feat[numeric_cols] = df_feat[numeric_cols].fillna(0)
    
    # --- THE PRODUCTION FEATURE LIST ---
    # STRICTLY NO 'pl', 'result_score', or 'current_session_pl'
    feature_cols = [
        'my_acpl', 
        'my_blunder_count', 
        'my_avg_secs_per_move', 
        'speed_vs_start',
        'games_played'
    ] + numeric_cols
    
    return df_feat, feature_cols

# ---------------------------------------------------------
# 3. TARGETS
# ---------------------------------------------------------
def build_target(df):
    session_max = df.groupby('session_id')['session_cum_pl'].transform('max')
    df_temp = df.copy()
    df_temp['is_max'] = (df['session_cum_pl'] == session_max)
    
    # First max only
    target_idx = df_temp[df_temp['is_max']].groupby('session_id')['game_in_session'].idxmin()
    
    df['target'] = 0
    df.loc[target_idx.values, 'target'] = 1
    return df

# ---------------------------------------------------------
# 4. TRAIN & SAVE
# ---------------------------------------------------------
def train_production_model():
    csv_file = find_data_file("games_enriched_full.csv")
    if not csv_file:
        print("Error: Data not found.")
        return

    df = load_and_preprocess(csv_file)
    
    # Filter training data (ignore short sessions to reduce noise)
    session_counts = df['session_id'].value_counts()
    long_sessions = session_counts[session_counts >= MIN_SESSION_LENGTH].index
    df_train = df[df['session_id'].isin(long_sessions)].copy()
    
    # Build Targets & Features
    df_train = build_target(df_train)
    df_train, feature_cols = add_features(df_train)
    
    print(f"\nTraining on {len(df_train)} games...")
    print(f"Features used ({len(feature_cols)}): {feature_cols}")
    print("Strategy: PURE TILT (Blind to Score)")
    
    model = HistGradientBoostingClassifier(
        max_iter=500, 
        max_depth=5, 
        early_stopping=True, 
        class_weight='balanced', 
        random_state=RANDOM_STATE
    )
    
    model.fit(df_train[feature_cols], df_train['target'])
    
    # Save Artifacts
    artifact = {
        "model": model,
        "feature_cols": feature_cols,
        "version": "v1_pure_tilt_blind",
        "threshold_recommended": 0.55
    }
    
    joblib.dump(artifact, OUTPUT_MODEL_PATH)
    print(f"\nSUCCESS! Model saved to: {OUTPUT_MODEL_PATH}")
    print("To use this model in your app, load it and pass ONLY the behavioral features.")

if __name__ == "__main__":
    train_production_model()