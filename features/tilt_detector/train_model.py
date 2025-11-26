# File: features/tilt_detector/train_model.py
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import os
from pathlib import Path
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score

# --- CONFIGURATION ---
BASE_DIR = Path(__file__).resolve().parents[2] # Root of Repo
DATA_FILE = BASE_DIR / "data/train_ready_data/games_dataset_model.csv"
ASSETS_DIR = BASE_DIR / "assets"
OUTPUT_ANALYSIS_DIR = BASE_DIR / "output_analysis/tilt_detector"

# Output Paths
MODEL_FILE = ASSETS_DIR / "tilt_model.json"
CONFIG_FILE = ASSETS_DIR / "tilt_config.joblib"
SUMMARY_FILE = OUTPUT_ANALYSIS_DIR / "training_summary.txt"

# STABILIZED HYPERPARAMETERS (Locked for Production)
# Shallow trees + moderate weights for small data stability
PARAMS = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'max_depth': 3,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'n_estimators': 150,
    'gamma': 1.0,             # Anti-noise
    'min_child_weight': 5,    # Anti-outlier
    'scale_pos_weight': 5,    # Bias towards "Stop"
    'n_jobs': -1,
    'random_state': 42
}

def run_pipeline():
    print(f"--- Starting Production Pipeline ---")
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    
    if not DATA_FILE.exists():
        print(f" Error: Data file not found at {DATA_FILE}")
        return

    # ---------------------------------------------------------
    # PHASE 1: TRAINING
    # ---------------------------------------------------------
    print("\n--- [Phase 1] Model Training ---")
    df = pd.read_csv(DATA_FILE)
    print(f"Loaded {len(df)} games.")
    
    # Define Features (Exclude Metadata & Target)
    exclude_cols = ['id', 'session_id', 'created_at', 'target', 'rating_diff', 'session_cum_pl']
    features = [c for c in df.columns if c not in exclude_cols]
    
    print(f"Features ({len(features)}): {features}")
    
    X = df[features]
    y = df['target']
    groups = df['session_id']
    
    # 5-Fold CV
    cv = StratifiedGroupKFold(n_splits=5)
    auc_scores = []
    
    print("Running 5-Fold CV...")
    for i, (train_idx, val_idx) in enumerate(cv.split(X, y, groups=groups)):
        model = xgb.XGBClassifier(**PARAMS)
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        
        val_probs = model.predict_proba(X.iloc[val_idx])[:, 1]
        try:
            score = roc_auc_score(y.iloc[val_idx], val_probs)
        except:
            score = 0.5 # Edge case safety
        auc_scores.append(score)
        # print(f"  Fold {i+1}: AUC {score:.4f}") # Optional verbosity
        
    avg_auc = np.mean(auc_scores)
    print(f"Average CV AUC: {avg_auc:.4f}")
    
    # Final Training on Full Data
    print("Training Final Model...")
    final_model = xgb.XGBClassifier(**PARAMS)
    final_model.fit(X, y)
    
    # Save Initial Model
    final_model.save_model(MODEL_FILE)
    print(f" Model saved to {MODEL_FILE}")

    # ---------------------------------------------------------
    # PHASE 2: THRESHOLD OPTIMIZATION
    # ---------------------------------------------------------
    print("\n--- [Phase 2] Threshold Optimization ---")
    
    # Predict probabilities for the whole dataset
    # (In a perfect world we'd use OOF predictions, but for threshold calibration 
    # on small data, using full train set is acceptable to find the 'operating point')
    df['tilt_prob'] = final_model.predict_proba(X)[:, 1]
    
    # Ensure P/L column exists
    if 'rating_diff' not in df.columns:
        print("Warning: 'rating_diff' missing. Defaulting to 0 P/L.")
        df['rating_diff'] = 0
    
    # Simulation Loop
    thresholds = np.arange(0.30, 0.90, 0.02)
    results = []
    baseline_pl = df['rating_diff'].sum()
    
    for thresh in thresholds:
        sim_pl = 0
        games_played = 0
        
        # Group by session
        for sess_id, group in df.groupby('session_id'):
            # Logic: Stop immediately AFTER the first game where prob > thresh
            
            # Reset index to handle the group as a sequential list
            g_reset = group.reset_index()
            
            # Find indices where model says STOP
            stop_flags = g_reset.index[g_reset['tilt_prob'] > thresh]
            
            if not stop_flags.empty:
                first_stop_idx = stop_flags[0]
                # We play the game that triggered the stop, then quit.
                # Slice includes the stop game (iloc is exclusive of end, so +1)
                games_to_play = g_reset.iloc[:first_stop_idx+1]
                
                sim_pl += games_to_play['rating_diff'].sum()
                games_played += len(games_to_play)
            else:
                # Never stopped
                sim_pl += group['rating_diff'].sum()
                games_played += len(group)
                
        results.append({'thresh': thresh, 'pl': sim_pl, 'games': games_played})
        
    # Find Best
    res_df = pd.DataFrame(results)
    best_row = res_df.loc[res_df['pl'].idxmax()]
    
    best_thresh = float(best_row['thresh'])
    best_pl = float(best_row['pl'])
    improvement = best_pl - baseline_pl
    
    print(f" Optimal Threshold: {best_thresh:.2f}")
    print(f"   Baseline P/L: {baseline_pl:.0f}")
    print(f"   Simulated P/L: {best_pl:.0f}")
    print(f"   Net Gain: {improvement:+.0f}")

    # ---------------------------------------------------------
    # PHASE 3: EXPORT & REPORTING
    # ---------------------------------------------------------
    print("\n--- [Phase 3] Finalizing ---")
    
    # Save Config
    config = {
        'features': features,
        'params': PARAMS,
        'cv_auc': avg_auc,
        'threshold': best_thresh,
        'pl_improvement_est': improvement
    }
    joblib.dump(config, CONFIG_FILE)
    
    # Save Summary Text
    summary = f"""
    TILT MODEL PRODUCTION SUMMARY
    =============================
    Date: {pd.Timestamp.now()}
    
    [TRAINING]
    Data Rows: {len(df)}
    CV AUC:    {avg_auc:.4f} (Std: {np.std(auc_scores):.4f})
    
    [OPTIMIZATION]
    Optimal Threshold: {best_thresh:.2f}
    Baseline P/L:      {baseline_pl:.0f}
    Optimized P/L:     {best_pl:.0f}
    Net Improvement:   {improvement:+.0f} rating points
    Games Avoided:     {len(df) - int(best_row['games'])}
    
    [CONFIGURATION]
    Features: {features}
    Params:   {PARAMS}
    """
    
    with open(SUMMARY_FILE, 'w') as f:
        f.write(summary)
        
    print(f" Config saved: {CONFIG_FILE}")
    print(f"Summary saved: {SUMMARY_FILE}")
    print("Pipeline Complete.")

if __name__ == "__main__":
    run_pipeline()