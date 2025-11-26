# File: features/tilt_detector/optimize_threshold.py
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import os
from pathlib import Path

# --- CONFIG ---
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_FILE = BASE_DIR / "data/train_ready_data/games_dataset_model.csv"
MODEL_FILE = BASE_DIR / "assets/tilt_model.json"
CONFIG_FILE = BASE_DIR / "assets/tilt_config.joblib"
SUMMARY_FILE = BASE_DIR / "output_analysis/tilt_detector/threshold_optimisation_summary.txt"

def optimize():
    print("--- Starting Threshold Optimization ---")
    
    if not MODEL_FILE.exists() or not CONFIG_FILE.exists():
        print("❌ Error: Model assets not found. Run training first.")
        return

    # 1. Load Assets
    df = pd.read_csv(DATA_FILE)
    config = joblib.load(CONFIG_FILE)
    features = config['features']
    
    model = xgb.XGBClassifier()
    model.load_model(MODEL_FILE)
    
    print(f"Loaded {len(df)} games.")
    
    # 2. Inference
    X = df[features]
    df['tilt_prob'] = model.predict_proba(X)[:, 1]
    
    # Ensure we have session P/L data
    # If 'rating_diff' exists, use it. If not, rely on what's in CSV.
    if 'rating_diff' not in df.columns:
        print("Warning: 'rating_diff' missing from CSV. P/L sim might be wrong if not present.")
        df['rating_diff'] = 0 # Fail safe
        
    # 3. Simulation Loop
    thresholds = np.arange(0.30, 0.90, 0.02) # Scan reasonable range
    results = []
    
    baseline_pl = df['rating_diff'].sum()
    
    for thresh in thresholds:
        sim_pl = 0
        games_played = 0
        
        # Iterate by session
        for sess_id, group in df.groupby('session_id'):
            # Check for Stop Signals
            stop_idx = group.index[group['tilt_prob'] > thresh]
            
            if not stop_idx.empty:
                first_stop = stop_idx[0]
                # Include the game that triggered stop
                # We use .loc range. Since index might not be sequential per session in grouped,
                # we slice by position logic.
                
                # Reset index for safe slicing within group
                g_reset = group.reset_index()
                
                # Find the *local* index of the first stop
                local_stop_indices = g_reset.index[g_reset['tilt_prob'] > thresh]
                if not local_stop_indices.empty:
                    local_stop_idx = local_stop_indices[0]
                    # Take games up to and including this one
                    played_games = g_reset.iloc[:local_stop_idx+1]
                    sim_pl += played_games['rating_diff'].sum()
                    games_played += len(played_games)
                else:
                    # Should be caught by outer if, but safety fallback
                    sim_pl += group['rating_diff'].sum()
                    games_played += len(group)
            else:
                # Play all
                sim_pl += group['rating_diff'].sum()
                games_played += len(group)
                
        results.append({
            'thresh': thresh,
            'pl': sim_pl,
            'games': games_played
        })
        
    # 4. Find Optimal
    res_df = pd.DataFrame(results)
    best_row = res_df.loc[res_df['pl'].idxmax()]
    
    best_thresh = float(best_row['thresh'])
    best_pl = float(best_row['pl'])
    improvement = best_pl - baseline_pl
    
    print(f"🏆 Optimal Threshold: {best_thresh:.2f}")
    print(f"   Baseline P/L: {baseline_pl}")
    print(f"   Simulated P/L: {best_pl}")
    print(f"   Improvement: {improvement:+}")
    
    # 5. Update Config
    config['threshold'] = best_thresh
    config['pl_improvement_est'] = improvement
    joblib.dump(config, CONFIG_FILE)
    print(f"✅ Updated Config: {CONFIG_FILE}")
    
    # 6. Save Summary
    summary = f"""
    THRESHOLD OPTIMIZATION SUMMARY
    ==============================
    Date: {pd.Timestamp.now()}
    Optimal Threshold: {best_thresh:.2f}
    
    Baseline P/L:    {baseline_pl:.0f}
    Optimized P/L:   {best_pl:.0f}
    Net Improvement: {improvement:+.0f} rating points
    
    Games Played: {int(best_row['games'])} / {len(df)} ({int(best_row['games']/len(df)*100)}%)
    """
    
    with open(SUMMARY_FILE, 'w') as f:
        f.write(summary)
    print(f"   Summary saved to {SUMMARY_FILE}")

if __name__ == "__main__":
    optimize()