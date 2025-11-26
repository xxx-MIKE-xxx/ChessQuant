# File: features/tilt_detector/train_model.py
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import os
from pathlib import Path
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix

# --- CONFIG ---
BASE_DIR = Path(__file__).resolve().parents[2] # Root of Repo
DATA_FILE = BASE_DIR / "data/train_ready_data/games_dataset_model.csv"
ASSETS_DIR = BASE_DIR / "assets"
OUTPUT_ANALYSIS_DIR = BASE_DIR / "output_analysis/tilt_detector"

MODEL_FILE = ASSETS_DIR / "tilt_model.json"
CONFIG_FILE = ASSETS_DIR / "tilt_config.joblib"
SUMMARY_FILE = OUTPUT_ANALYSIS_DIR / "training_summary.txt"

# STABILIZED HYPERPARAMETERS (Locked)
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

def train_model():
    print(f"--- Starting Training Pipeline ---")
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    
    if not DATA_FILE.exists():
        print(f"❌ Error: Data file not found at {DATA_FILE}")
        return

    # 1. Load Data
    df = pd.read_csv(DATA_FILE)
    print(f"Loaded {len(df)} games.")
    
    # 2. Define Features (Exclude Metadata & Target)
    exclude_cols = ['id', 'session_id', 'created_at', 'target', 'rating_diff', 'session_cum_pl']
    features = [c for c in df.columns if c not in exclude_cols]
    
    print(f"Training with {len(features)} features: {features}")
    
    X = df[features]
    y = df['target']
    groups = df['session_id']
    
    # 3. Cross-Validation Evaluation (Sanity Check)
    cv = StratifiedGroupKFold(n_splits=5)
    auc_scores = []
    
    print("\nRunning 5-Fold CV...")
    for i, (train_idx, val_idx) in enumerate(cv.split(X, y, groups=groups)):
        model = xgb.XGBClassifier(**PARAMS)
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        
        val_probs = model.predict_proba(X.iloc[val_idx])[:, 1]
        try:
            score = roc_auc_score(y.iloc[val_idx], val_probs)
        except:
            score = 0.5 # Handle single-class fold edge case
        auc_scores.append(score)
        print(f"  Fold {i+1}: AUC {score:.4f}")
        
    avg_auc = np.mean(auc_scores)
    print(f"Average CV AUC: {avg_auc:.4f}")
    
    # 4. Final Training (Full Dataset)
    print("\nTraining Final Model on Full Data...")
    final_model = xgb.XGBClassifier(**PARAMS)
    final_model.fit(X, y)
    
    # 5. Save Artifacts
    final_model.save_model(MODEL_FILE)
    
    config = {
        'features': features,
        'params': PARAMS,
        'cv_auc': avg_auc,
        'threshold': 0.5 # Default, will be optimized in next step
    }
    joblib.dump(config, CONFIG_FILE)
    
    # 6. Save Summary
    summary = f"""
    TILT MODEL TRAINING SUMMARY
    ===========================
    Date: {pd.Timestamp.now()}
    Data Rows: {len(df)}
    Features: {len(features)}
    CV AUC: {avg_auc:.4f} (Std: {np.std(auc_scores):.4f})
    
    Parameters:
    {PARAMS}
    
    Feature List:
    {features}
    """
    
    with open(SUMMARY_FILE, 'w') as f:
        f.write(summary)
        
    print(f"\n✅ Success!")
    print(f"   Model: {MODEL_FILE}")
    print(f"   Config: {CONFIG_FILE}")
    print(f"   Summary: {SUMMARY_FILE}")

if __name__ == "__main__":
    train_model()