# File: scripts/demo_tilt_sdk.py
import sys
from pathlib import Path
import pandas as pd
import numpy as np  # <--- Added this missing import

# Add project root to path so we can import from 'features'
BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(BASE_DIR))

from features.tilt_detector.train_model_sdk import TiltModel

def run_demo():
    print("🤖 INITIALIZING TILT MODEL SDK")
    print("==============================")
    
    # Initialize
    tilt_ai = TiltModel()
    
    # -------------------------------------------------
    # SCENARIO 1: Training / Retraining (Admin Task)
    # -------------------------------------------------
    # You would run this weekly or when new data is available.
    print("\n[1] SIMULATING RETRAINING...")
    try:
        # Uses default paths defined in the class, or you can override
        tilt_ai.train_from_file()
        print("   Training successful.")
    except Exception as e:
        print(f"   Training skipped (Data missing?): {e}")

    # -------------------------------------------------
    # SCENARIO 2: Production Inference (User Request)
    # -------------------------------------------------
    print("\n[2] SIMULATING LIVE INFERENCE...")
    
    # 1. Load the production model
    model_path = BASE_DIR / "assets/tilt_model.json"
    config_path = BASE_DIR / "assets/tilt_config.joblib"
    
    if not model_path.exists():
        print("❌ Model not found. Please run training first.")
        return

    tilt_ai.load(model_path)
    
    # 2. Mock "Live" Data from Lichess API
    # Imagine a user just finished a game. This is their session history.
    session_history = [
        {
            "id": "game_1",
            "createdAt": 1763900000000,
            "lastMoveAt": 1763900300000, # 5 min game
            "result": 1.0, # Win
            "my_acpl": 35,
            "my_blunder_count": 0,
            "my_avg_secs_per_move": 4.5,
            "rating_diff": 12
        },
        {
            "id": "game_2",
            "createdAt": 1763900310000, # Started 10s later (Fast requeue)
            "lastMoveAt": 1763900400000, # Short game
            "result": 0.0, # Loss
            "my_acpl": 120, # Bad play
            "my_blunder_count": 3,
            "my_avg_secs_per_move": 1.2, # Playing too fast
            "rating_diff": -11
        },
        {
            "id": "game_3",
            "createdAt": 1763900405000, # Started 5s later (Rage queue)
            "lastMoveAt": 1763900450000, # Very short game
            "result": 0.0, # Loss
            "my_acpl": 150, # Terrible play
            "my_blunder_count": 4,
            "my_avg_secs_per_move": 0.8, # Speed running moves
            "rating_diff": -12
        }
    ]
    
    print(f"   Analyzing session with {len(session_history)} games...")
    prediction = tilt_ai.predict(session_history)
    
    if prediction:
        print("\n📊 TILT REPORT")
        print(f"   Stop Probability: {prediction['stop_probability']:.2%}")
        print(f"   Threshold:        {prediction['threshold']:.2%}")
        print(f"   Decision:         {'🛑 STOP PLAYING' if prediction['should_stop'] else '✅ KEEP PLAYING'}")
        
        print("\n   Feature Explainability (Why?):")
        feats = prediction['features']
        print(f"   - Speed vs Start: {feats['speed_vs_start']:.2f}x (Faster than session start)")
        print(f"   - Loss Streak:    {feats['loss_streak']}")
        print(f"   - Session P/L:    {feats['session_pl']}")
        # np.expm1 inverses the log1p transform we did in training
        print(f"   - Break Time:     {np.expm1(feats['log_break_time']):.1f} seconds") 
    else:
        print("   Prediction failed (Empty result).")

if __name__ == "__main__":
    run_demo()