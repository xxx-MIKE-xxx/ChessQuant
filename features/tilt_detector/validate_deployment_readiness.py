# File: features/tilt_detector/validate_deployment_readiness.py
import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path

# Add project root to path
BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(BASE_DIR))

from features.tilt_detector.train_model_sdk import TiltModel

# --- CONFIG ---
DATA_PATH = BASE_DIR / "data/train_ready_data/games_dataset_model.csv"
MODEL_PATH = BASE_DIR / "assets/tilt_model.json"
CONFIG_PATH = BASE_DIR / "assets/tilt_config.joblib"

def run_verification():
    print("🛡️  STARTING DEPLOYMENT VERIFICATION")
    print("=====================================")

    # 1. CHECK DATA AVAILABILITY
    print("\n[1] Verifying Data...")
    if not DATA_PATH.exists():
        print(f"   ❌ ERROR: Training data not found at {DATA_PATH}")
        print("   Run 'scripts/prepare_training_data.py' first.")
        return
    print(f"   ✅ Found training data: {DATA_PATH.name}")

    # 2. TEST TRAINING (FAST MODE)
    print("\n[2] Testing Training Pipeline...")
    try:
        # Initialize SDK
        sdk = TiltModel()
        # Run training (this saves the model too)
        sdk.train_from_file(str(DATA_PATH), str(MODEL_PATH))
        print("   ✅ Training completed successfully.")
    except Exception as e:
        print(f"   ❌ Training Failed: {e}")
        return

    # 3. TEST PERSISTENCE (Load from disk)
    print("\n[3] Testing Model Loading...")
    try:
        sdk_live = TiltModel()
        sdk_live.load(str(MODEL_PATH))
        print(f"   ✅ Model loaded. Threshold: {sdk_live.config.get('threshold', 'N/A')}")
    except Exception as e:
        print(f"   ❌ Loading Failed: {e}")
        return

    # 4. TEST INFERENCE (With Correct Flat Format)
    print("\n[4] Testing Inference (Mock Session)...")
    
    # Create a fake "Tilt Session"
    # Note: We send FLAT fields as expected by _enrich_json
    base_time = 1760000000000
    fake_session = []
    
    for i in range(5):
        fake_session.append({
            "id": f"test_game_{i}",
            "createdAt": base_time + (i * 300000), # 5 min gaps
            "lastMoveAt": base_time + (i * 300000) + 60000,
            "my_acpl": 100 + (i * 20),      # Degrading performance
            "my_blunder_count": i,          # Increasing blunders
            "my_avg_secs_per_move": 2.0,    # Playing fast
            "result": 0.0,                  # Losing
            "rating_diff": -10
        })

    try:
        result = sdk_live.predict(fake_session)
        print("\n   📊 Inference Result:")
        print(json.dumps(result, indent=3))
        
        if result['should_stop']:
            print("   ✅ Logic Check: Model correctly identified tilt behavior.")
        else:
            print("   ⚠️ Logic Check: Model was lenient (predicted safe). Check threshold.")
            
    except Exception as e:
        print(f"   ❌ Inference Failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_verification()