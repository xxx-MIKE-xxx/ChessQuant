import sys
import json
from pathlib import Path
from tilt_model_sdk import TiltModel

# --- AUTO-LOCATE DATA ---
def find_data():
    search_names = ["games_enriched_full.csv", "julio_amigo_dos_games_full.json"]
    # Search up to 3 parent directories
    base = Path.cwd()
    search_paths = [base] + [base.parents[i] for i in range(3)]
    
    for p in search_paths:
        # Check raw/formatted/root folders
        subdirs = [p, p / "data" / "formatted_data", p / "data" / "raw_data", p / "output_dataset"]
        for d in subdirs:
            if not d.exists(): continue
            for name in search_names:
                if (d / name).exists():
                    return (d / name), ('csv' if 'csv' in name else 'json')
    return None, None

def run_verification():
    print("--- 1. Locating Data ---")
    data_path, file_type = find_data()
    
    if not data_path:
        print("[ERROR] Could not find 'games_enriched_full.csv' or JSON file.")
        print("Please verify the file exists in 'data/formatted_data/'.")
        return

    print(f"Found {file_type.upper()} at: {data_path}")
    
    # Initialize
    sdk = TiltModel(user_id="julio_amigo_dos")
    
    # TRAIN
    print("\n--- 2. Testing Training ---")
    if file_type == 'json':
        with open(data_path, 'r') as f:
            data = json.load(f)
        sdk.train(data, source_type='json')
    else:
        sdk.train(str(data_path), source_type='csv')
        
    # INFERENCE SIMULATION
    print("\n--- 3. Testing Live Inference (Hot Path) ---")
    # We construct a fake "Last 20 Games" payload to simulate Vercel input
    fake_games = []
    # Create dummy JSON-like structure
    base_time = 1700000000000
    for i in range(20):
        fake_games.append({
            "createdAt": base_time + (i * 600000), # 10 mins apart
            "players": {
                "white": {"user": {"name": "julio_amigo_dos"}, "analysis": {"acpl": 10 + i*5}}, # Getting worse
                "black": {"user": {"name": "opponent"}}
            },
            "moves": "[%clk 0:00:30] " * 40, # Constant speed
            "winner": "black" # Losing
        })
    
    result = sdk.predict(fake_games)
    print("Result Payload:")
    print(json.dumps(result, indent=2))
    
    if result['stop_probability'] > 0:
        print("\n[SUCCESS] Pipeline Ready for Deployment.")
        print("Next Step: Copy 'tilt_model_sdk.py' to your Vercel API folder.")

if __name__ == "__main__":
    run_verification()