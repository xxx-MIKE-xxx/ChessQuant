import re
import joblib
import pandas as pd
import numpy as np
from pathlib import Path

class TiltInference:
    def __init__(self, model_path="tilt_model.joblib", user_id="julio_amigo_dos"):
        self.user_id = user_id.lower()
        self.model = None
        self.feature_cols = []
        self.threshold = 0.55
        self.window_sizes = [3, 5]
        self._load_model(model_path)

    def _load_model(self, path):
        # Resolve path relative to current script if simple filename provided
        if not Path(path).exists():
            candidate = Path(__file__).parent / path
            if candidate.exists(): path = candidate
            
        if not Path(path).exists():
            print(f"Warning: Model not found at {path}")
            return
        
        artifact = joblib.load(path)
        self.model = artifact["model"]
        self.feature_cols = artifact["features"]
        print(f"TiltInference loaded model from {path}")

    @staticmethod
    def _parse_clock(moves_str):
        """Extracts move times from PGN comments like { [%clk 0:00:19] }"""
        if not isinstance(moves_str, str): return 30.0
        
        times = re.findall(r"\[%clk (\d+):(\d+):(\d+)\]", moves_str)
        if len(times) < 2: return 30.0
        
        seconds = [int(h)*3600 + int(m)*60 + int(s) for h, m, s in times]
        diffs = np.abs(np.diff(seconds))
        diffs = diffs[diffs < 300] # Filter start/end pauses
        return float(np.mean(diffs)) if len(diffs) > 0 else 30.0

    def _process_json_game(self, game):
        """Extracts metrics from Lichess JSON."""
        # Handle Player Color & Identity
        white = game.get('players', {}).get('white', {})
        black = game.get('players', {}).get('black', {})
        
        # Robust ID matching
        white_id = white.get('user', {}).get('id', '').lower()
        white_name = white.get('user', {}).get('name', '').lower()
        
        is_white = (white_id == self.user_id) or (white_name == self.user_id)
        
        p_data = white if is_white else black
        analysis = p_data.get('analysis', {})
        
        return {
            'created_at': pd.to_datetime(game.get('createdAt', 0), unit='ms', utc=True),
            'my_acpl': analysis.get('acpl', 50),
            'my_blunder_count': analysis.get('blunder', 0),
            'my_avg_secs_per_move': self._parse_clock(game.get('moves', ''))
        }

    def _enrich_context(self, games_df):
        """Calculates Rolling Features on the context window."""
        df = games_df.copy()
        df = df.sort_values('created_at').reset_index(drop=True)
        
        # Rolling Features
        for w in self.window_sizes:
            df[f'roll_{w}_acpl_mean'] = df['my_acpl'].rolling(w).mean()
            df[f'roll_{w}_time_per_move'] = df['my_avg_secs_per_move'].rolling(w).mean()
            
        # Context Features
        first_speed = df['my_avg_secs_per_move'].iloc[0] if not df.empty else 30.0
        df['speed_vs_start'] = df['my_avg_secs_per_move'] / (first_speed + 1e-5)
        df['games_played'] = np.arange(len(df)) + 1
        
        # Clean NaNs
        num_cols = [c for c in df.columns if 'roll_' in c] + ['speed_vs_start']
        df[num_cols] = df[num_cols].fillna(0)
        
        return df

    def predict(self, games_json_list):
        """
        Input: List of raw game dicts (Session History).
        Output: Prediction for the LAST game in the list.
        """
        if self.model is None: return {"error": "Model not loaded"}
        if not games_json_list: return {"error": "Empty game list"}
            
        # 1. JSON -> DF
        raw_data = [self._process_json_game(g) for g in games_json_list]
        df_raw = pd.DataFrame(raw_data)
        
        # 2. Enrich
        df_enriched = self._enrich_context(df_raw)
        
        # 3. Predict on Last Game
        last_row = df_enriched.iloc[[-1]][self.feature_cols]
        prob = self.model.predict_proba(last_row)[0, 1]
        
        return {
            "should_stop": bool(prob > self.threshold),
            "probability": float(prob),
            "metrics": {
                "acpl": float(last_row['my_acpl'].iloc[0]),
                "speed_vs_start": float(last_row['speed_vs_start'].iloc[0]),
                "current_speed": float(last_row['my_avg_secs_per_move'].iloc[0])
            }
        }