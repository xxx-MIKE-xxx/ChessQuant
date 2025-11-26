import re
import joblib
import json
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingClassifier

# --- CONFIG ---
# If running on Vercel, use /tmp for temp files if needed, 
# but models should be loaded from memory or cloud storage.
DEFAULT_MODEL_NAME = "tilt_model.joblib"
WINDOW_SIZES = [3, 5]
MIN_TRAINING_SESSIONS = 3

class TiltModel:
    def __init__(self, user_id="default"):
        self.user_id = user_id
        self.model = None
        self.feature_cols = []
        self.threshold = 0.55

    # ------------------------------------------------------------------
    # 1. CORE LOGIC: Feature Extraction
    # ------------------------------------------------------------------
    @staticmethod
    def _parse_clock(moves_str):
        """Extracts move times from PGN clock comments."""
        if not isinstance(moves_str, str): return 30.0
        # Regex for [%clk 0:00:00]
        times = re.findall(r"\[%clk (\d+):(\d+):(\d+)\]", moves_str)
        if len(times) < 2: return 30.0
        
        seconds = [int(h)*3600 + int(m)*60 + int(s) for h, m, s in times]
        diffs = np.abs(np.diff(seconds))
        # Filter artifacts > 300s (game start/end pauses)
        diffs = diffs[diffs < 300]
        return float(np.mean(diffs)) if len(diffs) > 0 else 30.0

    def _enrich_data(self, df):
        """
        Shared logic for Training (CSV) and Inference (JSON).
        Calculates Rolling Features from raw columns.
        """
        df = df.copy()
        df = df.sort_values('created_at').reset_index(drop=True)

        # Session ID Logic (1 hour gap)
        df['time_diff'] = df['created_at'].diff()
        df['new_session'] = (df['time_diff'] > pd.Timedelta(hours=1)).astype(int)
        df['session_id'] = df['new_session'].cumsum()

        # Rolling Stats
        grp = df.groupby('session_id')
        for w in WINDOW_SIZES:
            df[f'roll_{w}_acpl_mean'] = grp['my_acpl'].transform(lambda x: x.rolling(w).mean())
            df[f'roll_{w}_time_per_move'] = grp['my_avg_secs_per_move'].transform(lambda x: x.rolling(w).mean())

        # Speed vs Session Start
        first_speed = grp['my_avg_secs_per_move'].transform('first')
        df['speed_vs_start'] = df['my_avg_secs_per_move'] / (first_speed + 1e-5)
        df['games_played'] = grp.cumcount() + 1
        
        # Cleanup
        num_cols = [c for c in df.columns if 'roll_' in c] + ['speed_vs_start']
        df[num_cols] = df[num_cols].fillna(0)
        
        return df

    # ------------------------------------------------------------------
    # 2. INPUT ADAPTERS (JSON vs CSV)
    # ------------------------------------------------------------------
    def _df_from_json(self, games_list):
        """Converts raw Lichess JSON list to DataFrame."""
        rows = []
        for g in games_list:
            # Safe Player Extraction
            white = g.get('players', {}).get('white', {})
            black = g.get('players', {}).get('black', {})
            
            # Guess user color
            u_name = self.user_id.lower()
            is_white = (white.get('user', {}).get('name', '').lower() == u_name) or \
                       (white.get('user', {}).get('id', '').lower() == u_name)
            
            p_data = white if is_white else black
            analysis = p_data.get('analysis', {})
            
            # Rating Change (PL)
            if 'ratingDiff' in p_data: pl = p_data['ratingDiff']
            else: 
                # Fallback
                winner = g.get('winner')
                my_color = 'white' if is_white else 'black'
                pl = 6 if winner == my_color else (-6 if winner else 0)

            rows.append({
                'created_at': pd.to_datetime(g.get('createdAt', 0), unit='ms', utc=True),
                'my_acpl': analysis.get('acpl', 50),
                'my_blunder_count': analysis.get('blunder', 0),
                'my_avg_secs_per_move': self._parse_clock(g.get('moves', '')),
                'pl': pl
            })
        return pd.DataFrame(rows)

    def _df_from_csv(self, csv_path):
        """Loads your specific games_enriched_full.csv format."""
        df = pd.read_csv(csv_path)
        df['created_at'] = pd.to_datetime(df['created_at'], format='mixed', utc=True)
        # Map your CSV columns to standard names if needed
        # Assuming your CSV already has 'my_acpl', 'my_avg_secs_per_move', 'pl'
        # Fill NAs
        df['my_acpl'] = df['my_acpl'].fillna(50)
        df['my_avg_secs_per_move'] = df['my_avg_secs_per_move'].fillna(30)
        df['my_blunder_count'] = df['my_blunder_count'].fillna(0)
        return df

    # ------------------------------------------------------------------
    # 3. PUBLIC API
    # ------------------------------------------------------------------
    def train(self, input_data, source_type='csv'):
        """
        input_data: filepath (str) if CSV, or list of dicts if JSON
        """
        print(f"Loading data from {source_type}...")
        if source_type == 'csv':
            df = self._df_from_csv(input_data)
        elif source_type == 'json':
            df = self._df_from_json(input_data)
        else:
            raise ValueError("source_type must be 'csv' or 'json'")

        # Enrich
        df = self._enrich_data(df)

        # Targets (Best Stop Logic)
        df['sess_pl'] = df.groupby('session_id')['pl'].cumsum()
        s_max = df.groupby('session_id')['sess_pl'].transform('max')
        df['is_max'] = (df['sess_pl'] == s_max)
        
        # Label 1 for first max
        df['target'] = 0
        idx = df[df['is_max']].groupby('session_id').head(1).index
        df.loc[idx, 'target'] = 1
        
        # Filter Short Sessions
        valid_sess = df['session_id'].value_counts()
        valid_sess = valid_sess[valid_sess >= MIN_TRAINING_SESSIONS].index
        df_train = df[df['session_id'].isin(valid_sess)].copy()
        
        # Feature Selection (Blind to PL)
        self.feature_cols = [
            'my_acpl', 'my_blunder_count', 'my_avg_secs_per_move', 
            'speed_vs_start', 'games_played'
        ] + [c for c in df.columns if 'roll_' in c]

        print(f"Training on {len(df_train)} games...")
        self.model = HistGradientBoostingClassifier(
            learning_rate=0.03, max_iter=500, max_depth=5, 
            early_stopping=True, class_weight='balanced', random_state=42
        )
        self.model.fit(df_train[self.feature_cols], df_train['target'])
        print("Training Complete.")

    def predict(self, recent_games_json):
        """
        Predicts on the LAST game in the list.
        Expects a list of at least 1 game (dict).
        """
        if self.model is None: raise ValueError("Model not loaded")
        
        df = self._df_from_json(recent_games_json)
        df = self._enrich_data(df)
        
        if df.empty: return {"error": "No data"}
        
        last_row = df.iloc[[-1]][self.feature_cols]
        prob = self.model.predict_proba(last_row)[0, 1]
        
        # Explainability metrics
        metrics = {
            "speed_vs_start": float(last_row['speed_vs_start'].values[0]),
            "acpl": float(last_row['my_acpl'].values[0]),
            "recent_speed": float(last_row.get('roll_3_time_per_move', 0).values[0])
        }

        return {
            "stop_probability": float(prob),
            "should_stop": bool(prob > self.threshold),
            "metrics": metrics
        }

    def save_model(self, path):
        joblib.dump({'model': self.model, 'feats': self.feature_cols}, path)

    def load_model(self, path):
        art = joblib.load(path)
        self.model = art['model']
        self.feature_cols = art['feats']