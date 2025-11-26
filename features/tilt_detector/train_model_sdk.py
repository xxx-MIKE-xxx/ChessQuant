# File: features/tilt_detector/train_model.py
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import json
import os
import pytz
from pathlib import Path
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score

# --- DEFAULT PATHS ---
BASE_DIR = Path(__file__).resolve().parents[2]
DEFAULT_DATA = BASE_DIR / "data/train_ready_data/games_dataset_model.csv"
DEFAULT_MODEL = BASE_DIR / "assets/tilt_model.json"
DEFAULT_CONFIG = BASE_DIR / "assets/tilt_config.joblib"
DEFAULT_SUMMARY = BASE_DIR / "output_analysis/tilt_detector/training_summary.txt"

class TiltModel:
    def __init__(self, local_tz='Europe/Warsaw'):
        self.local_tz = local_tz
        self.model = None
        self.config = {}
        
        # Features expected by the model
        self.feature_cols = [
            'my_acpl', 'my_blunder_count', 'my_avg_secs_per_move', 'result',
            'games_played', 'speed_vs_start', 'session_pl', 'loss_streak',
            'roll_5_acpl_mean', 'roll_5_time_per_move',
            'log_break_time',
            'tod_morning', 'tod_midday', 'tod_evening', 'tod_night'
        ]
        
        # Stabilized Hyperparameters
        self.params = {
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'max_depth': 3,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'n_estimators': 150,
            'gamma': 1.0,
            'min_child_weight': 5,
            'scale_pos_weight': 5,
            'n_jobs': -1,
            'random_state': 42
        }

    # ------------------------------------------------------------------
    # 1. INFERENCE LOGIC (Raw JSON -> Model Features)
    # ------------------------------------------------------------------
    def _assign_time_of_day(self, hour):
        if 5 <= hour < 9: return 'morning'
        elif 9 <= hour < 18: return 'midday'
        elif 18 <= hour < 23: return 'evening'
        else: return 'night'

    def _enrich_json(self, games_list):
        """
        Transforms raw JSON (session history) into model features.
        Mirrors the logic of prepare_training_data.py exactly.
        """
        if not games_list:
            return pd.DataFrame()

        # 1. Normalize JSON
        # We assume games_list is a list of dicts from the Lichess API
        # We need to extract fields similar to 'pd.json_normalize' but safely
        rows = []
        for g in games_list:
            # Determine Hero (User) Color - assumes user is identifying the session
            # For inference, we often assume the user providing the history IS the hero.
            # But we need to parse the complex JSON structure if it comes raw from Lichess.
            
            # Simplified extraction for inference speed:
            # We expect the input to be slightly pre-parsed OR raw Lichess structure.
            # Let's handle Raw Lichess structure (players.white.analysis...)
            
            # Heuristic: Identify 'me' by looking for the user who played ALL games? 
            # Or pass user_id. For now, assume we extract basic stats.
            
            # NOTE: In production app, you likely pass cleaned stats. 
            # Here we implement robust extraction assuming standard Lichess fields.
            
            # Extract timestamps
            created_at = pd.to_datetime(g.get('createdAt'), unit='ms', utc=True)
            last_move = pd.to_datetime(g.get('lastMoveAt'), unit='ms', utc=True)
            
            # P/L
            # Try to find 'ratingDiff' in players
            # We can't easily know which player is 'me' without user_id. 
            # We assume the API caller filtered for 'me' or provided 'rating_diff'
            rating_diff = g.get('rating_diff', 0) # Expect pre-calculated or flat
            
            # Stats
            # Expect keys: my_acpl, my_blunder_count, my_avg_secs_per_move, result
            # If they don't exist (Raw JSON), we'd need the parser logic.
            # To keep this SDK lightweight, we assume the Input List has these keys.
            # (Your Next.js backend should parse the Lichess JSON into these metrics)
            
            row = {
                'created_at': created_at,
                'last_move_at': last_move,
                'my_acpl': g.get('my_acpl', 50), # Default average
                'my_blunder_count': g.get('my_blunder_count', 0),
                'my_avg_secs_per_move': g.get('my_avg_secs_per_move', 5.0),
                'result': g.get('result', 0.5),
                'rating_diff': rating_diff
            }
            rows.append(row)
            
        df = pd.DataFrame(rows)
        df = df.sort_values('created_at').reset_index(drop=True)
        
        # 2. Feature Engineering (Session Context)
        # Session ID is 1 (we assume input is one session)
        df['games_played'] = df.index + 1
        df['session_pl'] = df['rating_diff'].cumsum()
        
        # Streak
        df['is_loss'] = (df['result'] == 0.0).astype(int)
        streak_group = (df['is_loss'] == 0).cumsum()
        df['loss_streak'] = df.groupby(streak_group).cumcount()
        df.loc[df['is_loss'] == 0, 'loss_streak'] = 0
        
        # Rolling
        df['roll_5_acpl_mean'] = df['my_acpl'].rolling(5, min_periods=1).mean().fillna(50)
        df['roll_5_time_per_move'] = df['my_avg_secs_per_move'].rolling(5, min_periods=1).mean().fillna(5)
        
        # Speed vs Start
        first_speed = df['my_avg_secs_per_move'].iloc[0] + 0.001
        df['speed_vs_start'] = df['my_avg_secs_per_move'] / first_speed
        
        # Break Time (Log)
        df['prev_game_end'] = df['last_move_at'].shift(1)
        df['break_time'] = (df['created_at'] - df['prev_game_end']).dt.total_seconds()
        df['break_time'] = df['break_time'].fillna(0).clip(lower=0)
        df['log_break_time'] = np.log1p(df['break_time'])
        
        # Time of Day
        try:
            tz = pytz.timezone(self.local_tz)
            local_time = df['created_at'].dt.tz_convert(tz)
        except:
            local_time = df['created_at']
            
        df['tod_label'] = local_time.dt.hour.apply(self._assign_time_of_day)
        for t in ['morning', 'midday', 'evening', 'night']:
            df[f'tod_{t}'] = (df['tod_label'] == t).astype(int)
            
        return df

    # ------------------------------------------------------------------
    # 2. TRAINING API
    # ------------------------------------------------------------------
    def train_from_file(self, csv_path=DEFAULT_DATA, save_path=DEFAULT_MODEL):
        """
        Loads processed CSV, trains model, optimizes threshold, saves assets.
        """
        print(f"Loading training data from {csv_path}...")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"{csv_path} not found.")
            
        df = pd.read_csv(csv_path)
        
        # Validate columns
        missing = [c for c in self.feature_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing features in CSV: {missing}")
            
        X = df[self.feature_cols]
        y = df['target']
        groups = df['session_id']
        
        print(f"Training on {len(df)} games...")
        
        # 1. Train
        self.model = xgb.XGBClassifier(**self.params)
        self.model.fit(X, y)
        
        # 2. Optimize
        print("Optimizing P/L Threshold...")
        df['tilt_prob'] = self.model.predict_proba(X)[:, 1]
        best_thresh, best_pl, improvement = self._optimize_threshold(df)
        
        # 3. Config
        self.config = {
            'features': self.feature_cols,
            'params': self.params,
            'threshold': best_thresh,
            'pl_improvement_est': improvement
        }
        
        # 4. Save
        self.save(save_path)
        self._save_summary(df, best_thresh, best_pl, improvement)
        
        print(f"✅ Training Complete.")
        print(f"   Best Threshold: {best_thresh:.2f}")
        print(f"   Est. P/L Gain: {improvement:+.0f}")

    def _optimize_threshold(self, df):
        if 'rating_diff' not in df.columns: return 0.5, 0, 0
        
        thresholds = np.arange(0.30, 0.90, 0.02)
        baseline = df['rating_diff'].sum()
        best_pl = -float('inf')
        best_t = 0.5
        
        for t in thresholds:
            sim_pl = 0
            for _, grp in df.groupby('session_id'):
                # Find first stop signal
                stops = grp.index[grp['tilt_prob'] > t]
                if not stops.empty:
                    # Stop AFTER this game
                    idx = stops[0]
                    # Use .loc to include the stop game. 
                    # We need the position in the group, so reset index is safer
                    g_reset = grp.reset_index()
                    local_idx = g_reset.index[g_reset['index'] == idx][0]
                    played = g_reset.iloc[:local_idx+1]
                    sim_pl += played['rating_diff'].sum()
                else:
                    sim_pl += grp['rating_diff'].sum()
            
            if sim_pl > best_pl:
                best_pl = sim_pl
                best_t = t
                
        return best_t, best_pl, (best_pl - baseline)

    def _save_summary(self, df, thresh, pl, improve):
        summary_path = Path(DEFAULT_SUMMARY)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        
        txt = f"""
        TILT MODEL SUMMARY
        ==================
        Date: {pd.Timestamp.now()}
        Training Rows: {len(df)}
        
        Optimal Threshold: {thresh:.2f}
        Projected P/L:     {pl:.0f}
        Net Improvement:   {improve:+.0f}
        
        Features: {self.feature_cols}
        """
        with open(summary_path, 'w') as f:
            f.write(txt)

    # ------------------------------------------------------------------
    # 3. INFERENCE API
    # ------------------------------------------------------------------
    def predict(self, session_history_json):
        """
        Predicts tilt risk for the LATEST game in the provided history.
        session_history_json: List of game dictionaries.
        """
        if self.model is None:
            raise ValueError("Model not loaded.")
            
        # Enrich
        df = self._enrich_json(session_history_json)
        if df.empty: return None
        
        # Select last game
        last_row = df.iloc[[-1]]
        
        # Ensure columns align
        X = last_row[self.feature_cols]
        
        # Predict
        prob = self.model.predict_proba(X)[0, 1]
        thresh = self.config.get('threshold', 0.5)
        
        return {
            "stop_probability": float(prob),
            "threshold": float(thresh),
            "should_stop": bool(prob > thresh),
            "features": X.to_dict(orient='records')[0]
        }

    # ------------------------------------------------------------------
    # 4. PERSISTENCE
    # ------------------------------------------------------------------
    def save(self, model_path):
        model_path = Path(model_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.model.save_model(model_path)
        
        config_path = model_path.parent / "tilt_config.joblib"
        joblib.dump(self.config, config_path)

    def load(self, model_path=DEFAULT_MODEL):
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found at {model_path}")
            
        self.model = xgb.XGBClassifier()
        self.model.load_model(model_path)
        
        config_path = model_path.parent / "tilt_config.joblib"
        if config_path.exists():
            self.config = joblib.load(config_path)
            self.feature_cols = self.config.get('features', self.feature_cols)

if __name__ == "__main__":
    # Run pipeline when executed as script
    model = TiltModel()
    model.train_from_file()