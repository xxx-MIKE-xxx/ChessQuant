import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from pathlib import Path

# --- CONFIG ---
RANDOM_STATE = 42
MIN_SESSION_LENGTH = 3
STOP_THRESHOLD = 0.55
WINDOW_SIZES = [3, 5]

# ---------------------------------------------------------
# 1. ROBUST PATH FINDING
# ---------------------------------------------------------
def find_data_file(filename):
    """
    Tries to find the dataset by looking in common relative paths.
    """
    # 1. Where the script is located
    script_dir = Path(__file__).resolve().parent
    
    # 2. Where the command was run from (CWD)
    cwd = Path.cwd()
    
    candidates = [
        # Direct relative paths
        script_dir / filename,
        cwd / filename,
        
        # Standard project structure (running from root)
        cwd / "data" / "formatted_data" / filename,
        
        # Standard project structure (running from features/tilt_detector)
        script_dir.parent.parent / "data" / "formatted_data" / filename,
        
        # Just in case
        cwd / "formatted_data" / filename,
    ]
    
    print("Searching for data in:")
    for c in candidates:
        # print(f"  ? {c}") # Uncomment to debug
        if c.exists():
            return c
            
    return None

# ---------------------------------------------------------
# 2. DATA LOADING & PREP
# ---------------------------------------------------------
def load_and_preprocess(csv_path):
    print(f"Found data at: {csv_path}")
    df = pd.read_csv(csv_path)
    df['created_at'] = pd.to_datetime(df['created_at'], format='mixed', utc=True)
    
    if 'session_id' not in df.columns:
        print("Grouping sessions...")
        df = df.sort_values('created_at').reset_index(drop=True)
        df['time_diff'] = df['created_at'].diff()
        df['new_session'] = (df['time_diff'] > pd.Timedelta(hours=1)).astype(int)
        df['session_id'] = df['new_session'].cumsum()
        
    df = df.sort_values(['session_id', 'created_at']).reset_index(drop=True)
    df['game_in_session'] = df.groupby('session_id').cumcount()
    df['session_cum_pl'] = df.groupby('session_id')['pl'].cumsum()
    
    # Fill NAs
    df['my_acpl'] = df['my_acpl'].fillna(50)
    df['my_blunder_count'] = df['my_blunder_count'].fillna(0)
    df['my_avg_secs_per_move'] = df['my_avg_secs_per_move'].fillna(df['my_avg_secs_per_move'].mean())
    
    return df

def add_features(df):
    df_feat = df.copy()
    grp = df_feat.groupby('session_id')
    
    for w in WINDOW_SIZES:
        df_feat[f'roll_{w}_pl_sum'] = grp['pl'].transform(lambda x: x.rolling(w).sum())
        df_feat[f'roll_{w}_acpl_mean'] = grp['my_acpl'].transform(lambda x: x.rolling(w).mean())
        df_feat[f'roll_{w}_time_per_move'] = grp['my_avg_secs_per_move'].transform(lambda x: x.rolling(w).mean())

    # Fatigue / Speed (Current speed vs session start speed)
    first_game_speed = grp['my_avg_secs_per_move'].transform('first')
    df_feat['speed_vs_start'] = df_feat['my_avg_secs_per_move'] / (first_game_speed + 1e-5)
    
    df_feat['current_session_pl'] = df_feat['session_cum_pl']
    df_feat['games_played'] = df_feat['game_in_session'] + 1
    
    numeric_cols = [c for c in df_feat.columns if 'roll_' in c] + ['speed_vs_start']
    df_feat[numeric_cols] = df_feat[numeric_cols].fillna(0)
    
    # --- MODEL A: THE BANKER (Has Score Info) ---
    cols_all = [
        'pl', 'current_session_pl', 
        'my_acpl', 'my_blunder_count', 'my_avg_secs_per_move', 'speed_vs_start',
        'games_played'
    ] + numeric_cols

    # --- MODEL B: PURE TILT (Blind to Score) ---
    # Removes all PL related columns
    cols_pure = [
        'my_acpl', 'my_blunder_count', 'my_avg_secs_per_move', 'speed_vs_start',
        'games_played'
    ] + [c for c in numeric_cols if 'pl' not in c]

    return df_feat, cols_all, cols_pure

def build_target(df):
    # Target = 1 if this is the Session Max
    session_max = df.groupby('session_id')['session_cum_pl'].transform('max')
    is_max = (df['session_cum_pl'] == session_max)
    
    df_temp = df.copy()
    df_temp['is_max'] = is_max
    
    # First max only
    target_idx = df_temp[df_temp['is_max']].groupby('session_id')['game_in_session'].idxmin()
    
    df['target'] = 0
    df.loc[target_idx.values, 'target'] = 1 
    return df

# ---------------------------------------------------------
# 3. EXECUTION & PLOTTING
# ---------------------------------------------------------
def run_ablation(df):
    print("\n" + "="*50)
    print("RUNNING ABLATION TEST: 'Banker' vs 'Tilt Detector'")
    print("="*50)
    
    # Split
    uniq_sess = df['session_id'].unique()
    split = int(len(uniq_sess) * 0.75)
    train_ids = uniq_sess[:split]
    test_ids = uniq_sess[split:]
    
    # Train Data
    df_train = df[df['session_id'].isin(train_ids)].copy()
    long_sess = df_train['session_id'].value_counts()
    long_sess = long_sess[long_sess >= MIN_SESSION_LENGTH].index
    df_train = df_train[df_train['session_id'].isin(long_sess)]
    
    df_train = build_target(df_train)
    df_train, cols_all, cols_pure = add_features(df_train)
    
    # Test Data
    df_test = df[df['session_id'].isin(test_ids)].copy()
    df_test = build_target(df_test)
    df_test, _, _ = add_features(df_test)
    
    # --- TRAINING ---
    print("\n1. Training Model A (The Banker - Sees Score)...")
    model_a = HistGradientBoostingClassifier(
        max_iter=500, max_depth=5, early_stopping=True, 
        class_weight='balanced', random_state=RANDOM_STATE
    )
    model_a.fit(df_train[cols_all], df_train['target'])
    
    print("2. Training Model B (Pure Tilt - Blind to Score)...")
    model_b = HistGradientBoostingClassifier(
        max_iter=500, max_depth=5, early_stopping=True, 
        class_weight='balanced', random_state=RANDOM_STATE
    )
    model_b.fit(df_train[cols_pure], df_train['target'])
    
    # --- FEATURE IMPORTANCE (Model B) ---
    print("\nAnalyzing Pure Tilt Features...")
    imp_b = permutation_importance(model_b, df_train[cols_pure], df_train['target'], n_repeats=5, random_state=RANDOM_STATE)
    
    # Plot Importance
    feat_imp = pd.DataFrame({'feature': cols_pure, 'importance': imp_b.importances_mean})
    feat_imp = feat_imp.sort_values('importance', ascending=False).head(10)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=feat_imp, x='importance', y='feature', hue='feature', palette='viridis', legend=False)
    plt.title("What Indicates Tilt? (Top Predictors for Model B)")
    plt.tight_layout()
    plt.savefig("feature_importance_pure_tilt.png")
    print(">> Saved feature importance plot to 'feature_importance_pure_tilt.png'")

    # --- SIMULATION ---
    results = []
    
    for sid, group in df_test.groupby('session_id'):
        group = group.sort_values('game_in_session').reset_index(drop=True)
        
        # Predictions
        probs_a = model_a.predict_proba(group[cols_all])[:, 1]
        probs_b = model_b.predict_proba(group[cols_pure])[:, 1]
        
        # Policy A
        stop_a = len(group) - 1
        for i in range(len(group)):
            if probs_a[i] > STOP_THRESHOLD:
                stop_a = i; break
                
        # Policy B
        stop_b = len(group) - 1
        for i in range(len(group)):
            if probs_b[i] > STOP_THRESHOLD:
                stop_b = i; break
        
        actual_end = group.iloc[-1]['session_cum_pl']
        pl_a = group.iloc[stop_a]['session_cum_pl']
        pl_b = group.iloc[stop_b]['session_cum_pl']
        optimal = group['session_cum_pl'].max()
        
        results.append({
            'session_id': sid,
            'actual': actual_end,
            'model_a': pl_a,
            'model_b': pl_b,
            'optimal': optimal
        })
        
    res_df = pd.DataFrame(results)
    
    # --- METRICS ---
    print("\n" + "-"*30)
    print("FINAL COMPARISON")
    print("-"*30)
    print(f"Total Sessions:      {len(res_df)}")
    print(f"Actual Result:       {res_df['actual'].sum()}")
    print(f"Model A (Banker):    {res_df['model_a'].sum()} (Diff: {res_df['model_a'].sum() - res_df['actual'].sum()})")
    print(f"Model B (Pure Tilt): {res_df['model_b'].sum()} (Diff: {res_df['model_b'].sum() - res_df['actual'].sum()})")
    print(f"Optimal (God Mode):  {res_df['optimal'].sum()}")

    # --- PLOT COMPARISON ---
    plt.figure(figsize=(12, 6))
    plt.plot(res_df['actual'].cumsum(), label='Actual Play', color='gray', linestyle=':', linewidth=2)
    plt.plot(res_df['model_a'].cumsum(), label='Model A (Smart Banker)', color='blue', linewidth=2)
    plt.plot(res_df['model_b'].cumsum(), label='Model B (Pure Tilt)', color='green', linewidth=2)
    plt.plot(res_df['optimal'].cumsum(), label='God Mode', color='orange', alpha=0.3, linestyle='--')
    
    plt.title("Strategy Comparison: Can we detect tilt without looking at the score?")
    plt.xlabel("Sessions Played")
    plt.ylabel("Cumulative Rating Change")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("ablation_cumulative_pl.png")
    print(">> Saved cumulative results plot to 'ablation_cumulative_pl.png'")

if __name__ == "__main__":
    csv_file = find_data_file("games_enriched_full.csv")
    if csv_file:
        df_raw = load_and_preprocess(csv_file)
        run_ablation(df_raw)
    else:
        print("CRITICAL ERROR: Could not find 'games_enriched_full.csv'")
        print("Please ensure the file exists in 'data/formatted_data/' relative to the project root.")