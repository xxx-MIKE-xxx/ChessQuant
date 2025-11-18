import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import joblib

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, log_loss
from sklearn.utils import shuffle as sk_shuffle

# ----------------------------------
# CONFIG
# ----------------------------------

USERNAME = "julio_amigo_dos"

INPUT_PKL = Path("data") / "formatted_data" / "games_enriched_full.pkl"


OUTPUT_DATASET_DIR = Path("output_dataset")
OUTPUT_DATASET_DIR.mkdir(exist_ok=True)

OUTPUT_ANALYSIS_ROOT = Path("output_analysis")
OUTPUT_ANALYSIS_DIR = OUTPUT_ANALYSIS_ROOT / "stop_policy_ml_raw_features"
OUTPUT_ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

MODELS_DIR = Path("assets")
MODELS_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42
TEST_FRACTION = 0.2
VAL_FRACTION = 0.2  # of remaining after test split


# ----------------------------------
# HELPERS
# ----------------------------------

def build_best_stop_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each session:
      - compute cumulative P/L G_t = sum_{i<=t} pl_i
      - label 1 where G_t reaches its maximum in that session (best is best)
      - label 0 elsewhere
    """
    df = df.sort_values(["session_id", "game_in_session"]).copy()

    # cumulative pl per session
    df["cum_pl_session"] = df.groupby("session_id")["pl"].cumsum()

    # max cum_pl per session
    max_pl = df.groupby("session_id")["cum_pl_session"].transform("max")

    # label 1 where cum_pl_session == max_pl
    df["label_best_stop_pl"] = (df["cum_pl_session"] == max_pl).astype(int)
    return df


def train_val_test_split_by_session(df: pd.DataFrame, random_state: int = 42):
    """
    Split sessions into train/val/test sets to avoid leakage.
    """
    rng = np.random.RandomState(random_state)
    sessions = df["session_id"].unique()
    rng.shuffle(sessions)

    n = len(sessions)
    n_test = int(np.floor(TEST_FRACTION * n))
    n_val = int(np.floor(VAL_FRACTION * (n - n_test)))

    test_sessions = sessions[:n_test]
    val_sessions = sessions[n_test : n_test + n_val]
    train_sessions = sessions[n_test + n_val :]

    df_train = df[df["session_id"].isin(train_sessions)].copy()
    df_val = df[df["session_id"].isin(val_sessions)].copy()
    df_test = df[df["session_id"].isin(test_sessions)].copy()

    return df_train, df_val, df_test


def prepare_raw_feature_matrix(df: pd.DataFrame):
    """
    Select numeric "raw" features (no tilt columns) for the model.

    We use:
      - game_in_session, session_len, phase
      - result_score, pl, rolling_result_w5
      - rating_diff, my_rating, opp_rating
      - time_of_day, day_of_week
      - rated (as int)
    """
    df_feats = df.copy()

    if "rolling_result_w5" not in df_feats.columns:
        # ensure it exists; fallback rolling if needed
        df_feats = df_feats.sort_values(["session_id", "game_in_session"])
        df_feats["rolling_result_w5"] = (
            df_feats.groupby("session_id")["result_score"]
            .rolling(window=5, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
        )

    # cast 'rated' to int if present
    if "rated" in df_feats.columns:
        df_feats["rated_int"] = df_feats["rated"].astype(int)
    else:
        df_feats["rated_int"] = 0

    feature_cols = [
        "game_in_session",
        "session_len",
        "phase",
        "result_score",
        "pl",
        "rolling_result_w5",
        "rating_diff",
        "my_rating",
        "opp_rating",
        "time_of_day",
        "day_of_week",
        "rated_int",
    ]

    # Ensure all feature cols exist
    missing = [c for c in feature_cols if c not in df_feats.columns]
    if missing:
        raise ValueError(f"Missing raw feature columns: {missing}")

    X = df_feats[feature_cols].astype(float).values
    return X, feature_cols


def compute_class_weights(y: np.ndarray):
    """
    Compute simple balanced weights for binary labels.
    """
    n = len(y)
    pos = (y == 1).sum()
    neg = n - pos
    if pos == 0 or neg == 0:
        return np.ones_like(y, dtype=float)

    w_pos = n / (2.0 * pos)
    w_neg = n / (2.0 * neg)

    weights = np.where(y == 1, w_pos, w_neg)
    return weights


# ----------------------------------
# MAIN
# ----------------------------------

def main():
    if not INPUT_PKL.exists():
        raise FileNotFoundError(
            f"{INPUT_PKL} not found. Run tilt_detection_unified_tilt_score_games.py first."
        )

    print(f"Loading games dataset from {INPUT_PKL} ...")
    df = pd.read_pickle(INPUT_PKL)

    required_cols = ["session_id", "game_in_session", "pl", "result_score", "rating_diff"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in input DF: {missing}")

    # 1) Build labels
    print("Building best-stop labels based on cumulative P/L (best is best)...")
    df = build_best_stop_labels(df)

    # 2) Split by session
    print("Splitting into train/val/test by session...")
    df_train, df_val, df_test = train_val_test_split_by_session(df, random_state=RANDOM_STATE)

    print(
        f"Sessions: train={df_train['session_id'].nunique()}, "
        f"val={df_val['session_id'].nunique()}, "
        f"test={df_test['session_id'].nunique()}"
    )

    # 3) Prepare feature matrices
    print("Preparing raw feature matrices...")
    X_train, feature_cols = prepare_raw_feature_matrix(df_train)
    X_val, _ = prepare_raw_feature_matrix(df_val)
    X_test, _ = prepare_raw_feature_matrix(df_test)

    y_train = df_train["label_best_stop_pl"].values
    y_val = df_val["label_best_stop_pl"].values
    y_test = df_test["label_best_stop_pl"].values

    # 4) Compute sample weights for training
    sample_weight_train = compute_class_weights(y_train)

    # 5) Train HistGradientBoostingClassifier (probabilistic output)
    print("Training HistGradientBoostingClassifier on raw features...")
    clf = HistGradientBoostingClassifier(
        loss="log_loss",
        max_depth=3,
        learning_rate=0.05,
        max_iter=400,
        random_state=RANDOM_STATE,
    )

    clf.fit(X_train, y_train, sample_weight=sample_weight_train)

    # 6) Evaluate
    def eval_split(name, X, y_true):
        y_prob = clf.predict_proba(X)[:, 1]
        try:
            auc = roc_auc_score(y_true, y_prob)
        except ValueError:
            auc = float("nan")
        try:
            ap = average_precision_score(y_true, y_prob)
        except ValueError:
            ap = float("nan")
        try:
            # clip probabilities manually to avoid log(0) issues
            y_prob_safe = np.clip(y_prob, 1e-7, 1 - 1e-7)
            ll = log_loss(y_true, y_prob_safe)
        except ValueError:
            ll = float("nan")
        print(
            f"[{name}] "
            f"n={len(y_true)}, "
            f"pos_rate={y_true.mean():.3f}, "
            f"ROC-AUC={auc:.3f}, "
            f"PR-AUC={ap:.3f}, "
            f"logloss={ll:.3f}"
        )
        return y_prob


    print("Evaluating model...")
    y_train_prob = eval_split("train", X_train, y_train)
    y_val_prob = eval_split("val", X_val, y_val)
    y_test_prob = eval_split("test", X_test, y_test)

    # 7) Save model
    model_path = MODELS_DIR / "stop_policy_raw_features_hgb.joblib"
    joblib.dump(
        {
            "model": clf,
            "feature_cols": feature_cols,
            "random_state": RANDOM_STATE,
        },
        model_path,
    )
    print(f"Saved raw-features stop-policy model to {model_path}")

    # 8) Attach test predictions back to df (for inspection)
    df_train = df_train.copy()
    df_val = df_val.copy()
    df_test = df_test.copy()

    df_train["stop_now_prob_raw"] = y_train_prob
    df_val["stop_now_prob_raw"] = y_val_prob
    df_test["stop_now_prob_raw"] = y_test_prob

    df_all = pd.concat([df_train, df_val, df_test], axis=0).sort_values(
        ["session_id", "game_in_session"]
    )

    out_pkl = OUTPUT_DATASET_DIR / "games_enriched_full_stop_policy_raw.pkl"
    out_csv = OUTPUT_DATASET_DIR / "games_enriched_full_stop_policy_raw.csv"
    df_all.to_pickle(out_pkl)
    df_all.to_csv(out_csv, index=False)
    print(f"Saved dataset with raw stop probabilities to:")
    print(f"  {out_pkl}")
    print(f"  {out_csv}")

    # 9) Summary file
    summary_path = OUTPUT_ANALYSIS_DIR / "stop_policy_raw_summary.txt"
    with summary_path.open("w", encoding="utf-8") as f:
        f.write(f"Stop-policy ML (raw features) summary ({datetime.utcnow().isoformat()} UTC)\n\n")
        f.write(f"Random state: {RANDOM_STATE}\n")
        f.write(f"Train sessions: {df_train['session_id'].nunique()}\n")
        f.write(f"Val sessions  : {df_val['session_id'].nunique()}\n")
        f.write(f"Test sessions : {df_test['session_id'].nunique()}\n\n")
        f.write("Label (best stop) positive rate by split:\n")
        f.write(f"  train: {y_train.mean():.4f}\n")
        f.write(f"  val  : {y_val.mean():.4f}\n")
        f.write(f"  test : {y_test.mean():.4f}\n")

    print(f"Saved summary to {summary_path}")
    print("Done.")


if __name__ == "__main__":
    main()
