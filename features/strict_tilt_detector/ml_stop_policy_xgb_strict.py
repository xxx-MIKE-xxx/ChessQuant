"""
Strict stop-policy model (XGB) using a future-regret band labeling.

Goal:
    Train a model that says "STOP" on games where stopping would be
    near-optimal for final rating in that session:

        - Best: at the peak of cumulative P/L.
        - Second-best: just after the peak, while you are still close
          to the best you could have achieved.

Labeling scheme (per session):

    For each game t:
        G_t = cumulative sum of pl up to game t
        H_t = max_{s >= t} G_s  (best possible cum P/L if you keep playing)
        regret_t = H_t - G_t

    label_strict_stop(t) = 1 if:
        regret_t <= EPSILON_REGRET
        and G_t >= MIN_PROFIT
        and game_in_session >= MIN_GAMES
    else 0

This:
    - Marks the "stop band" around the session's best achievable P/L.
    - Allows multiple 1's in a session if multiple points are near-optimal.
    - Penalizes stopping too early or too late.

Model:
    XGBClassifier (binary:logistic), trained on raw game/session features.

Inputs:
    data/formatted_data/games_enriched_full.pkl

Outputs:
    output_dataset/games_enriched_full_stop_policy_strict.pkl
    output_dataset/games_enriched_full_stop_policy_strict.csv

Model:
    assets/stop_policy_raw_features_xgb_strict.joblib

Summary:
    output_analysis/stop_policy_ml_raw_features_xgb_strict/stop_policy_raw_summary.txt
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import joblib

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    log_loss,
)

from xgboost import XGBClassifier


# ----------------------------------
# CONFIG
# ----------------------------------

USERNAME = "julio_amigo_dos"

INPUT_PKL = Path("data") / "formatted_data" / "games_enriched_full.pkl"

OUTPUT_DATASET_DIR = Path("output_dataset")
OUTPUT_DATASET_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_ANALYSIS_ROOT = Path("output_analysis")
OUTPUT_ANALYSIS_DIR = OUTPUT_ANALYSIS_ROOT / "stop_policy_ml_raw_features_xgb_strict"
OUTPUT_ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

MODELS_DIR = Path("assets")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
TEST_FRACTION = 0.2
VAL_FRACTION = 0.2  # of remaining after test split

# Labeling hyperparameters for the strict stop band
EPSILON_REGRET = 20.0   # how many rating points below the best possible is still "good enough"
MIN_PROFIT = 0.0        # require cum_pl_session >= MIN_PROFIT to consider stopping
MIN_GAMES = 2           # don't recommend stopping before this game index in a session


# ----------------------------------
# LABELING: strict stop band via future regret
# ----------------------------------

def build_strict_stop_labels(
    df: pd.DataFrame,
    epsilon_regret: float = EPSILON_REGRET,
    min_profit: float = MIN_PROFIT,
    min_games: int = MIN_GAMES,
) -> pd.DataFrame:
    """
    Label games as "strict stop" (1) if stopping there is near-optimal
    for that session, as measured by future regret.

    For each session:
        - cum_pl_session G_t = cumsum(pl)
        - future_max_pl H_t = max_{s >= t} G_s  (reverse cumulative max)
        - regret_if_stop_t = H_t - G_t

    Then:
        label_strict_stop = 1 if:
            regret_if_stop_t <= epsilon_regret
            and cum_pl_session >= min_profit
            and game_in_session >= min_games
        else 0
    """
    df = df.sort_values(["session_id", "game_in_session"]).copy()

    # Cumulative P/L per session
    df["cum_pl_session"] = df.groupby("session_id")["pl"].cumsum()

    # Future max per session: reverse cummax over cum_pl_session
    def _future_max(series: pd.Series) -> pd.Series:
        return series[::-1].cummax()[::-1]

    df["future_max_pl"] = df.groupby("session_id")["cum_pl_session"].transform(_future_max)

    # regret if you stop now
    df["regret_if_stop"] = df["future_max_pl"] - df["cum_pl_session"]

    # strict stop band condition
    good_stop = (
        (df["regret_if_stop"] <= epsilon_regret) &
        (df["cum_pl_session"] >= min_profit) &
        (df["game_in_session"] >= min_games)
    )

    df["label_strict_stop"] = good_stop.astype(int)
    return df


# ----------------------------------
# SPLIT
# ----------------------------------

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


# ----------------------------------
# FEATURES
# ----------------------------------

def prepare_raw_feature_matrix(df: pd.DataFrame):
    """
    Same raw feature set as your other stop/tilt models, for consistency.

    Uses:
      - game_in_session, session_len, phase
      - result_score, pl, rolling_result_w5
      - rating_diff, my_rating, opp_rating
      - time_of_day, day_of_week
      - rated_int
    """
    df_feats = df.copy()

    if "rolling_result_w5" not in df_feats.columns:
        df_feats = df_feats.sort_values(["session_id", "game_in_session"])
        df_feats["rolling_result_w5"] = (
            df_feats.groupby("session_id")["result_score"]
            .rolling(window=5, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
        )

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

    missing = [c for c in feature_cols if c not in df_feats.columns]
    if missing:
        raise ValueError(f"Missing raw feature columns: {missing}")

    X = df_feats[feature_cols].astype(float).values
    return X, feature_cols


def compute_class_weights(y: np.ndarray):
    """
    Balanced weights for positive/negative classes.
    """
    n = len(y)
    pos = (y == 1).sum()
    neg = n - pos
    if pos == 0 or neg == 0:
        return np.ones_like(y, dtype=float)

    w_pos = n / (2.0 * pos)
    w_neg = n / (2.0 * neg)

    return np.where(y == 1, w_pos, w_neg)


# ----------------------------------
# METRICS + THRESHOLD SEARCH
# ----------------------------------

def eval_split(name, X, y_true, clf):
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


def find_best_threshold_f1(y_true: np.ndarray, y_prob: np.ndarray):
    """
    Scan thresholds in [0.01, 0.99] to find the one that maximizes F1.
    """
    best_thr = 0.5
    best_f1 = -1.0

    thresholds = np.linspace(0.01, 0.99, 99)
    for thr in thresholds:
        y_pred = (y_prob >= thr).astype(int)
        tp = np.logical_and(y_pred == 1, y_true == 1).sum()
        fp = np.logical_and(y_pred == 1, y_true == 0).sum()
        fn = np.logical_and(y_pred == 0, y_true == 1).sum()

        if tp == 0 and fp == 0 and fn == 0:
            continue

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        if precision + recall == 0:
            continue
        f1 = 2 * precision * recall / (precision + recall)

        if f1 > best_f1:
            best_f1 = f1
            best_thr = float(thr)

    return best_thr, best_f1


# ----------------------------------
# MAIN
# ----------------------------------

def main():
    print("=== Strict stop-policy XGB training (future-regret band) ===")

    # 0) Load data
    if not INPUT_PKL.exists():
        raise FileNotFoundError(
            f"{INPUT_PKL} not found. Generate formatted games first."
        )

    print(f"Loading games dataset from {INPUT_PKL} ...")
    df = pd.read_pickle(INPUT_PKL)

    required_cols = ["session_id", "game_in_session", "pl", "result_score", "rating_diff"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in input DF: {missing}")

    # 1) Build strict stop-band labels
    print(
        f"Building strict stop labels using future regret band "
        f"(EPSILON_REGRET={EPSILON_REGRET}, MIN_PROFIT={MIN_PROFIT}, MIN_GAMES={MIN_GAMES})..."
    )
    df = build_strict_stop_labels(
        df,
        epsilon_regret=EPSILON_REGRET,
        min_profit=MIN_PROFIT,
        min_games=MIN_GAMES,
    )

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

    y_train = df_train["label_strict_stop"].values
    y_val = df_val["label_strict_stop"].values
    y_test = df_test["label_strict_stop"].values

    print("Label positive rates (strict stop band):")
    print(f"  train: {y_train.mean():.4f}")
    print(f"  val  : {y_val.mean():.4f}")
    print(f"  test : {y_test.mean():.4f}")

    # 4) Class weights
    sample_weight_train = compute_class_weights(y_train)

    # 5) Train XGBClassifier
    print("Training XGBClassifier (strict stop band)...")
    clf = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        n_estimators=400,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        tree_method="hist",
    )
    clf.fit(X_train, y_train, sample_weight=sample_weight_train)

    # 6) Evaluate & get probabilities
    print("Evaluating model...")
    y_train_prob = eval_split("train", X_train, y_train, clf)
    y_val_prob = eval_split("val", X_val, y_val, clf)
    y_test_prob = eval_split("test", X_test, y_test, clf)

    # 7) Best threshold on val
    best_thr, best_f1 = find_best_threshold_f1(y_val, y_val_prob)
    print(f"Best decision threshold on val (F1, strict): {best_thr:.3f}, F1={best_f1:.3f}")

    # Evaluate F1 on test at that threshold (just for info)
    y_test_pred_bin = (y_test_prob >= best_thr).astype(int)
    tp = np.logical_and(y_test_pred_bin == 1, y_test == 1).sum()
    fp = np.logical_and(y_test_pred_bin == 1, y_test == 0).sum()
    fn = np.logical_and(y_test_pred_bin == 0, y_test == 1).sum()
    if tp > 0:
        prec_test = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec_test = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_test = (
            2 * prec_test * rec_test / (prec_test + rec_test)
            if (prec_test + rec_test) > 0
            else 0.0
        )
    else:
        f1_test = 0.0
    print(f"[test] F1 at threshold {best_thr:.3f}: {f1_test:.3f}")

    # 8) Save model payload
    model_path = MODELS_DIR / "stop_policy_raw_features_xgb_strict.joblib"
    joblib.dump(
        {
            "model": clf,
            "feature_cols": feature_cols,
            "random_state": RANDOM_STATE,
            "decision_threshold": best_thr,
            "label_type": "strict_stop_band",
            "epsilon_regret": EPSILON_REGRET,
            "min_profit": MIN_PROFIT,
            "min_games": MIN_GAMES,
        },
        model_path,
    )
    print(f"Saved strict stop-policy model to {model_path}")

    # 9) Attach predictions back to df (for inspection)
    df_train = df_train.copy()
    df_val = df_val.copy()
    df_test = df_test.copy()

    df_train["stop_now_prob_strict"] = y_train_prob
    df_val["stop_now_prob_strict"] = y_val_prob
    df_test["stop_now_prob_strict"] = y_test_prob

    df_train["stop_now_strict_binary"] = (df_train["stop_now_prob_strict"] >= best_thr).astype(int)
    df_val["stop_now_strict_binary"] = (df_val["stop_now_prob_strict"] >= best_thr).astype(int)
    df_test["stop_now_strict_binary"] = (df_test["stop_now_prob_strict"] >= best_thr).astype(int)

    df_all = pd.concat([df_train, df_val, df_test], axis=0).sort_values(
        ["session_id", "game_in_session"]
    )

    out_pkl = OUTPUT_DATASET_DIR / "games_enriched_full_stop_policy_strict.pkl"
    out_csv = OUTPUT_DATASET_DIR / "games_enriched_full_stop_policy_strict.csv"
    df_all.to_pickle(out_pkl)
    df_all.to_csv(out_csv, index=False)
    print("Saved dataset with strict stop probabilities and binary decisions to:")
    print(f"  {out_pkl}")
    print(f"  {out_csv}")

    # 10) Summary
    summary_path = OUTPUT_ANALYSIS_DIR / "stop_policy_raw_summary.txt"
    with summary_path.open("w", encoding="utf-8") as f:
        f.write(
            f"Stop-policy ML (XGB raw features, STRICT stop band) summary "
            f"({datetime.utcnow().isoformat()} UTC)\n\n"
        )
        f.write(f"Random state: {RANDOM_STATE}\n")
        f.write(f"Train sessions: {df_train['session_id'].nunique()}\n")
        f.write(f"Val sessions  : {df_val['session_id'].nunique()}\n")
        f.write(f"Test sessions : {df_test['session_id'].nunique()}\n\n")
        f.write("Label (strict stop band) positive rate by split:\n")
        f.write(f"  train: {y_train.mean():.4f}\n")
        f.write(f"  val  : {y_val.mean():.4f}\n")
        f.write(f"  test : {y_test.mean():.4f}\n\n")
        f.write(f"Decision threshold (val F1 max): {best_thr:.4f}\n")
        f.write(f"Test F1 at threshold           : {f1_test:.4f}\n")
        f.write("\nLabeling hyperparameters:\n")
        f.write(f"  EPSILON_REGRET: {EPSILON_REGRET}\n")
        f.write(f"  MIN_PROFIT    : {MIN_PROFIT}\n")
        f.write(f"  MIN_GAMES     : {MIN_GAMES}\n")

    print(f"Saved summary to {summary_path}")
    print("Done.")


if __name__ == "__main__":
    main()
