import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_curve,
    precision_recall_curve,
    auc,
)

# ----------------------------------
# CONFIG
# ----------------------------------

USERNAME = "julio_amigo_dos"

# Base datasets
TILT_UNIFIED_PKL = Path("output_dataset") / "games_enriched_full_tilt_unified.pkl"

RAW_STOP_PKL = Path("output_dataset") / "games_enriched_full_stop_policy_raw.pkl"
TILT_STOP_PKL = Path("output_dataset") / "games_enriched_full_stop_policy_tilt.pkl"

OUTPUT_ANALYSIS_ROOT = Path("output_analysis")
OUTPUT_ANALYSIS_DIR = OUTPUT_ANALYSIS_ROOT / "stop_policy_models_comparison"
OUTPUT_ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
TEST_FRACTION = 0.2
VAL_FRACTION = 0.2  # of remaining after test split


# ----------------------------------
# HELPERS
# ----------------------------------

def build_best_stop_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Label best stop points within each session using cumulative P/L.
    label_best_stop_pl = 1 where cum_pl_session reaches its maximum.
    """
    df = df.sort_values(["session_id", "game_in_session"]).copy()
    df["cum_pl_session"] = df.groupby("session_id")["pl"].cumsum()
    max_pl = df.groupby("session_id")["cum_pl_session"].transform("max")
    df["label_best_stop_pl"] = (df["cum_pl_session"] == max_pl).astype(int)
    return df


def train_val_test_split_by_session(df: pd.DataFrame, random_state: int = 42):
    """
    Re-create train/val/test split by session to match the model scripts.
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

    return df_train, df_val, df_test, train_sessions, val_sessions, test_sessions


def merge_predictions(df_base: pd.DataFrame, df_raw: pd.DataFrame, df_tilt: pd.DataFrame):
    """
    Merge stop_now_prob_raw and stop_now_prob_tilt into a single DF
    based on (session_id, game_in_session).
    """
    keys = ["session_id", "game_in_session"]

    df_raw_small = df_raw[keys + ["stop_now_prob_raw"]].copy()
    df_tilt_small = df_tilt[keys + ["stop_now_prob_tilt"]].copy()

    df_merged = (
        df_base.merge(df_raw_small, on=keys, how="left")
        .merge(df_tilt_small, on=keys, how="left")
    )
    return df_merged


def plot_roc_pr_curves(df_test: pd.DataFrame, out_dir: Path):
    """
    Plot ROC and Precision-Recall curves for raw vs tilt models on test set.
    """
    y_true = df_test["label_best_stop_pl"].values.astype(int)
    y_raw = df_test["stop_now_prob_raw"].values
    y_tilt = df_test["stop_now_prob_tilt"].values

    # ROC
    fpr_raw, tpr_raw, _ = roc_curve(y_true, y_raw)
    fpr_tilt, tpr_tilt, _ = roc_curve(y_true, y_tilt)
    auc_raw = auc(fpr_raw, tpr_raw)
    auc_tilt = auc(fpr_tilt, tpr_tilt)

    plt.figure()
    plt.plot(fpr_raw, tpr_raw, label=f"Raw features (AUC={auc_raw:.3f})")
    plt.plot(fpr_tilt, tpr_tilt, label=f"Tilt features (AUC={auc_tilt:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC curves – test set")
    plt.legend()
    plt.grid(True)
    roc_path = out_dir / "roc_curves_test.png"
    plt.savefig(roc_path, bbox_inches="tight")
    plt.close()
    print(f"Saved ROC curves to {roc_path}")

    # Precision-Recall
    prec_raw, rec_raw, _ = precision_recall_curve(y_true, y_raw)
    prec_tilt, rec_tilt, _ = precision_recall_curve(y_true, y_tilt)

    # Compute AUC under PR curve
    def pr_auc(rec, prec):
        # sklearn has average_precision_score already, but we can integrate directly
        # rec is sorted from 0..1, so we can do simple trapezoid
        return auc(rec, prec)

    pr_auc_raw = pr_auc(rec_raw, prec_raw)
    pr_auc_tilt = pr_auc(rec_tilt, prec_tilt)

    baseline = y_true.mean()

    plt.figure()
    plt.plot(rec_raw, prec_raw, label=f"Raw features (PR-AUC={pr_auc_raw:.3f})")
    plt.plot(rec_tilt, prec_tilt, label=f"Tilt features (PR-AUC={pr_auc_tilt:.3f})")
    plt.hlines(baseline, 0, 1, linestyles="--", label=f"Baseline (pos_rate={baseline:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall curves – test set")
    plt.legend()
    plt.grid(True)
    pr_path = out_dir / "pr_curves_test.png"
    plt.savefig(pr_path, bbox_inches="tight")
    plt.close()
    print(f"Saved PR curves to {pr_path}")


def plot_example_sessions(df_test: pd.DataFrame, out_dir: Path, max_sessions: int = 3):
    """
    For a few test sessions, plot:
      - game_in_session on x-axis
      - cumulative P/L
      - stop_now_prob_raw
      - stop_now_prob_tilt
      - highlight label_best_stop_pl == 1
    """
    test_sessions = df_test["session_id"].unique()
    # pick the longest few sessions (more informative plots)
    session_lengths = (
        df_test.groupby("session_id")["game_in_session"]
        .max()
        .sort_values(ascending=False)
    )
    selected_sessions = session_lengths.index[:max_sessions]

    for sid in selected_sessions:
        sub = df_test[df_test["session_id"] == sid].sort_values("game_in_session")
        if sub.empty:
            continue

        games = sub["game_in_session"].values
        cum_pl = sub["cum_pl_session"].values
        y_true = sub["label_best_stop_pl"].values.astype(int)
        p_raw = sub["stop_now_prob_raw"].values
        p_tilt = sub["stop_now_prob_tilt"].values

        plt.figure(figsize=(10, 5))
        ax1 = plt.gca()

        # Cumulative P/L
        ax1.plot(games, cum_pl, marker="o", label="Cumulative P/L")
        ax1.set_xlabel("Game in session")
        ax1.set_ylabel("Cumulative P/L")
        ax1.grid(True)

        # Highlight true best-stop points
        best_mask = y_true == 1
        if best_mask.any():
            ax1.scatter(
                games[best_mask],
                cum_pl[best_mask],
                s=80,
                marker="*",
                label="Best stop (oracle)",
            )

        # Second axis for probabilities
        ax2 = ax1.twinx()
        ax2.plot(games, p_raw, linestyle="-", marker=None, label="stop_now_prob_raw")
        ax2.plot(games, p_tilt, linestyle="--", marker=None, label="stop_now_prob_tilt")
        ax2.set_ylabel("Stop-now probability")
        ax2.set_ylim(0, 1)

        # Combine legends
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")

        plt.title(f"Session {sid} – P/L & stop-now probabilities")
        fig_path = out_dir / f"session_{sid:03d}_stop_probs.png"
        plt.savefig(fig_path, bbox_inches="tight")
        plt.close()
        print(f"Saved session plot to {fig_path}")


# ----------------------------------
# MAIN
# ----------------------------------

def main():
    # 1) Load base DF with tilt unified (for pl, session_id, game_in_session, etc.)
    if not TILT_UNIFIED_PKL.exists():
        raise FileNotFoundError(
            f"{TILT_UNIFIED_PKL} not found. Run tilt_detection_unified_tilt_score_games.py first."
        )
    print(f"Loading base tilt-unified dataset from {TILT_UNIFIED_PKL} ...")
    df_base = pd.read_pickle(TILT_UNIFIED_PKL)

    required_cols = ["session_id", "game_in_session", "pl"]
    missing = [c for c in required_cols if c not in df_base.columns]
    if missing:
        raise ValueError(f"Missing required columns in base DF: {missing}")

    # 2) Build labels (best stop) and cumulative P/L
    print("Building best-stop labels...")
    df_base = build_best_stop_labels(df_base)

    # 3) Load prediction DFs
    if not RAW_STOP_PKL.exists():
        raise FileNotFoundError(
            f"{RAW_STOP_PKL} not found. Run ml_raw_model.py (or equivalent) first."
        )
    if not TILT_STOP_PKL.exists():
        raise FileNotFoundError(
            f"{TILT_STOP_PKL} not found. Run ml_precomputed_model.py (or equivalent) first."
        )

    print(f"Loading raw-model predictions from {RAW_STOP_PKL} ...")
    df_raw = pd.read_pickle(RAW_STOP_PKL)

    print(f"Loading tilt-model predictions from {TILT_STOP_PKL} ...")
    df_tilt = pd.read_pickle(TILT_STOP_PKL)

    # 4) Merge predictions into base DF
    print("Merging predictions into base DF...")
    df_all = merge_predictions(df_base, df_raw, df_tilt)

    # Sanity check
    if df_all["stop_now_prob_raw"].isna().any() or df_all["stop_now_prob_tilt"].isna().any():
        raise ValueError("Some predictions are NaN after merge. Check join keys.")

    # 5) Split train/val/test by session (same logic as training)
    print("Splitting into train/val/test by session (for evaluation)...")
    df_train, df_val, df_test, train_sessions, val_sessions, test_sessions = \
        train_val_test_split_by_session(df_all, random_state=RANDOM_STATE)

    print(
        f"Sessions in split: train={len(train_sessions)}, "
        f"val={len(val_sessions)}, test={len(test_sessions)}"
    )

    # 6) Plot ROC and PR curves on test split
    print("Plotting ROC & PR curves on test split...")
    plot_roc_pr_curves(df_test, OUTPUT_ANALYSIS_DIR)

    # 7) Plot a few example sessions from test split
    print("Plotting example sessions from test split...")
    plot_example_sessions(df_test, OUTPUT_ANALYSIS_DIR, max_sessions=3)

    print("Done.")


if __name__ == "__main__":
    main()
