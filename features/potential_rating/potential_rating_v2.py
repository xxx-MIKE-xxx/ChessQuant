# features/potential_rating/potential_rating.py

"""
Unified potential rating + optimizer for tilt threshold and warmup.

This file has two parts:

1) Core engine:
   - Given:
       * a tilt threshold (or None for "no tilt filter"),
       * a warmup config per time-control category,
     it:
       * loads your per-game dataset,
       * applies tilt filtering per session (optional),
       * applies warmup cuts per session,
       * merges the remaining games into a global chronological sequence,
       * runs a continuous Glicko-2 process over all games,
       * returns a final "potential rating" and some diagnostics.

2) Optimizer:
   - Grid-searches over:
       * a list of tilt thresholds, and
       * a search space of warmup configs (bullet/blitz/rapid/classic),
     and finds the combination that maximizes the final potential rating.

Outputs:
   - Default config summary:
       output/potential_rating/potential_rating_default_summary.json
   - Grid search summary:
       output/potential_rating/potential_rating_grid_search_tilt_warmup.json
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Local Glicko2 implementation (features/potential_rating/glicko2.py)
import glicko2


# =====================================================================
# CONFIG
# =====================================================================

USERNAME = "julio_amigo_dos"

# Default tilt threshold (used in "default run" when called as script)
DEFAULT_TILT_THRESHOLD: float = 0.85

# Default warmup configuration per time-control category
DEFAULT_WARMUP_CONFIG: Dict[str, int] = {
    "bullet": 3,
    "blitz": 2,
    "rapid": 1,
    "classic": 0,
}

# Candidate columns for tilt score
TILT_COL_CANDIDATES = ["tilt_score", "tilt_prob", "stop_now_prob_raw"]

# Candidate columns for time-control category (for warmup)
TIME_CATEGORY_COL_CANDIDATES = [
    "speed",          # lichess: bullet, blitz, rapid, classical
    "time_class",     # chess.com: bullet, blitz, rapid, daily
    "time_control_category",
    "game_speed",
]

# Candidate datetime columns (if present, used to globally order games)
DATETIME_COL_CANDIDATES = [
    "game_datetime",
    "game_utc_datetime",
    "created_at",
    "start_time_utc",
    "start_datetime",
]

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PKL = PROJECT_ROOT / "output_dataset" / "games_enriched_full_stop_policy_raw.pkl"

OUTPUT_DIR = PROJECT_ROOT / "output" / "potential_rating"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_DEFAULT_JSON = OUTPUT_DIR / "potential_rating_default_summary.json"
OUTPUT_GRID_JSON = OUTPUT_DIR / "potential_rating_grid_search_tilt_warmup.json"


# =====================================================================
# DATA STRUCTURES
# =====================================================================

@dataclass
class SessionSummary:
    session_id: str
    time_control_category: str
    n_games_total: int
    n_games_used: int
    rating_after_last_game: Optional[float]


# =====================================================================
# UTILS
# =====================================================================

def detect_tilt_column(df: pd.DataFrame) -> str:
    """
    Find a tilt/stop-probability column in the dataframe.
    """
    for col in TILT_COL_CANDIDATES:
        if col in df.columns:
            return col
    raise ValueError(
        f"Could not find any tilt column in dataframe. "
        f"Tried: {TILT_COL_CANDIDATES}"
    )


def detect_time_category_column(df: pd.DataFrame) -> Optional[str]:
    """
    Find a column that encodes bullet / blitz / rapid / classical.
    """
    for col in TIME_CATEGORY_COL_CANDIDATES:
        if col in df.columns:
            return col
    return None


def normalize_time_category(raw_value: str) -> str:
    """
    Map raw speed/time_class values to one of:
        bullet, blitz, rapid, classic, other
    """
    if not isinstance(raw_value, str):
        return "other"
    v = raw_value.strip().lower()

    if "bullet" in v:
        return "bullet"
    if "blitz" in v:
        return "blitz"
    if "rapid" in v:
        return "rapid"
    if "classic" in v or "classical" in v:
        return "classic"
    if "daily" in v or "corres" in v:
        return "classic"  # treat correspondence/daily as classic

    return "other"


def detect_datetime_column(df: pd.DataFrame) -> Optional[str]:
    """
    Try to find a datetime-like column to define global game order.
    """
    for col in DATETIME_COL_CANDIDATES:
        if col in df.columns and np.issubdtype(df[col].dtype, np.datetime64):
            return col

    # fallback: any datetime column
    for col in df.columns:
        if np.issubdtype(df[col].dtype, np.datetime64):
            return col

    return None


# =====================================================================
# CORE LOADING & FILTERING
# =====================================================================

def load_games() -> pd.DataFrame:
    """
    Load the games dataset (tilt-enriched).
    """
    if not DATA_PKL.exists():
        raise FileNotFoundError(
            f"{DATA_PKL} not found. Expected dataset at this path.\n"
            f"Make sure you've generated games_enriched_full_stop_policy_raw.pkl first."
        )
    print(f"Loading games dataset from {DATA_PKL} ...")
    df = pd.read_pickle(DATA_PKL)

    required = ["session_id", "game_in_session", "result_score", "my_rating", "opp_rating"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in dataset: {missing}")

    return df


def build_peak_games(
    df: pd.DataFrame,
    tilt_threshold: Optional[float],
    warmup_config: Dict[str, int],
) -> Tuple[pd.DataFrame, List[SessionSummary]]:
    """
    Build the set of "peak" games given:
      - tilt_threshold (None = no tilt filter),
      - warmup_config (dict: category -> #games to drop from left).

    For each session:
      1. Sort by game_in_session.
      2. Tilt filter:
           - If tilt_threshold is not None:
                keep games up to (but not including) first tilt_score > threshold.
           - Else: keep all games.
      3. Infer time-control category.
      4. Warmup filter:
           - Drop the first warmup_config[category] games.
      5. Remaining games are "used" games for this session.

    Returns:
      df_peak: concatenated used games from all sessions.
      session_summaries: list of SessionSummary with per-session stats.
    """
    tilt_col = None
    if tilt_threshold is not None:
        tilt_col = detect_tilt_column(df)

    time_cat_col = detect_time_category_column(df)

    session_ids = sorted(df["session_id"].unique())
    used_list: List[pd.DataFrame] = []
    summaries: List[SessionSummary] = []

    for session_id in session_ids:
        sess_df = df[df["session_id"] == session_id].sort_values("game_in_session").copy()
        n_total = len(sess_df)

        # --- Step 1: tilt filter (right side) ---
        if tilt_col is not None:
            tilt_vals = sess_df[tilt_col].astype(float).values
            above = np.where(tilt_vals > tilt_threshold)[0]
            if len(above) > 0:
                first_tilt_idx = int(above[0])
                pre_tilt_df = sess_df.iloc[:first_tilt_idx].copy()
            else:
                pre_tilt_df = sess_df
        else:
            pre_tilt_df = sess_df

        # --- Step 2: time-control category (for warmup) ---
        if time_cat_col is not None and time_cat_col in pre_tilt_df.columns and not pre_tilt_df.empty:
            cats = pre_tilt_df[time_cat_col].dropna().astype(str).map(normalize_time_category)
            if len(cats) == 0:
                time_cat = "other"
            else:
                time_cat = cats.mode().iloc[0]
        else:
            time_cat = "other"

        warmup_n = warmup_config.get(time_cat, 0)

        # --- Step 3: warmup filter (left side) ---
        if warmup_n > 0:
            used_df = pre_tilt_df.iloc[warmup_n:].copy()
        else:
            used_df = pre_tilt_df.copy()

        n_used = len(used_df)

        if n_used > 0:
            used_list.append(used_df)

        summaries.append(
            SessionSummary(
                session_id=str(session_id),
                time_control_category=time_cat,
                n_games_total=int(n_total),
                n_games_used=int(n_used),
                rating_after_last_game=None,
            )
        )

    if not used_list:
        # No usable games after tilt + warmup
        return pd.DataFrame(), summaries

    df_peak = pd.concat(used_list, axis=0).copy()
    return df_peak, summaries


def order_games_globally(df_used: pd.DataFrame) -> pd.DataFrame:
    """
    Determine a global order for all used games and return a sorted copy.

    Priority:
      1. Use a datetime column if present.
      2. Else sort by (session_id, game_in_session).
    """
    dt_col = detect_datetime_column(df_used)
    if dt_col is not None:
        df_sorted = df_used.sort_values([dt_col, "session_id", "game_in_session"])
    else:
        df_sorted = df_used.sort_values(["session_id", "game_in_session"])

    df_sorted = df_sorted.reset_index(drop=True)
    df_sorted["global_game_idx"] = df_sorted.index
    return df_sorted


# =====================================================================
# GLICKO CORE
# =====================================================================

def run_glicko(df_sorted: pd.DataFrame) -> Tuple[pd.DataFrame, float, float]:
    """
    Run Glicko-2 sequentially over all used games.

    - Initial rating = my_rating of the first used game.
    - RD=350, vol=0.06.
    - For each game:
        player.update_player([opp_rating], [350], [result_score])
      and record rating and RD after that game.
    """
    if df_sorted.empty:
        raise ValueError("df_sorted is empty; nothing to rate.")

    required = ["my_rating", "opp_rating", "result_score"]
    for c in required:
        if c not in df_sorted.columns:
            raise ValueError(f"Missing column {c} in df_sorted")

    first_row = df_sorted.iloc[0]
    initial_rating = float(first_row["my_rating"])

    player = glicko2.Player(rating=initial_rating, rd=350, vol=0.06)

    ratings: List[float] = []
    rds: List[float] = []

    for _, row in df_sorted.iterrows():
        opp_rating = float(row["opp_rating"])
        result = float(row["result_score"])

        player.update_player([opp_rating], [350], [result])

        ratings.append(float(player.rating))
        rds.append(float(player.rd))

    df_out = df_sorted.copy()
    df_out["potential_rating_after_game"] = ratings
    df_out["potential_rd_after_game"] = rds

    final_rating = ratings[-1]
    final_rd = rds[-1]
    return df_out, final_rating, final_rd


def attach_session_end_ratings(
    df_rated: pd.DataFrame,
    summaries: List[SessionSummary],
) -> List[SessionSummary]:
    """
    For each session with used games, record the rating after its last used game.
    """
    last_rating_by_session: Dict[str, float] = {}

    grouped = df_rated.groupby("session_id")
    for session_id, g in grouped:
        last_row = g.sort_values("game_in_session").iloc[-1]
        last_rating_by_session[str(session_id)] = float(
            last_row["potential_rating_after_game"]
        )

    updated: List[SessionSummary] = []
    for s in summaries:
        if s.n_games_used > 0 and s.session_id in last_rating_by_session:
            s.rating_after_last_game = last_rating_by_session[s.session_id]
        updated.append(s)

    return updated


def compute_potential_rating(
    df: pd.DataFrame,
    tilt_threshold: Optional[float],
    warmup_config: Dict[str, int],
) -> Tuple[Optional[float], Optional[float], int, int, Optional[pd.DataFrame], List[SessionSummary]]:
    """
    High-level API: compute potential rating given tilt + warmup.

    Args:
      df: full per-game dataframe.
      tilt_threshold: float or None (no tilt filter).
      warmup_config: dict category -> #games to drop from left.

    Returns:
      final_rating (float or None),
      final_rd (float or None),
      n_sessions_with_games (int),
      n_games_used (int),
      df_rated (per-game ratings DataFrame or None),
      session_summaries (List[SessionSummary])
    """
    df_peak, summaries = build_peak_games(df, tilt_threshold, warmup_config)
    if df_peak.empty:
        return None, None, 0, 0, None, summaries

    df_sorted = order_games_globally(df_peak)
    n_sessions_with_games = int(df_sorted["session_id"].nunique())
    n_games_used = int(len(df_sorted))

    df_rated, final_rating, final_rd = run_glicko(df_sorted)
    summaries = attach_session_end_ratings(df_rated, summaries)

    return (
        float(final_rating),
        float(final_rd),
        n_sessions_with_games,
        n_games_used,
        df_rated,
        summaries,
    )


# =====================================================================
# OPTIMIZER: tilt + warmup grid search
# =====================================================================

def grid_search_tilt_and_warmup(
    df: pd.DataFrame,
    tilt_thresholds: List[float],
    bullet_candidates: List[int],
    blitz_candidates: List[int],
    rapid_candidates: List[int],
    classic_candidates: List[int],
) -> Tuple[Optional[Dict], List[Dict]]:
    """
    Grid-search over (tilt_threshold, warmup_config) combinations.

    For each combination:
      - compute potential rating using compute_potential_rating(),
      - track the best combination by final_rating.

    Returns:
      best_result: dict with keys
          {
            "tilt_threshold": ...,
            "warmup_config": {...},
            "final_potential_rating": ...,
            "final_rating_rd": ...,
            "n_sessions_with_games": ...,
            "n_games_used": ...
          }
        or None if nothing worked.

      all_results: list of dicts with the same shape.
    """
    all_results: List[Dict] = []
    best_result: Optional[Dict] = None

    for thr in tilt_thresholds:
        for b in bullet_candidates:
            for bl in blitz_candidates:
                for r in rapid_candidates:
                    for c in classic_candidates:
                        warmup_config = {
                            "bullet": b,
                            "blitz": bl,
                            "rapid": r,
                            "classic": c,
                        }

                        final_rating, final_rd, n_sessions, n_games, _, _ = compute_potential_rating(
                            df=df,
                            tilt_threshold=thr,
                            warmup_config=warmup_config,
                        )

                        result = {
                            "tilt_threshold": float(thr),
                            "warmup_config": warmup_config,
                            "final_potential_rating": final_rating,
                            "final_rating_rd": final_rd,
                            "n_sessions_with_games": int(n_sessions),
                            "n_games_used": int(n_games),
                        }
                        all_results.append(result)

                        print(
                            f"tilt={thr:.2f}, warmup={warmup_config} -> "
                            f"final_potential_rating={final_rating}, "
                            f"n_sessions={n_sessions}, "
                            f"n_games={n_games}"
                        )

                        if final_rating is not None:
                            if best_result is None or final_rating > best_result["final_potential_rating"]:
                                best_result = result

    return best_result, all_results


# =====================================================================
# JSON HELPERS
# =====================================================================

def session_summaries_to_dict_list(summaries: List[SessionSummary]) -> List[Dict]:
    out: List[Dict] = []
    for s in summaries:
        out.append(
            {
                "session_id": s.session_id,
                "time_control_category": s.time_control_category,
                "n_games_total": s.n_games_total,
                "n_games_used": s.n_games_used,
                "rating_after_last_game": s.rating_after_last_game,
            }
        )
    return out


def save_default_summary(
    tilt_threshold: Optional[float],
    warmup_config: Dict[str, int],
    final_rating: Optional[float],
    final_rd: Optional[float],
    n_sessions: int,
    n_games: int,
    df_rated: Optional[pd.DataFrame],
    summaries: List[SessionSummary],
    path: Path = OUTPUT_DEFAULT_JSON,
) -> None:
    """
    Save a summary JSON for a single (tilt, warmup) config.
    """
    if df_rated is not None and not df_rated.empty:
        last_row = df_rated.iloc[-1]
        last_game_info = {
            "session_id": str(last_row["session_id"]),
            "game_in_session": int(last_row["game_in_session"]),
        }
    else:
        last_game_info = None

    summary = {
        "username": USERNAME,
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "tilt_threshold": tilt_threshold,
        "warmup_config": warmup_config,
        "n_sessions_with_games": int(n_sessions),
        "n_games_used": int(n_games),
        "final_potential_rating": final_rating,
        "final_rating_rd": final_rd,
        "final_game_info": last_game_info,
        "sessions": session_summaries_to_dict_list(summaries),
    }

    with path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved default potential rating summary to: {path}")


def save_grid_search_summary(
    tilt_thresholds: List[float],
    bullet_candidates: List[int],
    blitz_candidates: List[int],
    rapid_candidates: List[int],
    classic_candidates: List[int],
    best_result: Optional[Dict],
    all_results: List[Dict],
    path: Path = OUTPUT_GRID_JSON,
) -> None:
    """
    Save JSON for the grid search over tilt + warmup.
    """
    summary: Dict[str, object] = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "search_space": {
            "tilt_thresholds": [float(t) for t in tilt_thresholds],
            "bullet_candidates": bullet_candidates,
            "blitz_candidates": blitz_candidates,
            "rapid_candidates": rapid_candidates,
            "classic_candidates": classic_candidates,
        },
        "results": all_results,
        "best": best_result,
    }

    with path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved grid search summary to: {path}")


# =====================================================================
# MAIN (when run as a script)
# =====================================================================

def main():
    # 1) Load data
    df = load_games()

    # 2) Compute potential rating for default config
    print("\n=== Computing default potential rating ===")
    default_final_rating, default_final_rd, default_n_sessions, default_n_games, df_rated, summaries = (
        compute_potential_rating(
            df=df,
            tilt_threshold=DEFAULT_TILT_THRESHOLD,
            warmup_config=DEFAULT_WARMUP_CONFIG,
        )
    )

    print(
        f"Default config -> tilt={DEFAULT_TILT_THRESHOLD}, "
        f"warmup={DEFAULT_WARMUP_CONFIG}, "
        f"final_potential_rating={default_final_rating}, "
        f"RD={default_final_rd}, "
        f"sessions={default_n_sessions}, games={default_n_games}"
    )

    save_default_summary(
        tilt_threshold=DEFAULT_TILT_THRESHOLD,
        warmup_config=DEFAULT_WARMUP_CONFIG,
        final_rating=default_final_rating,
        final_rd=default_final_rd,
        n_sessions=default_n_sessions,
        n_games=default_n_games,
        df_rated=df_rated,
        summaries=summaries,
    )

    # 3) Grid search over tilt thresholds & warmup configs
    print("\n=== Grid search over tilt threshold + warmup ===")

    # You can tweak these ranges however you like.
    tilt_thresholds = [round(x, 2) for x in np.linspace(0.5, 0.95, 10)]
    bullet_candidates = [0, 1, 2, 3, 4]
    blitz_candidates = [0, 1, 2, 3]
    rapid_candidates = [0, 1, 2]
    classic_candidates = [0, 1]

    best_result, all_results = grid_search_tilt_and_warmup(
        df=df,
        tilt_thresholds=tilt_thresholds,
        bullet_candidates=bullet_candidates,
        blitz_candidates=blitz_candidates,
        rapid_candidates=rapid_candidates,
        classic_candidates=classic_candidates,
    )

    if best_result is None:
        print("\nNo combination produced a valid potential rating.")
    else:
        print(
            "\n=== Best combination ===\n"
            f"tilt_threshold={best_result['tilt_threshold']}, "
            f"warmup={best_result['warmup_config']}, "
            f"final_potential_rating={best_result['final_potential_rating']}, "
            f"RD={best_result['final_rating_rd']}, "
            f"sessions={best_result['n_sessions_with_games']}, "
            f"games={best_result['n_games_used']}"
        )

    save_grid_search_summary(
        tilt_thresholds=tilt_thresholds,
        bullet_candidates=bullet_candidates,
        blitz_candidates=blitz_candidates,
        rapid_candidates=rapid_candidates,
        classic_candidates=classic_candidates,
        best_result=best_result,
        all_results=all_results,
    )


if __name__ == "__main__":
    main()
