#!/usr/bin/env python
"""
Quickly enrich games_enriched_full with extra warmup-related features
derived from the raw Lichess JSON export.

Input:
  - data/raw_data/julio_amigo_dos_games_full.json
  - data/formatted_data/games_enriched_full.pkl

Output:
  - data/formatted_data/games_enriched_full_warmup.pkl
  - data/formatted_data/games_enriched_full_warmup.csv

New columns added per game (when info is available in raw JSON):

  num_blunders
  num_mistakes
  num_inaccuracies
  acpl
  avg_move_time                # sec per move, your moves only
  early_blunders_20plies       # in first 10 moves (20 plies)
  early_mistakes_20plies
  early_inaccuracies_20plies
  max_eval_swing               # max |eval_i - eval_{i-1}|
  avg_eval_swing               # mean |eval_i - eval_{i-1}|
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------------------

USERNAME = "julio_amigo_dos"  # your lichess username (case-insensitive)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_JSON_PATH = PROJECT_ROOT / "data" / "raw_data" / "julio_amigo_dos_games_full.json"
ENRICHED_PKL_PATH = PROJECT_ROOT / "data" / "formatted_data" / "games_enriched_full.pkl"

OUTPUT_PKL_PATH = PROJECT_ROOT / "data" / "formatted_data" / "games_enriched_full_warmup.pkl"
OUTPUT_CSV_PATH = PROJECT_ROOT / "data" / "formatted_data" / "games_enriched_full_warmup.csv"


# -----------------------------------------------------------------------------
# RAW LOADING HELPERS
# -----------------------------------------------------------------------------

def load_raw_games(path: Path) -> List[Dict[str, Any]]:
    """
    Loads the raw lichess games JSON.
    Expects a JSON list. If that fails, you can adjust to NDJSON as needed.
    """
    if not path.exists():
        raise FileNotFoundError(f"Raw games file not found: {path}")

    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                return data
            elif isinstance(data, dict) and "games" in data:
                return data["games"]
            else:
                raise ValueError(f"Unexpected JSON structure in {path}")
    except json.JSONDecodeError as e:
        raise RuntimeError(
            f"Failed to parse {path} as a JSON array. "
            f"If it's NDJSON, you'll need to adapt this function to read line by line.\n"
            f"Original error: {e}"
        )


def get_my_color(game: Dict[str, Any], username: str) -> Optional[str]:
    """
    Determine whether you're white or black in this game.
    Returns 'white', 'black', or None if we cannot determine.
    """
    username_lower = username.lower()

    players = game.get("players", {})
    for color in ("white", "black"):
        p = players.get(color, {})
        user_info = p.get("user") or {}
        # lichess exports may have name or username
        name = (user_info.get("name") or user_info.get("username") or "").lower()
        if name == username_lower:
            return color

    return None


# -----------------------------------------------------------------------------
# FEATURE COMPUTATION
# -----------------------------------------------------------------------------

def compute_game_features(game: Dict[str, Any], username: str) -> Dict[str, Any]:
    """
    Compute warmup-related features for a single game using the raw lichess JSON.
    Robust to missing fields; returns None/np.nan when info is missing.
    """

    game_id = game.get("id")

    # Default missing values
    out = {
        "game_id": game_id,
        "num_blunders": np.nan,
        "num_mistakes": np.nan,
        "num_inaccuracies": np.nan,
        "acpl": np.nan,
        "avg_move_time": np.nan,
        "early_blunders_20plies": np.nan,
        "early_mistakes_20plies": np.nan,
        "early_inaccuracies_20plies": np.nan,
        "max_eval_swing": np.nan,
        "avg_eval_swing": np.nan,
    }

    if not game_id:
        return out

    my_color = get_my_color(game, username)
    if my_color is None:
        # Could be an anonymous opponent or something odd
        # We still might compute some aggregate features that don't need color
        my_color = "white"  # dummy, only used when safe

    players = game.get("players", {})
    my_player = players.get(my_color, {})

    # --- Game-level engine summary: blunders/mistakes/inaccuracies, ACPL ---
    analysis_summary = my_player.get("analysis", {}) or {}
    out["num_blunders"] = analysis_summary.get("blunder", np.nan)
    out["num_mistakes"] = analysis_summary.get("mistake", np.nan)
    out["num_inaccuracies"] = analysis_summary.get("inaccuracy", np.nan)
    out["acpl"] = analysis_summary.get("acpl", np.nan)

    # --- Clocks: average time per move (your moves only) ---
    clocks = game.get("clocks")
    if isinstance(clocks, list) and len(clocks) >= 2:
        # clocks[i] is time left (e.g. centiseconds) after ply i
        # so time spent on ply i is clocks[i-1] - clocks[i]
        times_spent = []
        for i in range(1, len(clocks)):
            prev_t = clocks[i - 1]
            cur_t = clocks[i]
            if isinstance(prev_t, (int, float)) and isinstance(cur_t, (int, float)):
                times_spent.append(prev_t - cur_t)

        # Separate "my" plies:
        # white moves are plies 0,2,4,...; black moves 1,3,5,...
        start_index = 0 if my_color == "white" else 1
        my_spent = times_spent[start_index::2]

        if my_spent:
            # convert centiseconds -> seconds (if clocks are in centiseconds)
            out["avg_move_time"] = float(np.mean(my_spent) / 100.0)

    # --- Per-move engine analysis: early blunders/mistakes, eval swings ---
    move_analysis = game.get("analysis", [])
    evals = []
    for a in move_analysis:
        ev = a.get("eval")
        if isinstance(ev, (int, float)):
            evals.append(ev)

    if len(evals) >= 2:
        swings = [abs(evals[i] - evals[i - 1]) for i in range(1, len(evals))]
        out["max_eval_swing"] = float(np.max(swings))
        out["avg_eval_swing"] = float(np.mean(swings))

    # Early judgments in first 20 plies (10 moves each side)
    early_plies = move_analysis[:20]
    if early_plies:
        b = m = i = 0
        for a in early_plies:
            j = a.get("judgment") or {}
            name = j.get("name", "")
            if name == "Blunder":
                b += 1
            elif name == "Mistake":
                m += 1
            elif name == "Inaccuracy":
                i += 1
        out["early_blunders_20plies"] = float(b)
        out["early_mistakes_20plies"] = float(m)
        out["early_inaccuracies_20plies"] = float(i)

    return out


def build_features_from_raw_games(
    raw_games: List[Dict[str, Any]],
    username: str,
) -> pd.DataFrame:
    """
    Compute all warmup features for all games in the raw JSON.
    Returns a DataFrame with columns:
      - game_id
      - num_blunders
      - num_mistakes
      - num_inaccuracies
      - acpl
      - avg_move_time
      - early_blunders_20plies
      - early_mistakes_20plies
      - early_inaccuracies_20plies
      - max_eval_swing
      - avg_eval_swing
    """
    rows = []
    for g in raw_games:
        rows.append(compute_game_features(g, username=username))

    df_features = pd.DataFrame(rows)
    # Drop games without id just in case
    df_features = df_features.dropna(subset=["game_id"])
    df_features["game_id"] = df_features["game_id"].astype(str)
    return df_features


# -----------------------------------------------------------------------------
# MERGE WITH ENRICHED DF
# -----------------------------------------------------------------------------

def find_game_id_column(df: pd.DataFrame) -> str:
    """
    Try to find which column identifies the game (to merge on).
    Typical candidates: 'game_id', 'id'.
    """
    if "game_id" in df.columns:
        return "game_id"
    if "id" in df.columns:
        return "id"
    raise ValueError(
        "Cannot find a game id column to merge on. "
        "Expected 'game_id' or 'id' in games_enriched_full."
    )


def main():
    # --- 1) Load raw games ---
    print(f"Loading raw games from {RAW_JSON_PATH} ...")
    raw_games = load_raw_games(RAW_JSON_PATH)
    print(f"Loaded {len(raw_games)} raw games.")

    # --- 2) Compute features from raw games ---
    print("Computing warmup-related features from raw games...")
    df_features = build_features_from_raw_games(raw_games, username=USERNAME)
    print(f"Built feature rows for {len(df_features)} games.")

    # --- 3) Load enriched games DF ---
    if not ENRICHED_PKL_PATH.exists():
        raise FileNotFoundError(f"Enriched games file not found: {ENRICHED_PKL_PATH}")

    print(f"Loading enriched games DF from {ENRICHED_PKL_PATH} ...")
    df_enriched = pd.read_pickle(ENRICHED_PKL_PATH)
    print(f"Enriched DF shape: {df_enriched.shape}")

    key_col = find_game_id_column(df_enriched)
    print(f"Merging on key column: {key_col}")

    # Make sure the key col is string for safe merge
    df_enriched[key_col] = df_enriched[key_col].astype(str)

    # Rename df_features key to match
    df_features_renamed = df_features.rename(columns={"game_id": key_col})

    # --- 4) Merge ---
    df_merged = df_enriched.merge(df_features_renamed, on=key_col, how="left")
    print(f"Merged DF shape: {df_merged.shape}")

    # --- 5) Save outputs ---
    df_merged.to_pickle(OUTPUT_PKL_PATH)
    df_merged.to_csv(OUTPUT_CSV_PATH, index=False)

    print("Saved enriched games with warmup features to:")
    print(f"  {OUTPUT_PKL_PATH}")
    print(f"  {OUTPUT_CSV_PATH}")


if __name__ == "__main__":
    main()
