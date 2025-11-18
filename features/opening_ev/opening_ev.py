# features/opening_ev/opening_ev.py

"""
Opening EV analysis based on engine evaluations.

Goal
-----
Instead of judging openings by final game result, we estimate the
expected value (EV) you get *out of the opening / early midgame*
using engine evaluation in centipawns.

High-level steps
----------------
1) Load per-game dataset (with engine evaluation):
   - Expected path:
       data/formatted_data/games_enriched_full.pkl

   Required columns:
     - opening_eco, opening_name (or similar)   # opening identity
     - my_color                                 # "white" or "black"
     - engine eval column in centipawns (from White's POV)
     - optionally result_score (0, 0.5, 1) for naive winrate

2) For each game:
   - Determine opening key:
       "{opening_eco} | {opening_name}"
       (with fallbacks for missing fields)
   - Take engine evaluation in centipawns, from White's POV.
   - Convert to "my POV":
       if my_color == "black": eval_cp_pov = -eval_cp
       else:                    eval_cp_pov = eval_cp
   - Optionally clip eval to [-MAX_ABS_EVAL, +MAX_ABS_EVAL].
   - Store as `opening_ev_cp_pov`.

3) Aggregate per opening:
   - n_games
   - mean_eval_cp_pov
   - median_eval_cp_pov
   - std_eval_cp_pov
   - share_eval_gt0     (fraction with eval_cp_pov > 0)
   - share_eval_ge50    (fraction with eval_cp_pov >= 50)
   - mean_result_score  (optional, if result_score exists)

4) Save:
   - Game-level dataset with `opening_ev_cp_pov`:
       output_dataset/games_enriched_full_opening_ev.csv
       output_dataset/games_enriched_full_opening_ev.pkl

   - Opening-level stats:
       output/opening_ev/opening_ev_stats.csv
       output/opening_ev/opening_ev_stats.json

Notes
-----
- We rely on Lichess' own opening classification exported in the JSON:
  opening_eco / opening_name. Under the hood this is backed by the
  public `lichess-org/chess-openings` dataset.
- Engine evaluation is assumed to be *from White's perspective* and
  roughly in the "exiting the opening / early midgame" region. If you
  later add more detailed per-move evals, you can adjust the script
  to compute the EV specifically at ply = opening_ply + k.

"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# --------------------------------------------------------------------
# CONFIG
# --------------------------------------------------------------------

USERNAME = "julio_amigo_dos"

# Where the formatted (per-game) dataset lives
PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_PKL = PROJECT_ROOT / "data" / "formatted_data" / "games_enriched_full.pkl"

# Where to save outputs
OUTPUT_DATASET_DIR = PROJECT_ROOT / "output_dataset"
OUTPUT_DATASET_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_DIR = PROJECT_ROOT / "output" / "opening_ev"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OPENING_EV_CSV = OUTPUT_DIR / "opening_ev_stats.csv"
OPENING_EV_JSON = OUTPUT_DIR / "opening_ev_stats.json"

GAMES_OUT_CSV = OUTPUT_DATASET_DIR / "games_enriched_full_opening_ev.csv"
GAMES_OUT_PKL = OUTPUT_DATASET_DIR / "games_enriched_full_opening_ev.pkl"

# Candidate engine-eval columns (centipawns, from White's POV)
ENGINE_EVAL_COL_CANDIDATES = [
    "eval_cp",           # generic
    "engine_eval_cp",
    "cp_eval",
    "opening_eval_cp",
    "eval_after_opening_cp",
]

# Candidate opening columns
OPENING_ECO_COLS = ["opening_eco", "eco"]
OPENING_NAME_COLS = ["opening_name", "openingName"]

# Minimum games required to report opening stats
MIN_GAMES_PER_OPENING = 5

# Clip eval to avoid mates / huge swings dominating stats
MAX_ABS_EVAL_CP = 1000.0


# --------------------------------------------------------------------
# DATA STRUCTURES
# --------------------------------------------------------------------

@dataclass
class OpeningStats:
    opening_key: str
    opening_eco: Optional[str]
    opening_name: Optional[str]
    n_games: int
    mean_eval_cp_pov: float
    median_eval_cp_pov: float
    std_eval_cp_pov: float
    share_eval_gt0: float
    share_eval_ge50: float
    mean_result_score: Optional[float]


# --------------------------------------------------------------------
# HELPERS
# --------------------------------------------------------------------

def load_games() -> pd.DataFrame:
    if not INPUT_PKL.exists():
        raise FileNotFoundError(
            f"{INPUT_PKL} not found. Run your raw_to_panda_frame / "
            f"data formatting pipeline first."
        )

    print(f"Loading games dataset from {INPUT_PKL} ...")
    df = pd.read_pickle(INPUT_PKL)

    required = ["my_color"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in dataset: {missing}")

    return df


def detect_engine_eval_column(df: pd.DataFrame) -> str:
    for col in ENGINE_EVAL_COL_CANDIDATES:
        if col in df.columns:
            return col
    # If you know the exact name, set it up top instead of using candidates.
    raise ValueError(
        "Could not find an engine-eval column in the dataframe. "
        f"Tried: {ENGINE_EVAL_COL_CANDIDATES}"
    )


def get_opening_eco_and_name(row: pd.Series) -> Tuple[Optional[str], Optional[str]]:
    eco = None
    name = None

    for c in OPENING_ECO_COLS:
        if c in row and pd.notna(row[c]):
            eco = str(row[c])
            break

    for c in OPENING_NAME_COLS:
        if c in row and pd.notna(row[c]):
            name = str(row[c])
            break

    return eco, name


def build_opening_key(eco: Optional[str], name: Optional[str]) -> str:
    if eco and name:
        return f"{eco} | {name}"
    if eco:
        return f"{eco} | <unknown name>"
    if name:
        return f"<no ECO> | {name}"
    return "<unknown opening>"


def add_opening_ev_to_games(df: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
    """
    Add `opening_key` and `opening_ev_cp_pov` columns to the per-game dataframe.

    Returns:
      df_out, eval_col_name
    """
    eval_col = detect_engine_eval_column(df)

    df = df.copy()

    # Build opening_eco, opening_name, opening_key
    ecos: List[Optional[str]] = []
    names: List[Optional[str]] = []
    keys: List[str] = []

    for _, row in df.iterrows():
        eco, name = get_opening_eco_and_name(row)
        ecos.append(eco)
        names.append(name)
        keys.append(build_opening_key(eco, name))

    df["opening_eco_resolved"] = ecos
    df["opening_name_resolved"] = names
    df["opening_key"] = keys

    # POV eval (flip sign if playing black)
    if "my_color" not in df.columns:
        raise ValueError("Dataset must contain 'my_color' column (white/black).")

    eval_vals = df[eval_col].astype(float)
    my_color = df["my_color"].astype(str).str.lower()

    eval_pov = eval_vals.copy()
    # Positive cp means better for White; flip for Black
    eval_pov[my_color == "black"] = -eval_pov[my_color == "black"]

    # Clip extreme values
    eval_pov = eval_pov.clip(lower=-MAX_ABS_EVAL_CP, upper=MAX_ABS_EVAL_CP)

    df["opening_ev_cp_pov"] = eval_pov

    return df, eval_col


def aggregate_opening_stats(df: pd.DataFrame) -> List[OpeningStats]:
    """
    Aggregate EV per opening_key (after adding opening_ev_cp_pov).
    """
    if "opening_key" not in df.columns or "opening_ev_cp_pov" not in df.columns:
        raise ValueError(
            "Dataframe must contain 'opening_key' and 'opening_ev_cp_pov' columns."
        )

    # Filter out games without eval
    df_use = df[~df["opening_ev_cp_pov"].isna()].copy()
    if df_use.empty:
        return []

    has_result = "result_score" in df_use.columns

    stats_list: List[OpeningStats] = []

    grouped = df_use.groupby("opening_key", dropna=False)

    for key, g in grouped:
        n_games = len(g)
        if n_games < MIN_GAMES_PER_OPENING:
            continue

        vals = g["opening_ev_cp_pov"].astype(float)
        mean_ev = float(vals.mean())
        median_ev = float(vals.median())
        std_ev = float(vals.std(ddof=0)) if n_games > 1 else 0.0

        share_gt0 = float((vals > 0.0).mean())
        share_ge50 = float((vals >= 50.0).mean())

        if has_result:
            mean_result = float(g["result_score"].astype(float).mean())
        else:
            mean_result = None

        # Recover one representative ECO/name
        eco = g["opening_eco_resolved"].dropna().astype(str)
        name = g["opening_name_resolved"].dropna().astype(str)
        eco_rep = eco.iloc[0] if len(eco) > 0 else None
        name_rep = name.iloc[0] if len(name) > 0 else None

        stats_list.append(
            OpeningStats(
                opening_key=str(key),
                opening_eco=eco_rep,
                opening_name=name_rep,
                n_games=n_games,
                mean_eval_cp_pov=mean_ev,
                median_eval_cp_pov=median_ev,
                std_eval_cp_pov=std_ev,
                share_eval_gt0=share_gt0,
                share_eval_ge50=share_ge50,
                mean_result_score=mean_result,
            )
        )

    # Sort by mean_eval_cp_pov descending
    stats_list.sort(key=lambda s: s.mean_eval_cp_pov, reverse=True)
    return stats_list


def opening_stats_to_dataframe(stats_list: List[OpeningStats]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for s in stats_list:
        rows.append(
            {
                "opening_key": s.opening_key,
                "opening_eco": s.opening_eco,
                "opening_name": s.opening_name,
                "n_games": s.n_games,
                "mean_eval_cp_pov": s.mean_eval_cp_pov,
                "median_eval_cp_pov": s.median_eval_cp_pov,
                "std_eval_cp_pov": s.std_eval_cp_pov,
                "share_eval_gt0": s.share_eval_gt0,
                "share_eval_ge50": s.share_eval_ge50,
                "mean_result_score": s.mean_result_score,
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------

def main() -> None:
    # 1) Load games
    df = load_games()

    # 2) Add opening EV per game
    df_with_ev, eval_col = add_opening_ev_to_games(df)

    print(
        f"Using engine eval column '{eval_col}' (White POV), "
        "converted to 'opening_ev_cp_pov' from your POV."
    )

    # 3) Aggregate per opening
    stats_list = aggregate_opening_stats(df_with_ev)
    df_openings = opening_stats_to_dataframe(stats_list)

    # 4) Save game-level dataset
    print(f"Saving games with opening EV -> {GAMES_OUT_CSV} / {GAMES_OUT_PKL}")
    df_with_ev.to_csv(GAMES_OUT_CSV, index=False)
    df_with_ev.to_pickle(GAMES_OUT_PKL)

    # 5) Save opening-level stats
    print(f"Saving opening EV stats -> {OPENING_EV_CSV} / {OPENING_EV_JSON}")
    df_openings.to_csv(OPENING_EV_CSV, index=False)

    stats_json = {
        "username": USERNAME,
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "engine_eval_column": eval_col,
        "max_abs_eval_cp": MAX_ABS_EVAL_CP,
        "min_games_per_opening": MIN_GAMES_PER_OPENING,
        "n_openings": int(len(df_openings)),
        "openings": df_openings.to_dict(orient="records"),
    }
    with OPENING_EV_JSON.open("w", encoding="utf-8") as f:
        json.dump(stats_json, f, indent=2)

    print("Done.")


if __name__ == "__main__":
    main()
