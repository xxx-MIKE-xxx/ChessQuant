"""
Warmup detector (stub version, no ML).

This module exposes a simple API for recommending warmup length based on a
pre-computed best configuration found via the potential_rating grid search.

It does NOT train any model. Instead it:

  - Reads the best (tilt_threshold, warmup_config) combination from:
        output/potential_rating/potential_rating_grid_search_tilt_warmup.json

  - Uses the `warmup_config` portion as a fixed per-time-control warmup rule:
        warmup_config = {
            "bullet": <int>,
            "blitz": <int>,
            "rapid": <int>,
            "classic": <int>,
        }

  - Provides helper functions so other parts of the system can query:
        - load_best_warmup_config_from_potential_rating()
        - recommend_warmup_for_speed(speed_str, config)

When run as a script, it:

  - Loads the best warmup configuration from the grid-search JSON.
  - Builds a compact summary describing:
        * warmup_config per time-control
        * the tilt threshold associated with that best config
        * the final potential rating it achieved
        * how many sessions/games were used
        * the search space for warmup & tilt

  - Saves this summary JSON to:
        output/warmup_detector/warmup_detector_summary.json
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional


# =====================================================================
# CONFIG & PATHS
# =====================================================================

USERNAME = "julio_amigo_dos"

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Where potential_rating grid search writes its result
POTENTIAL_RATING_GRID_JSON = (
    PROJECT_ROOT
    / "output"
    / "potential_rating"
    / "potential_rating_grid_search_tilt_warmup.json"
)

# Our own outputs
OUTPUT_DIR = PROJECT_ROOT / "output" / "warmup_detector"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_SUMMARY_JSON = OUTPUT_DIR / "warmup_detector_summary.json"

# Fallback warmup config in case we can't read the potential_rating JSON
DEFAULT_WARMUP_CONFIG: Dict[str, int] = {
    "bullet": 3,
    "blitz": 2,
    "rapid": 1,
    "classic": 0,
}


# =====================================================================
# DATA STRUCTURES
# =====================================================================

@dataclass
class WarmupDetectorConfig:
    warmup_config: Dict[str, int]
    tilt_threshold: Optional[float] = None
    source_json: Optional[str] = None
    final_potential_rating: Optional[float] = None
    final_rating_rd: Optional[float] = None
    n_sessions_with_games: Optional[int] = None
    n_games_used: Optional[int] = None


# =====================================================================
# UTILS
# =====================================================================

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
        # treat correspondence / daily as classic for warmup purposes
        return "classic"

    return "other"


# =====================================================================
# CORE API
# =====================================================================

def load_best_warmup_config_from_potential_rating(
    grid_json_path: Path = POTENTIAL_RATING_GRID_JSON,
) -> WarmupDetectorConfig:
    """
    Load the best warmup configuration from the potential_rating grid search JSON.

    Expected structure (written by potential_rating.py):

      {
        "generated_at": "...",
        "search_space": {...},
        "results": [...],
        "best": {
          "tilt_threshold": <float>,
          "warmup_config": {
            "bullet": <int>,
            "blitz": <int>,
            "rapid": <int>,
            "classic": <int>
          },
          "final_potential_rating": <float> or null,
          "final_rating_rd": <float> or null,
          "n_sessions_with_games": <int>,
          "n_games_used": <int>
        }
      }

    If the file or 'best' entry is missing, we fallback to DEFAULT_WARMUP_CONFIG.
    """
    if not grid_json_path.exists():
        print(
            f"[warmup_detector] Potential rating grid search JSON not found at "
            f"{grid_json_path}. Using DEFAULT_WARMUP_CONFIG."
        )
        return WarmupDetectorConfig(
            warmup_config=DEFAULT_WARMUP_CONFIG.copy(),
            tilt_threshold=None,
            source_json=None,
        )

    try:
        with grid_json_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(
            f"[warmup_detector] Failed to read {grid_json_path}: {e}. "
            f"Using DEFAULT_WARMUP_CONFIG."
        )
        return WarmupDetectorConfig(
            warmup_config=DEFAULT_WARMUP_CONFIG.copy(),
            tilt_threshold=None,
            source_json=str(grid_json_path),
        )

    best = data.get("best")
    if not isinstance(best, dict):
        print(
            f"[warmup_detector] 'best' field missing or invalid in {grid_json_path}. "
            f"Using DEFAULT_WARMUP_CONFIG."
        )
        return WarmupDetectorConfig(
            warmup_config=DEFAULT_WARMUP_CONFIG.copy(),
            tilt_threshold=None,
            source_json=str(grid_json_path),
        )

    warmup_config = best.get("warmup_config")
    tilt_threshold = best.get("tilt_threshold")

    if not isinstance(warmup_config, dict):
        print(
            f"[warmup_detector] 'warmup_config' missing in best result of "
            f"{grid_json_path}. Using DEFAULT_WARMUP_CONFIG."
        )
        warmup_config = DEFAULT_WARMUP_CONFIG.copy()

    # Ensure we have all required keys and cast to int
    merged_config = DEFAULT_WARMUP_CONFIG.copy()
    for k, v in warmup_config.items():
        try:
            merged_config[k] = int(v)
        except Exception:
            # Ignore malformed entries, keep defaults
            pass

    final_potential_rating = best.get("final_potential_rating")
    final_rating_rd = best.get("final_rating_rd")
    n_sessions_with_games = best.get("n_sessions_with_games")
    n_games_used = best.get("n_games_used")

    return WarmupDetectorConfig(
        warmup_config=merged_config,
        tilt_threshold=float(tilt_threshold) if tilt_threshold is not None else None,
        source_json=str(grid_json_path),
        final_potential_rating=final_potential_rating,
        final_rating_rd=final_rating_rd,
        n_sessions_with_games=n_sessions_with_games,
        n_games_used=n_games_used,
    )


def recommend_warmup_for_speed(
    speed_value: str,
    config: WarmupDetectorConfig,
) -> int:
    """
    Given a raw speed/time_class value and the global warmup_config,
    return the recommended number of warmup games.

    Example raw values: "bullet", "blitz", "rapid", "classical", "daily", etc.
    """
    cat = normalize_time_category(speed_value)
    return int(config.warmup_config.get(cat, 0))


# =====================================================================
# MAIN (script entrypoint)
# =====================================================================

def main():
    # 1) Load warmup config (from potential_rating grid search, or fallback)
    cfg = load_best_warmup_config_from_potential_rating()

    print("\n=== Warmup detector (stub, global config only) ===")
    print(f"Username: {USERNAME}")
    print(f"Source grid JSON: {cfg.source_json}")
    print(f"Tilt threshold (from potential rating): {cfg.tilt_threshold}")
    print(f"Warmup config: {cfg.warmup_config}")
    print(f"Final potential rating (best combo): {cfg.final_potential_rating}")
    print(f"Final rating RD (best combo): {cfg.final_rating_rd}")
    print(f"Sessions used (best combo): {cfg.n_sessions_with_games}")
    print(f"Games used (best combo): {cfg.n_games_used}")

    # 2) Try to also read the search space from the grid JSON, if present
    search_space = None
    if cfg.source_json is not None:
        try:
            with Path(cfg.source_json).open("r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict) and "search_space" in data:
                search_space = data["search_space"]
        except Exception as e:
            print(f"[warmup_detector] Could not read search_space from grid JSON: {e}")

    # 3) Build summary JSON (no per-session breakdown, just global info)
    summary = {
        "username": USERNAME,
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "source_grid_json": cfg.source_json,
        "tilt_threshold_from_potential_rating": cfg.tilt_threshold,
        "warmup_config": cfg.warmup_config,
        "final_potential_rating_best_combo": cfg.final_potential_rating,
        "final_rating_rd_best_combo": cfg.final_rating_rd,
        "n_sessions_with_games_best_combo": cfg.n_sessions_with_games,
        "n_games_used_best_combo": cfg.n_games_used,
        "search_space": search_space,
    }

    with OUTPUT_SUMMARY_JSON.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved warmup detector summary to: {OUTPUT_SUMMARY_JSON}")
    print("Done.")


if __name__ == "__main__":
    main()
