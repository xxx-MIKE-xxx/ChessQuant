"""
Transform raw Lichess JSON exports into tidy pandas DataFrames.

Outputs (in OUTPUT_DIR):
- profile.csv
- rating_history.csv
- games_enriched_full.csv
- games_enriched_full.pkl

This version:
- Uses the full Julio Amigo Dos games JSON (julio_amigo_dos_games_full.json).
- Extracts as much useful metadata as is reasonably flat.
- Properly reconstructs `time_control` (e.g. "3+0", "5+3") from the `clock` object.
- Adds duration_seconds / duration_minutes from createdAt / lastMoveAt.
- Adds engine- and time-based features for warmup/tilt models:
    - my/opp inaccuracy/mistake/blunder counts and ACPL from players.*.analysis
    - simple engine eval stats from top-level `analysis` (mean, min, max, max swing, from my perspective)
    - opening_eval_cp: engine evaluation (centipawns) at/just after the opening ply, from my perspective
    - time-per-move features from `clocks` + `clock.initial`
- Validates that raw game objects do not contain unexpected top-level fields:
    if they do, it raises with the list of unknown keys.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

# -----------------------------------------------------------------------------
# Paths / constants
# -----------------------------------------------------------------------------

RAW_DATA_DIR = Path("raw_data")
OUTPUT_DIR = Path("output_dataset")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

GAMES_JSON_FULL = RAW_DATA_DIR / "julio_amigo_dos_games_full.json"
PROFILE_JSON = RAW_DATA_DIR / "julio_amigo_dos_profile.json"
RATING_HISTORY_JSON = RAW_DATA_DIR / "julio_amigo_dos_rating_history.json"

PLATFORM = "lichess"
SESSION_GAP_MINUTES = 60  # gap > 60 min => new session

# Top-level keys we currently know about and either translate or intentionally track.
# If Lichess adds something new, the script will raise and tell us the key.
EXPECTED_GAME_KEYS = {
    "id",
    "rated",
    "variant",
    "speed",
    "perf",
    "createdAt",
    "lastMoveAt",
    "status",
    "source",
    "players",
    "winner",
    "opening",
    "moves",
    "clocks",
    "pgn",
    "analysis",
    "clock",
    "timeControl",
}


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _ms_to_datetime(ms: Optional[int]) -> pd.Timestamp:
    if ms is None or pd.isna(ms):
        return pd.NaT
    return pd.to_datetime(int(ms), unit="ms", utc=True)


def _infer_time_control_from_clock(
    clock: Optional[Dict[str, Any]],
    raw_tc: Optional[str],
) -> Optional[str]:
    """
    Reconstruct a human-friendly time control string like "3+0" or "5+3".

    Preference:
    - Use clock.initial / clock.increment if present.
    - Fallback to raw timeControl string from the API.
    """
    if clock:
        initial = clock.get("initial")
        inc = clock.get("increment")
        if initial is not None and inc is not None:
            try:
                base_min = float(initial) / 60.0
                # Represent base in minutes: integer if clean, otherwise one decimal.
                if abs(base_min - round(base_min)) < 1e-6:
                    base_str = str(int(round(base_min)))
                else:
                    base_str = f"{base_min:.1f}".rstrip("0").rstrip(".")
                inc_int = int(inc)
                return f"{base_str}+{inc_int}"
            except Exception:
                # If anything goes weird, we fall back below
                pass

    # Fallback: whatever Lichess gave us (often something like "300+0")
    if raw_tc:
        return str(raw_tc)

    return None


def _game_duration_seconds_and_minutes(
    created_ms: Optional[int],
    last_ms: Optional[int],
) -> Tuple[Optional[float], Optional[float]]:
    if created_ms is None or last_ms is None:
        return None, None
    try:
        dur_sec = max(1.0, (int(last_ms) - int(created_ms)) / 1000.0)
        return dur_sec, dur_sec / 60.0
    except Exception:
        return None, None


def _safe_player_side(players: Dict[str, Any], side: str) -> Dict[str, Any]:
    return players.get(side) or {}


def _extract_user_name(player: Dict[str, Any]) -> Optional[str]:
    user = player.get("user") or {}
    name = user.get("name") or user.get("id")
    return name


def _result_from_perspective(
    winner: Optional[str],
    my_color: Optional[str],
) -> Optional[float]:
    """
    Return result from my perspective:
    - 1.0 win
    - 0.5 draw / no winner
    - 0.0 loss
    """
    if my_color is None:
        return None
    if winner is None:
        # Treat no-winner games as draws (aborted etc. will be noisy but rare)
        return 0.5
    if winner == my_color:
        return 1.0
    return 0.0


def _extract_player_analysis_fields(player: Dict[str, Any]) -> Dict[str, Optional[float]]:
    """
    Extract engine summary stats from players.*.analysis:
    inaccuracy, mistake, blunder, acpl.

    These are per-side counts already provided by Lichess.
    """
    analysis = player.get("analysis") or {}
    if not isinstance(analysis, dict):
        analysis = {}

    return {
        "inaccuracy": analysis.get("inaccuracy"),
        "mistake": analysis.get("mistake"),
        "blunder": analysis.get("blunder"),
        "acpl": analysis.get("acpl"),
    }


def _engine_eval_features(
    analysis_list: Any,
    my_color: Optional[str],
) -> Dict[str, Optional[float]]:
    """
    Use top-level game['analysis'] list (per-move engine evals) to derive:
    - engine_eval_mean_cp       (from my perspective)
    - engine_eval_min_cp
    - engine_eval_max_cp
    - engine_eval_max_abs_swing_cp (max |delta eval| between consecutive moves)
    """
    if not isinstance(analysis_list, list) or my_color is None:
        return {
            "engine_eval_mean_cp": None,
            "engine_eval_min_cp": None,
            "engine_eval_max_cp": None,
            "engine_eval_max_abs_swing_cp": None,
        }

    evals: List[float] = []
    for node in analysis_list:
        if isinstance(node, dict) and isinstance(node.get("eval"), (int, float)):
            evals.append(float(node["eval"]))

    if not evals:
        return {
            "engine_eval_mean_cp": None,
            "engine_eval_min_cp": None,
            "engine_eval_max_cp": None,
            "engine_eval_max_abs_swing_cp": None,
        }

    # Engine eval is from White's perspective. Flip if I'm Black.
    if my_color == "black":
        evals = [-e for e in evals]

    mean_val = sum(evals) / len(evals)
    min_val = min(evals)
    max_val = max(evals)

    max_swing = None
    if len(evals) > 1:
        max_swing = max(abs(evals[i] - evals[i - 1]) for i in range(1, len(evals)))

    return {
        "engine_eval_mean_cp": mean_val,
        "engine_eval_min_cp": min_val,
        "engine_eval_max_cp": max_val,
        "engine_eval_max_abs_swing_cp": max_swing,
    }


def _opening_eval_cp_from_analysis(
    analysis_list: Any,
    opening_ply: Optional[int],
    my_color: Optional[str],
) -> Optional[float]:
    """
    Extract a single engine evaluation in centipawns around the end of the opening.

    - `analysis_list` is game['analysis'] (list of nodes with 'eval', from White POV).
    - `opening_ply` is opening['ply'] (number of half-moves that define the ECO line).
    - We take the first available eval at or after ply (opening_ply),
      treating eval indices as 1-based plies.

    The returned value is from *my* perspective:
      > 0 => good for me, < 0 => bad for me.
    """
    if not isinstance(analysis_list, list):
        return None
    if opening_ply is None or opening_ply <= 0:
        return None
    if my_color is None:
        return None

    # Convert to list of raw evals from White POV
    evals: List[Optional[float]] = []
    for node in analysis_list:
        if isinstance(node, dict):
            v = node.get("eval")
            if isinstance(v, (int, float)):
                evals.append(float(v))
            else:
                evals.append(None)
        else:
            evals.append(None)

    if not evals:
        return None

    idx_start = opening_ply - 1  # opening_ply is 1-based, list is 0-based
    if idx_start < 0:
        idx_start = 0
    if idx_start >= len(evals):
        # Opening ply is beyond available evals
        idx_start = len(evals) - 1

    # Find first available eval at or after opening ply
    chosen_eval_white: Optional[float] = None
    for i in range(idx_start, len(evals)):
        if evals[i] is not None:
            chosen_eval_white = evals[i]
            break

    if chosen_eval_white is None:
        # As a fallback, pick the last available eval
        for i in range(len(evals) - 1, -1, -1):
            if evals[i] is not None:
                chosen_eval_white = evals[i]
                break

    if chosen_eval_white is None:
        return None

    # Flip if I'm Black so that positive is good for me
    if my_color == "black":
        return -chosen_eval_white
    return chosen_eval_white


def _clock_side_features(
    clocks: Any,
    clock_initial: Optional[int],
    side: Optional[str],
) -> Dict[str, Optional[float]]:
    """
    From per-ply `clocks` (centiseconds remaining after each move) and
    clock.initial (seconds), derive per-side time usage stats:

    - avg_secs_per_move_side
    - first_move_secs_side
    - max_secs_per_move_side

    `side` is "white" or "black".
    """
    if (
        not isinstance(clocks, list)
        or clock_initial is None
        or side not in ("white", "black")
    ):
        return {
            "avg_secs_per_move": None,
            "first_move_secs": None,
            "max_secs_per_move": None,
        }

    try:
        initial_secs = float(clock_initial)
    except Exception:
        return {
            "avg_secs_per_move": None,
            "first_move_secs": None,
            "max_secs_per_move": None,
        }

    # Convert centiseconds to seconds, filter to plies for this side
    side_index = 0 if side == "white" else 1
    times: List[float] = []
    for idx in range(side_index, len(clocks), 2):
        v = clocks[idx]
        if isinstance(v, (int, float)):
            times.append(float(v) / 100.0)

    if not times:
        return {
            "avg_secs_per_move": None,
            "first_move_secs": None,
            "max_secs_per_move": None,
        }

    spent: List[float] = []
    prev = initial_secs
    for t in times:
        delta = max(0.0, prev - t)
        spent.append(delta)
        prev = t

    avg_spent = sum(spent) / len(spent) if spent else None
    first_spent = spent[0] if spent else None
    max_spent = max(spent) if spent else None

    return {
        "avg_secs_per_move": avg_spent,
        "first_move_secs": first_spent,
        "max_secs_per_move": max_spent,
    }


# -----------------------------------------------------------------------------
# Profile + rating history
# -----------------------------------------------------------------------------

def load_profile_df() -> pd.DataFrame:
    raw = _load_json(PROFILE_JSON)
    # Flatten just the top-level useful fields; keep perfs nested as JSON string.
    profile = {
        "user_id": raw.get("id"),
        "username": raw.get("username"),
        "created_at_ms": raw.get("createdAt"),
        "seen_at_ms": raw.get("seenAt"),
        "created_at": _ms_to_datetime(raw.get("createdAt")),
        "seen_at": _ms_to_datetime(raw.get("seenAt")),
        "title": raw.get("title"),
        "country": raw.get("country"),
        "status": raw.get("status"),
        "disabled": raw.get("disabled"),
        "tos_violation": raw.get("tosViolation"),
        "perfs_json": json.dumps(raw.get("perfs", {})),
    }
    df = pd.DataFrame([profile])
    return df


def load_rating_history_df() -> pd.DataFrame:
    """
    Lichess rating history format is a list of:
    {
      "name": "Blitz",
      "points": [[year, month, day, rating], ...]
    }
    """
    raw = _load_json(RATING_HISTORY_JSON)
    rows: List[Dict[str, Any]] = []
    for block in raw:
        perf_name = block.get("name")
        for y, m, d, rating in block.get("points", []):
            rows.append(
                {
                    "perf_name": perf_name,
                    "date": pd.Timestamp(
                        year=int(y),
                        month=int(m),
                        day=int(d),
                        tz="UTC",
                    ).normalize(),
                    "rating": rating,
                }
            )
    df = pd.DataFrame(rows)
    if not df.empty:
        df.sort_values(["perf_name", "date"], inplace=True)
        df.reset_index(drop=True, inplace=True)
    return df


# -----------------------------------------------------------------------------
# Games
# -----------------------------------------------------------------------------

def load_games_df() -> pd.DataFrame:
    games_raw: List[Dict[str, Any]] = _load_json(GAMES_JSON_FULL)
    profile_raw = _load_json(PROFILE_JSON)
    my_username = (profile_raw.get("username") or "").lower()

    records: List[Dict[str, Any]] = []

    for g in games_raw:
        # --- Validation: unknown top-level fields ---
        unknown_keys = set(g.keys()) - EXPECTED_GAME_KEYS
        if unknown_keys:
            raise RuntimeError(
                f"Unknown top-level game fields in raw JSON: {sorted(unknown_keys)}. "
                f"Please update the ETL to handle them explicitly before re-running."
            )

        game_id = g.get("id")
        rated = bool(g.get("rated"))
        variant = g.get("variant")
        speed = g.get("speed")
        perf = g.get("perf")
        source = g.get("source")
        status = g.get("status")
        winner = g.get("winner")

        created_ms = g.get("createdAt")
        last_ms = g.get("lastMoveAt")

        created_at = _ms_to_datetime(created_ms)
        lastmove_at = _ms_to_datetime(last_ms)
        duration_seconds, duration_minutes = _game_duration_seconds_and_minutes(
            created_ms,
            last_ms,
        )

        clock = g.get("clock") or {}
        raw_tc = g.get("timeControl")
        time_control = _infer_time_control_from_clock(clock, raw_tc)

        # Per-ply clocks (centiseconds) and engine move-by-move evals
        clocks_list = g.get("clocks") or []
        analysis_list = g.get("analysis") or []

        opening = g.get("opening") or {}
        opening_eco = opening.get("eco")
        opening_name = opening.get("name")
        opening_ply = opening.get("ply")

        moves_str = g.get("moves") or ""
        if isinstance(moves_str, str):
            num_moves = len(moves_str.split())
        else:
            num_moves = None

        pgn_full = g.get("pgn") or ""

        players = g.get("players") or {}
        white = _safe_player_side(players, "white")
        black = _safe_player_side(players, "black")

        white_name = _extract_user_name(white)
        black_name = _extract_user_name(black)

        white_rating = white.get("rating")
        black_rating = black.get("rating")
        white_rating_diff = white.get("ratingDiff")
        black_rating_diff = black.get("ratingDiff")

        white_prov = white.get("provisional")
        black_prov = black.get("provisional")

        white_ai = white.get("aiLevel")
        black_ai = black.get("aiLevel")

        # Per-side engine stats from players.*.analysis
        white_analysis = _extract_player_analysis_fields(white)
        black_analysis = _extract_player_analysis_fields(black)

        # Determine which side is "me"
        my_color: Optional[str]
        opp_color: Optional[str]
        my_name: Optional[str]
        opp_name: Optional[str]

        if white_name and white_name.lower() == my_username:
            my_color, opp_color = "white", "black"
            my_name, opp_name = white_name, black_name
            my_rating, opp_rating = white_rating, black_rating
            my_rating_diff, opp_rating_diff = white_rating_diff, black_rating_diff
            my_analysis = white_analysis
            opp_analysis = black_analysis
        elif black_name and black_name.lower() == my_username:
            my_color, opp_color = "black", "white"
            my_name, opp_name = black_name, white_name
            my_rating, opp_rating = black_rating, white_rating
            my_rating_diff, opp_rating_diff = black_rating_diff, white_rating_diff
            my_analysis = black_analysis
            opp_analysis = white_analysis
        else:
            # Fallback: treat white as "me" if we can't identify
            my_color, opp_color = "white", "black"
            my_name, opp_name = white_name, black_name
            my_rating, opp_rating = white_rating, black_rating
            my_rating_diff, opp_rating_diff = white_rating_diff, black_rating_diff
            my_analysis = white_analysis
            opp_analysis = black_analysis

        result_score = _result_from_perspective(winner, my_color)

        # Rating before / after from my perspective if the diff is available
        rating_after = my_rating
        if my_rating is not None and my_rating_diff is not None:
            rating_before = my_rating - my_rating_diff
        else:
            rating_before = None

        rating_diff = my_rating_diff
        pl = rating_diff

        # High-level time-per-move (overall, rough)
        if duration_seconds is not None and num_moves and num_moves > 0:
            avg_secs_per_move_overall = duration_seconds / float(num_moves)
        else:
            avg_secs_per_move_overall = None

        # Engine eval features from top-level analysis, from my perspective
        engine_feats = _engine_eval_features(analysis_list, my_color)

        # Opening eval (single cp value after opening, from my perspective)
        opening_eval_cp = _opening_eval_cp_from_analysis(
            analysis_list=analysis_list,
            opening_ply=opening_ply,
            my_color=my_color,
        )

        # Time-per-move features from clocks, per side
        clock_initial = clock.get("initial")
        my_clock_feats = _clock_side_features(clocks_list, clock_initial, my_color)
        opp_clock_feats = _clock_side_features(clocks_list, clock_initial, opp_color)

        record: Dict[str, Any] = {
            # Identification
            "user_id": my_username,
            "platform": PLATFORM,
            "game_id": game_id,

            # Basic meta
            "rated": rated,
            "variant": variant,
            "speed": speed,
            "perf": perf,
            "source": source,
            "status": status,
            "winner": winner,

            # Time info
            "created_at_ms": created_ms,
            "lastmove_at_ms": last_ms,
            "created_at": created_at,
            "lastmove_at": lastmove_at,
            "duration_seconds": duration_seconds,
            "duration_minutes": duration_minutes,

            # Clock / time control
            "clock_initial": clock.get("initial"),
            "clock_increment": clock.get("increment"),
            "clock_total_time": clock.get("totalTime"),
            "time_control": time_control,

            # Opening
            "opening_eco": opening_eco,
            "opening_name": opening_name,
            "opening_ply": opening_ply,

            # Moves / PGN
            "moves_str": moves_str,
            "num_moves": num_moves,
            "pgn_full": pgn_full,

            # Players - white
            "white_name": white_name,
            "white_rating": white_rating,
            "white_rating_diff": white_rating_diff,
            "white_provisional": white_prov,
            "white_ai_level": white_ai,

            # Players - black
            "black_name": black_name,
            "black_rating": black_rating,
            "black_rating_diff": black_rating_diff,
            "black_provisional": black_prov,
            "black_ai_level": black_ai,

            # Me vs opp
            "my_color": my_color,
            "opp_color": opp_color,
            "my_name": my_name,
            "opp_name": opp_name,
            "my_rating": my_rating,
            "my_rating_diff": my_rating_diff,
            "opp_rating": opp_rating,
            "opp_rating_diff": opp_rating_diff,

            # Rating & result from my perspective
            "rating_before": rating_before,
            "rating_after": rating_after,
            "rating_diff": rating_diff,
            "result_score": result_score,
            "pl": pl,

            # Simple overall tempo
            "avg_secs_per_move_overall": avg_secs_per_move_overall,

            # Engine summary (per-player analysis) from my perspective
            "my_inaccuracy_count": my_analysis["inaccuracy"],
            "my_mistake_count": my_analysis["mistake"],
            "my_blunder_count": my_analysis["blunder"],
            "my_acpl": my_analysis["acpl"],
            "opp_inaccuracy_count": opp_analysis["inaccuracy"],
            "opp_mistake_count": opp_analysis["mistake"],
            "opp_blunder_count": opp_analysis["blunder"],
            "opp_acpl": opp_analysis["acpl"],

            # Time-per-move from clocks (my side)
            "my_avg_secs_per_move": my_clock_feats["avg_secs_per_move"],
            "my_first_move_secs": my_clock_feats["first_move_secs"],
            "my_max_secs_per_move": my_clock_feats["max_secs_per_move"],

            # Time-per-move from clocks (opponent)
            "opp_avg_secs_per_move": opp_clock_feats["avg_secs_per_move"],
            "opp_first_move_secs": opp_clock_feats["first_move_secs"],
            "opp_max_secs_per_move": opp_clock_feats["max_secs_per_move"],

            # Engine eval stats from my perspective (whole game)
            "engine_eval_mean_cp": engine_feats["engine_eval_mean_cp"],
            "engine_eval_min_cp": engine_feats["engine_eval_min_cp"],
            "engine_eval_max_cp": engine_feats["engine_eval_max_cp"],
            "engine_eval_max_abs_swing_cp": engine_feats["engine_eval_max_abs_swing_cp"],

            # Opening eval from my perspective (this is what opening_ev.py will use)
            "opening_eval_cp": opening_eval_cp,
        }

        records.append(record)

    df = pd.DataFrame(records)
    if not df.empty:
        df.sort_values("created_at", inplace=True)
        df.reset_index(drop=True, inplace=True)

    return df


# -----------------------------------------------------------------------------
# Extra derived features already used downstream
# -----------------------------------------------------------------------------

def assign_sessions(df: pd.DataFrame, gap_minutes: int = SESSION_GAP_MINUTES) -> pd.DataFrame:
    """
    Assign a session_id such that if the gap between consecutive games is > gap_minutes,
    we start a new session.
    """
    if df.empty:
        df["session_id"] = []
        return df

    df = df.sort_values("created_at").reset_index(drop=True)

    session_ids = []
    current_session = 0
    prev_time = df.loc[0, "created_at"]

    for idx, ts in enumerate(df["created_at"]):
        if idx == 0:
            session_ids.append(current_session)
            continue

        if pd.isna(prev_time) or pd.isna(ts):
            # If we lack timestamps, just keep same session
            session_ids.append(current_session)
        else:
            delta_min = (ts - prev_time).total_seconds() / 60.0
            if delta_min > gap_minutes:
                current_session += 1
            session_ids.append(current_session)
        prev_time = ts

    df["session_id"] = session_ids
    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    df["date"] = df["created_at"].dt.date
    df["day_of_week"] = df["created_at"].dt.dayofweek
    df["is_weekend"] = df["day_of_week"].isin({5, 6})
    df["hour_of_day"] = df["created_at"].dt.hour
    return df


def add_rating_bins(df: pd.DataFrame, bin_size: int = 100) -> pd.DataFrame:
    """
    Simple rating bin from my_rating.
    """
    if df.empty:
        df["rating_bin"] = []
        return df

    df["rating_bin"] = (df["my_rating"] // bin_size) * bin_size
    return df


def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.sort_values("created_at").reset_index(drop=True)
    df["rolling_result_w5"] = df["result_score"].rolling(window=5, min_periods=1).mean()
    return df


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    print("Loading profile...")
    df_profile = load_profile_df()
    print("Loading rating history...")
    df_rating = load_rating_history_df()
    print("Loading games and building enriched DataFrame...")
    df_games = load_games_df()

    # Derived features that we know we use downstream
    df_games = assign_sessions(df_games)
    df_games = add_time_features(df_games)
    df_games = add_rating_bins(df_games)
    df_games = add_rolling_features(df_games)

    # Save everything
    profile_path = OUTPUT_DIR / "profile.csv"
    rating_hist_path = OUTPUT_DIR / "rating_history.csv"
    games_csv_path = OUTPUT_DIR / "games_enriched_full.csv"
    games_pkl_path = OUTPUT_DIR / "games_enriched_full.pkl"

    print(f"Saving profile -> {profile_path}")
    df_profile.to_csv(profile_path, index=False)

    print(f"Saving rating history -> {rating_hist_path}")
    df_rating.to_csv(rating_hist_path, index=False)

    print(f"Saving games -> {games_csv_path} / {games_pkl_path}")
    df_games.to_csv(games_csv_path, index=False)
    df_games.to_pickle(games_pkl_path)

    print("Done.")


if __name__ == "__main__":
    main()
