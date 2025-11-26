import os
import json
from typing import Iterator, Dict, Any, Optional
from pathlib import Path

import requests

BASE_URL = "https://lichess.org"
USERNAME = "julio_amigo_dos"  # change if needed

# Optional: personal access token for better rate limits / stability.
# Create at: https://lichess.org/account/oauth/token
# Then: export LICHESS_TOKEN="your_token_here"
API_TOKEN = os.getenv("LICHESS_TOKEN")

# Output layout aligned with your project
RAW_DATA_DIR = Path("raw_data")
RAW_DATA_DIR.mkdir(exist_ok=True)


def get_user_profile(username: str) -> Dict[str, Any]:
    """
    GET /api/user/{username}
    Public profile + perfs (ratings, game counts, etc.).
    """
    url = f"{BASE_URL}/api/user/{username}"
    resp = requests.get(url)
    resp.raise_for_status()
    return resp.json()


def get_rating_history(username: str):
    """
    GET /api/user/{username}/rating-history
    Returns rating history per speed (bullet, blitz, rapid, etc.).
    """
    url = f"{BASE_URL}/api/user/{username}/rating-history"
    resp = requests.get(url)
    resp.raise_for_status()
    return resp.json()


def stream_user_games(
    username: str,
    max_games: Optional[int] = None,
    perf_type: Optional[str] = None,
    analysed_only: bool = False,
) -> Iterator[Dict[str, Any]]:
    """
    Stream all games for a user as Python dicts, using NDJSON.

    Endpoint: GET /api/games/user/{username}

    We request:
      - moves:   all PGN moves
      - tags:    PGN tags (ratings, result, time control, etc.)
      - clocks:  per-move clock comments (when available)
      - evals:   engine eval comments (when available)
      - opening: ECO + opening name
      - pgnInJson: full PGN (including comments) embedded in the JSON object

    Optional:
      - perf_type: "bullet", "blitz", "rapid", ... if you want to restrict
      - analysed_only: if True, ask for games with analysis data (not guaranteed,
        but can increase chance of having evals/ACPL).
    """
    url = f"{BASE_URL}/api/games/user/{username}"

    params = {
        "moves": "true",
        "pgnInJson": "true",
        "tags": "true",
        "clocks": "true",
        "evals": "true",
        "opening": "true",
        "finished": "true",  # only finished games
        "sort": "dateAsc",   # oldest first (easier for time-series analysis)
    }

    if max_games is not None:
        params["max"] = str(max_games)

    if perf_type is not None:
        # e.g. "bullet", "blitz", "rapid", "classical", "correspondence"
        params["perfType"] = perf_type

    if analysed_only:
        # From docs: 'analysed' filter hints for games that have analysis.
        # Not strict, but useful if you care about evals/ACPL most.
        params["analysed"] = "true"

    headers = {
        "Accept": "application/x-ndjson",  # NDJSON = 1 game JSON per line
    }
    if API_TOKEN:
        headers["Authorization"] = f"Bearer {API_TOKEN}"

    with requests.get(url, params=params, headers=headers, stream=True) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines(decode_unicode=True):
            if not line:
                continue
            yield json.loads(line)


def main():
    # 1) Fetch and save user profile
    profile = get_user_profile(USERNAME)
    profile_path = RAW_DATA_DIR / f"{USERNAME}_profile.json"
    with profile_path.open("w", encoding="utf-8") as f:
        json.dump(profile, f, ensure_ascii=False, indent=2)
    print(f"Saved profile to {profile_path}")

    # 2) Fetch and save rating history
    rating_history = get_rating_history(USERNAME)
    rh_path = RAW_DATA_DIR / f"{USERNAME}_rating_history.json"
    with rh_path.open("w", encoding="utf-8") as f:
        json.dump(rating_history, f, ensure_ascii=False, indent=2)
    print(f"Saved rating history to {rh_path}")

    # 3) Stream all games and save both as NDJSON and as one big JSON array
    ndjson_path = RAW_DATA_DIR / f"{USERNAME}_games_full.ndjson"
    json_path = RAW_DATA_DIR / f"{USERNAME}_games_full.json"

    games_list = []
    count = 0

    # Example: only bullet. If you want *all* perf types, set perf_type=None.
    perf_type = None  # or "bullet" for just 1+0, etc.

    # Example: don't force analysed_only; keep all games for completeness.
    analysed_only = False

    with ndjson_path.open("w", encoding="utf-8") as ndjson_file:
        for game in stream_user_games(
            USERNAME,
            max_games=None,
            perf_type=perf_type,
            analysed_only=analysed_only,
        ):
            ndjson_file.write(json.dumps(game, ensure_ascii=False) + "\n")
            games_list.append(game)
            count += 1
            if count % 100 == 0:
                print(f"...fetched {count} games so far")

    with json_path.open("w", encoding="utf-8") as json_file:
        json.dump(games_list, json_file, ensure_ascii=False, indent=2)

    print(f"Finished. Fetched {count} games.")
    print(f"- NDJSON games: {ndjson_path}")
    print(f"- JSON array  : {json_path}")


if __name__ == "__main__":
    main()
