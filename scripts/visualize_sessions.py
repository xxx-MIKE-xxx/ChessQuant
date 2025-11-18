import json
from pathlib import Path
from datetime import datetime, timedelta

import matplotlib.pyplot as plt

USERNAME = "julio_amigo_dos"
GAMES_JSON = Path(f"{USERNAME}_games.json")

# Session detection settings
SESSION_GAP_MINUTES = 30  # minutes between games that starts a new session

# Output folder
OUTPUT_DIR = Path("session_P_L")
OUTPUT_DIR.mkdir(exist_ok=True)

# Plot mode:
#   "score"  -> cumulative P/L using +1 win, 0 draw, -1 loss (recommended for tilt)
#   "rating" -> rating curve within session (start rating + change after each game)
MODE = "score"  # change to "rating" if you want rating curves


def lichess_timestamp_to_dt(ms: int) -> datetime:
    """Lichess uses milliseconds since epoch."""
    return datetime.utcfromtimestamp(ms / 1000.0)


def get_user_color(game, username: str):
    """Return 'white' or 'black' depending on which side you played."""
    players = game.get("players", {})
    white_name = players.get("white", {}).get("user", {}).get("name", "")
    black_name = players.get("black", {}).get("user", {}).get("name", "")

    uname_lower = username.lower()
    if white_name.lower() == uname_lower:
        return "white"
    if black_name.lower() == uname_lower:
        return "black"
    return None


def result_to_pl(game, username: str) -> int:
    """
    Map game result to P/L from `username`'s perspective:
      +1 = win
       0 = draw / aborted
      -1 = loss
    """
    players = game.get("players", {})
    winner = game.get("winner")  # "white", "black", or None
    status = game.get("status")

    my_color = get_user_color(game, username)
    if my_color is None:
        # Shouldn't happen in your own export, but be safe
        return 0

    # Draw-ish outcomes
    if status in (
        "draw",
        "stalemate",
        "outoftimeVsInsufficientMaterial",
        "cheat",
        "noStart",
    ):
        return 0

    if winner is None:
        # Aborted / unknown -> treat as neutral
        return 0

    if winner == my_color:
        return +1
    else:
        return -1


def rating_series_for_session(session_games, username: str):
    """
    Build a rating curve for one session.

    We start at rating *before* the first game, then add ratingDiff after each game.
    So:
      x = 0..N
      y[0] = rating at session start
      y[i] = rating after i-th game
    """
    if not session_games:
        return [], []

    # figure out your color in the first game
    first_color = get_user_color(session_games[0], username)
    if first_color is None:
        return [], []

    # starting rating before first game
    first_players = session_games[0]["players"][first_color]
    cur_rating = first_players.get("rating")
    if cur_rating is None:
        return [], []

    x = [0]
    y = [cur_rating]

    for i, g in enumerate(session_games, start=1):
        my_color = get_user_color(g, username)
        if my_color is None:
            # if weird, just repeat rating
            x.append(i)
            y.append(cur_rating)
            continue

        p = g["players"][my_color]
        diff = p.get("ratingDiff", 0)
        cur_rating += diff
        x.append(i)
        y.append(cur_rating)

    return x, y


def score_series_for_session(session_games, username: str):
    """
    Build cumulative P/L curve (score mode).

    We start at 0 and add +1/0/-1 after each game:
      x = 0..N
      y[0] = 0
    """
    cumulative_pl = [0]
    total = 0

    for g in session_games:
        r = result_to_pl(g, username)
        total += r
        cumulative_pl.append(total)

    x = list(range(0, len(session_games) + 1))
    y = cumulative_pl
    return x, y


def split_into_sessions(games, gap_minutes=30):
    """
    Given games sorted by time, split into sessions based on time gaps.
    Returns a list of lists (sessions), where each session is a list of games.
    """
    sessions = []
    current = []
    last_time = None
    gap = timedelta(minutes=gap_minutes)

    for g in games:
        t = lichess_timestamp_to_dt(g["createdAt"])
        if last_time is None or t - last_time <= gap:
            current.append(g)
        else:
            if current:
                sessions.append(current)
            current = [g]
        last_time = t

    if current:
        sessions.append(current)

    return sessions


def plot_session(session_games, session_index: int, username: str, mode: str = "score"):
    """
    Plot a single session according to MODE and save to OUTPUT_DIR.
    Also annotate the session date/time in the title.
    """
    if not session_games:
        return

    start_time = lichess_timestamp_to_dt(session_games[0]["createdAt"])
    end_time = lichess_timestamp_to_dt(session_games[-1]["createdAt"])

    date_str = start_time.strftime("%Y-%m-%d")
    time_range_str = f"{start_time.strftime('%H:%M')}–{end_time.strftime('%H:%M')} UTC"

    if mode == "rating":
        x, y = rating_series_for_session(session_games, username)
        y_label = "Rating"
        mode_desc = "Rating"
    else:
        x, y = score_series_for_session(session_games, username)
        y_label = "Cumulative P/L (+1 win, 0 draw, -1 loss)"
        mode_desc = "Cumulative P/L"

    if not x or not y:
        print(f"Session {session_index}: no data for mode={mode}, skipping.")
        return

    plt.figure()
    plt.plot(x, y, marker="o")

    if mode == "score":
        # zero line useful for P/L
        plt.axhline(0, linestyle="--", linewidth=1)

    plt.title(
        f"Session {session_index} – {mode_desc}\n"
        f"{date_str} {time_range_str}"
    )
    plt.xlabel("Game number in session")
    plt.ylabel(y_label)
    plt.grid(True)

    start_str = start_time.strftime("%Y-%m-%d_%H-%M")
    filename = OUTPUT_DIR / f"session_{session_index:03d}_{start_str}_{mode}.png"
    plt.savefig(filename, bbox_inches="tight")
    plt.close()

    print(
        f"Saved Session {session_index} plot: {filename.name} "
        f"(games: {len(session_games)})"
    )


def main():
    if not GAMES_JSON.exists():
        raise FileNotFoundError(
            f"{GAMES_JSON} not found. Make sure it's in the current directory."
        )

    with open(GAMES_JSON, "r", encoding="utf-8") as f:
        games = json.load(f)

    # Sort games by creation time (oldest first)
    games.sort(key=lambda g: g["createdAt"])

    # Split into sessions
    sessions = split_into_sessions(games, gap_minutes=SESSION_GAP_MINUTES)
    print(f"Found {len(sessions)} sessions.")

    for i, sess in enumerate(sessions, start=1):
        plot_session(sess, i, USERNAME, mode=MODE)


if __name__ == "__main__":
    main()
