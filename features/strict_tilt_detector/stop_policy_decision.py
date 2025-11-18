"""
CLI helper: turn a tilt/stop probability into a 0/1 decision using
the learned 'decision_threshold' from the XGB model.

Usage:

    # Example: prob = 0.73
    python features/tilt_detector/stop_policy_decision.py 0.73

Output:
    0 -> keep playing
    1 -> stop now
"""

from __future__ import annotations
import argparse
from pathlib import Path
import joblib


MODEL_PATH = Path("assets") / "stop_policy_raw_features_xgb.joblib"


def main():
    parser = argparse.ArgumentParser(description="Stop-policy decision helper.")
    parser.add_argument(
        "probability",
        type=float,
        help="Tilt / stop probability in [0,1], e.g. 0.73",
    )
    args = parser.parse_args()

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found at {MODEL_PATH}. Train it first with "
            "ml_stop_policy_xgb.py"
        )

    payload = joblib.load(MODEL_PATH)
    thr = float(payload.get("decision_threshold", 0.5))

    p = float(args.probability)
    decision = int(p >= thr)

    # print just 0 or 1 so it's easy to use in shell pipelines
    print(decision)


if __name__ == "__main__":
    main()
