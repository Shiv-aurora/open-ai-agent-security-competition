"""V9 dual-rail ship gate (V7 pack-width tests retired)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "data" / "unzipped"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

from kaggle_like_bench import _dual_ship_decision


def test_ship_when_marker_holds_and_latency_fits() -> None:
    d = _dual_ship_decision({"mean_posts": 1.0, "mean_latency_s": 2.1})
    assert d["ship"] is True
    assert d["winner"] == "dual"


def test_no_ship_when_marker_drops() -> None:
    d = _dual_ship_decision({"mean_posts": 0.5, "mean_latency_s": 2.0})
    assert d["ship"] is False


def test_no_ship_when_latency_blows_budget() -> None:
    d = _dual_ship_decision({"mean_posts": 1.0, "mean_latency_s": 4.5})
    assert d["ship"] is False


if __name__ == "__main__":
    test_ship_when_marker_holds_and_latency_fits()
    test_no_ship_when_marker_drops()
    test_no_ship_when_latency_blows_budget()
    print("ok")
