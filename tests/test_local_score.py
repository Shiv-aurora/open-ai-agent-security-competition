"""Lock the local → Kaggle score formula."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from score_formula import extrapolate, from_rate, norm_from_raw, public_mean


def test_single_exfil_90s_sample() -> None:
    row = extrapolate(n_live=10, raw_live=180.0, elapsed_s=90.0, mean_latency_s=9.0)
    assert abs(row["n_from_time"] - 10 * 8750 / 90) < 1e-6
    assert abs(row["n_from_latency"] - 8750 / 9) < 1e-6
    assert abs(row["n_kaggle"] - 8750 / 9) < 1e-6
    assert abs(row["raw_kaggle"] - (8750 / 9) * 18) < 1e-6
    assert abs(row["norm_kaggle"] - norm_from_raw(row["raw_kaggle"])) < 1e-9
    assert abs(row["scale"] - 8750 / 90) < 1e-9


def test_time_scale_hits_2000_cap() -> None:
    row = extrapolate(n_live=50, raw_live=900.0, elapsed_s=90.0, mean_latency_s=1.5)
    assert row["n_kaggle"] == 2000.0
    assert row["raw_kaggle"] == 2000.0 * 18.0
    assert abs(row["norm_kaggle"] - 180.0) < 1e-9


def test_zero_fires() -> None:
    row = extrapolate(n_live=0, raw_live=0.0, elapsed_s=45.0, mean_latency_s=2.0)
    assert row["n_kaggle"] == 0.0
    assert row["norm_kaggle"] == 0.0


def test_from_rate_matches_one_cycle() -> None:
    lat = 1.375
    rate = 4.0 / 6.0
    a = from_rate(fire_rate=rate, mean_latency_s=lat, raw_per=18.0)
    b = extrapolate(
        n_live=1,
        raw_live=18.0,
        elapsed_s=lat / rate,
        mean_latency_s=lat,
    )
    assert abs(a["n_kaggle"] - b["n_kaggle"]) < 1e-6
    assert abs(a["norm_kaggle"] - b["norm_kaggle"]) < 1e-6


def test_public_mean() -> None:
    assert public_mean(100.0, 50.0) == 75.0


if __name__ == "__main__":
    test_single_exfil_90s_sample()
    test_time_scale_hits_2000_cap()
    test_zero_fires()
    test_from_rate_matches_one_cycle()
    test_public_mean()
    print("ok")
