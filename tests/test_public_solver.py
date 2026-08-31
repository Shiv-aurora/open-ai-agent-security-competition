"""Public 4-row solver: V5=90, V6≈60.6, V8=83, V9 Gemma cannot still be 180."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from public_solver import (
    OSS_SINGLE_S,
    invert,
    n_from_latency,
    predict,
    row_from_nk,
)


def test_v5_is_two_publics_at_180() -> None:
    p = predict(gemma_k=1, gemma_lat_s=0.92, oss_k=1, oss_lat_s=OSS_SINGLE_S)
    assert abs(p["displayed"] - 90.0) < 0.05
    assert p["gemma_public"]["n"] == 2000
    assert p["oss_public"]["n"] == 2000
    assert p["gemma_public"]["norm"] == 180.0
    assert p["oss_public"]["norm"] == 180.0


def test_v6_pack4_k1_matches_hosted() -> None:
    p = predict(
        gemma_k=1,
        gemma_lat_s=4 * 0.92,
        oss_k=1,
        oss_lat_s=4 * OSS_SINGLE_S,
    )
    assert abs(p["displayed"] - 60.555) < 0.5
    assert p["gemma_public"]["n"] == 2000
    assert p["oss_public"]["n"] < 800
    assert p["gemma_public"]["norm"] == 180.0


def test_v8_no_stack_slow_oss_is_83() -> None:
    p = predict(gemma_k=1, gemma_lat_s=1.87, oss_k=1, oss_lat_s=1.63 * OSS_SINGLE_S)
    assert abs(p["displayed"] - 83.0) < 1.0
    assert p["gemma_public"]["norm"] == 180.0


def test_v9_gemma_cannot_stay_180() -> None:
    inv = invert(39.870)
    assert inv["gemma_180_possible"] is False
    assert inv["oss_if_gemma_180"] < 0


def test_v8_gemma_k2_plus_oss_alive_is_not_83() -> None:
    p = predict(gemma_k=1.867, gemma_lat_s=1.869, oss_k=1, oss_lat_s=OSS_SINGLE_S)
    # This is the V10 upside (~125), not the V8 score.
    assert p["displayed"] > 110
    assert p["displayed"] < 140


def test_n_from_latency_cap() -> None:
    assert n_from_latency(0.92) == 2000
    assert abs(n_from_latency(OSS_SINGLE_S) - 2000) < 1e-6
    n = n_from_latency(4 * OSS_SINGLE_S)
    assert 680 < n < 720


def test_row_k2() -> None:
    r = row_from_nk(2000, 2)
    assert abs(r["norm"] - 340.0) < 1e-9


def test_displayed_is_quarter_not_half() -> None:
    p = predict(gemma_k=1, gemma_lat_s=0.92, oss_k=1, oss_lat_s=0.92)
    assert abs(p["displayed"] - 90.0) < 1e-6


if __name__ == "__main__":
    test_v5_is_two_publics_at_180()
    test_v6_pack4_k1_matches_hosted()
    test_v8_no_stack_slow_oss_is_83()
    test_v9_gemma_cannot_stay_180()
    test_v8_gemma_k2_plus_oss_alive_is_not_83()
    test_n_from_latency_cap()
    test_row_k2()
    test_displayed_is_quarter_not_half()
    print("ok")
