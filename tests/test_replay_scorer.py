"""Reliable scorer: miss-as-zero, V5 floor, V6-shaped extra-message tax."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from score_formula import (
    displayed_4row,
    forecast_miss_zero,
    pessimistic_k1,
    raw_from_posts,
    verdict_vs_v5,
)


def test_raw_from_posts() -> None:
    assert raw_from_posts(0) == 0.0
    assert raw_from_posts(1) == 18.0
    assert raw_from_posts(4) == 66.0


def test_misses_are_zero_raw() -> None:
    fam = forecast_miss_zero([4, 0], [3.8, 3.8], n_messages=4)
    assert abs(fam["mean_posts"] - 2.0) < 1e-9
    assert abs(fam["raw_per"] - 33.0) < 1e-9  # (66+0)/2


def test_v6_hosted_shape_is_unsafe() -> None:
    v5 = forecast_miss_zero([1] * 20, [1.03] * 20, n_messages=1)
    # 4 messages * 1.6s OSS single → 6.4s, K=1, n≈1367, displayed ~61
    pess = pessimistic_k1(n_messages=4, single_lat_s=1.6, fire_rate=1.0)
    attack = forecast_miss_zero([1] * 20, [6.5] * 20, n_messages=4)
    attack["pess_displayed"] = pess["displayed_both_private0"]
    d = verdict_vs_v5(attack, v5)
    assert d["ship"] is False
    assert d["verdict"] == "UNSAFE_V6_SHAPE"
    assert pess["displayed_both_private0"] < 70


def test_v5_singles_are_floor() -> None:
    v5 = forecast_miss_zero([1] * 20, [1.0] * 20, n_messages=1)
    assert abs(v5["displayed_both_private0"] - 90.0) < 1e-6
    d = verdict_vs_v5(v5, v5)
    assert d["verdict"] == "PUBLIC_FLOOR"
    assert d["ship"] is True


def test_true_k4_at_v5_latency_beats_v5() -> None:
    v5 = forecast_miss_zero([1] * 20, [1.0] * 20, n_messages=1)
    attack = forecast_miss_zero([4] * 20, [2.0] * 20, n_messages=1)
    attack["pess_displayed"] = v5["displayed_both_private0"]
    d = verdict_vs_v5(attack, v5)
    assert d["verdict"] == "BEATS_V5"
    assert d["ship"] is True


def test_displayed_4row() -> None:
    assert displayed_4row(180.0) == 90.0
    assert displayed_4row(180.0, 120.0) == 75.0


def test_pessimistic_pack4_oss() -> None:
    p = pessimistic_k1(n_messages=4, single_lat_s=1.6, fire_rate=1.0)
    assert p["n_kaggle"] < 1400
    assert p["displayed_both_private0"] < 70


def test_pessimistic_dual_rail_still_2000() -> None:
    p = pessimistic_k1(n_messages=2, single_lat_s=1.6, fire_rate=1.0)
    assert p["n_kaggle"] == 2000.0
    assert abs(p["displayed_both_private0"] - 90.0) < 1e-6


def test_cooperating_pipeline_v5_and_pack() -> None:
    """Toy poster: V5 posts once, pack-4 posts four times. Not a model forecast."""
    from replay_scorer import cooperating_factory, run_reliable_score

    report = run_reliable_score(
        cooperating_factory(),
        env_selection="sandbox",
        n_per=3,
        replay_s=30.0,
        include_pack_canary=True,
        pack_n=2,
        pack_m=4,
        pipeline_only=True,
    )
    v5 = report["families"]["v5_single"]
    pack = report["families"]["pack_m4"]
    att = report["families"]["attack"]
    assert v5["mean_posts"] == 1.0
    assert v5["fire_rate"] == 1.0
    assert abs(pack["mean_posts"] - 4.0) < 1e-9
    assert att["n_messages"] == 2.0  # V11 none-env farms dual-rail pairs
    assert report["verdict"] == "PIPELINE_ONLY"
    assert report["ship"] is False


if __name__ == "__main__":
    test_raw_from_posts()
    test_misses_are_zero_raw()
    test_v6_hosted_shape_is_unsafe()
    test_v5_singles_are_floor()
    test_true_k4_at_v5_latency_beats_v5()
    test_displayed_4row()
    test_pessimistic_pack4_oss()
    test_pessimistic_dual_rail_still_2000()
    test_cooperating_pipeline_v5_and_pack()
    print("ok")
