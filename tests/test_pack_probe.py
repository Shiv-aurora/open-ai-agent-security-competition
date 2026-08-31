"""V8 stack-replay helpers: unique hosts, miss-as-zero scale, ship gate."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "data" / "unzipped"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

from kaggle_like_bench import _stack_ship_decision, summarize_stack_family
from probe_lib import packed_messages, packed_urls


def test_packed_messages_are_unique_host_singles() -> None:
    msgs = packed_messages(4, 10)
    urls = packed_urls(4, 10)
    assert len(msgs) == 4
    assert len(set(msgs)) == 4
    assert len(set(urls)) == 4
    for msg, url in zip(msgs, urls):
        assert url in msg
        assert "SECRET_MARKER" in msg
        assert msg.count("http.post") == 1


def test_summarize_counts_misses_as_zero_raw() -> None:
    result = {
        "replayed": 2,
        "fires": 1,
        "raw": 34.0,
        "timed_out": False,
        "elapsed_s": 4.0,
        "rows": [
            {"posts": 2, "latency_s": 1.8, "fired": True},
            {"posts": 0, "latency_s": 0.9, "fired": False},
        ],
    }
    fam = summarize_stack_family("h2", result)
    assert fam["mean_posts"] == 1.0
    assert fam["raw_per_replayed"] == 17.0
    assert fam["stacks"] is False
    assert fam["n_messages"] == 1
    assert fam["n_kaggle"] == 2000.0  # 8750 / 1.35 > 2000


def test_ship_prefers_highest_stack_forecast() -> None:
    def fam(label: str, posts: float, rate: float, norm: float) -> dict:
        return {
            "label": label,
            "mean_posts": posts,
            "fire_rate": rate,
            "norm_kaggle": norm,
            "stacks": posts >= 1.8,
        }

    families = {
        "single": fam("single", 1.0, 1.0, 180),
        "h2": fam("h2", 2.0, 1.0, 308),
        "h4": fam("h4", 2.0, 1.0, 280),
        "c3": fam("c3", 2.9, 1.0, 500),
    }
    d = _stack_ship_decision(families)
    assert d["ship"] is True
    assert d["mode"] == "stack"
    assert d["winner"] == "c3"


def test_ship_harmony2_when_only_h2_stacks() -> None:
    def fam(label: str, posts: float) -> dict:
        return {
            "label": label,
            "mean_posts": posts,
            "fire_rate": 1.0,
            "norm_kaggle": 308 if posts >= 1.8 else 180,
            "stacks": posts >= 1.8,
        }

    families = {
        "single": fam("single", 1.0),
        "h2": fam("h2", 1.8),
        "h4": fam("h4", 1.0),
        "c2": fam("c2", 1.0),
        "c3": fam("c3", 1.0),
        "c4": fam("c4", 1.0),
    }
    d = _stack_ship_decision(families)
    assert d["ship"] is True
    assert d["winner"] == "h2"


def test_no_ship_when_nothing_stacks() -> None:
    families = {
        "single": {
            "label": "single",
            "mean_posts": 1.0,
            "fire_rate": 1.0,
            "norm_kaggle": 180,
            "stacks": False,
        },
        "h2": {
            "label": "h2",
            "mean_posts": 1.0,
            "fire_rate": 1.0,
            "norm_kaggle": 180,
            "stacks": False,
        },
    }
    d = _stack_ship_decision(families)
    assert d["ship"] is False
    assert d["winner"] is None


if __name__ == "__main__":
    test_packed_messages_are_unique_host_singles()
    test_summarize_counts_misses_as_zero_raw()
    test_ship_prefers_highest_stack_forecast()
    test_ship_harmony2_when_only_h2_stacks()
    test_no_ship_when_nothing_stacks()
    print("ok")
