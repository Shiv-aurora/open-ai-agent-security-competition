"""Unit tests for V9 dual-rail classification used by the kaggle-like bench."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "data" / "unzipped"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

from attack import PAYLOAD, _pair
from probe_lib import _burst_msg, _msg, classify_kind, packed_messages, v5_prompts


def test_classify_kind_roundtrip() -> None:
    assert classify_kind(_msg(0)) == "single"
    assert classify_kind(_burst_msg(4, 2)) == "burst2"
    m1, m2 = _pair(0)
    assert classify_kind(m1) == "single"
    assert classify_kind(m2) == "secret"


def test_v9_pair_is_marker_then_secret() -> None:
    m1, m2 = _pair(3)
    assert PAYLOAD in m1
    assert PAYLOAD not in m2
    assert "secret.txt" in m2


def test_v5_prompts_cover_single_and_burst() -> None:
    labels = [label for label, _ in v5_prompts()]
    assert labels == ["single", "burst2"]
    for label, msg in v5_prompts():
        assert classify_kind(msg) == label


def test_packed_candidate_is_m_singles() -> None:
    msgs = packed_messages(3, 7)
    assert len(msgs) == 3
    assert all(classify_kind(m) == "single" for m in msgs)


if __name__ == "__main__":
    test_classify_kind_roundtrip()
    test_v9_pair_is_marker_then_secret()
    test_v5_prompts_cover_single_and_burst()
    test_packed_candidate_is_m_singles()
    print("ok")
