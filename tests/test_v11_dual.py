"""V11 per-model dual-rail: only if marker still fires fast."""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SDK_ROOT = ROOT / "data" / "unzipped"
sys.path.insert(0, str(SDK_ROOT))
sys.path.insert(0, str(ROOT))

from aicomp_sdk.attacks import AttackRunConfig

from attack import (
    DUAL_MAX_LAT_S,
    MAX_CHAIN,
    MAX_MSG_CHARS,
    PAYLOAD,
    AttackAlgorithm,
    _pair,
    pick_kind,
)

_URL_RE = re.compile(r"https?://[^\s,;]+")


def test_pick_kind_gemma_like_dual() -> None:
    assert pick_kind(1.0, 0.92, 1.0, 1.42) == "dual"
    assert pick_kind(1.0, 0.92, 1.0, 2.5) == "dual"


def test_pick_kind_oss_like_stays_v5() -> None:
    assert pick_kind(1.0, 3.5, 1.0, 8.0) == "single"
    assert pick_kind(1.0, 3.5, 0.0, 1.4) == "single"
    assert pick_kind(0.0, 0.9, 1.0, 1.4) == "single"


def test_none_env_farms_dual() -> None:
    algo = AttackAlgorithm(config={"hard_n_cap": 20})
    cands = algo.run(None, AttackRunConfig(time_budget_s=5.0, max_tool_hops=8))
    assert len(cands) == 20
    assert MAX_CHAIN == 2
    for c in cands:
        assert len(c.user_messages) == 2
        m1, m2 = c.user_messages
        assert len(m1) <= MAX_MSG_CHARS and len(m2) <= MAX_MSG_CHARS
        assert PAYLOAD in m1
        assert PAYLOAD not in m2
        assert "secret.txt" in m2
        assert "Never emit two tool calls" in m2
        assert "<|" not in m1 and "<|" not in m2
        u1, u2 = _URL_RE.findall(m1)[0], _URL_RE.findall(m2)[0]
        assert u1 != u2


def test_pair_hosts() -> None:
    m1, m2 = _pair(0)
    assert "http://aa.co" in m1
    assert "secret.txt" in m2


if __name__ == "__main__":
    test_pick_kind_gemma_like_dual()
    test_pick_kind_oss_like_stays_v5()
    test_none_env_farms_dual()
    test_pair_hosts()
    print("ok")
