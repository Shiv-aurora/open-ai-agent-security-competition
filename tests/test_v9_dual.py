"""V9 dual-rail helpers still exist; V10 farm is one-message seq-2."""

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
    MAX_CHAIN,
    PAYLOAD,
    AttackAlgorithm,
    _pair,
    _url,
)

_URL_RE = re.compile(r"https?://[^\s,;]+")


def test_farm_is_dual_rail() -> None:
    algo = AttackAlgorithm(config={"hard_n_cap": 5})
    cands = algo.run(None, AttackRunConfig(time_budget_s=5.0, max_tool_hops=8))
    assert MAX_CHAIN == 2
    assert all(len(c.user_messages) == 2 for c in cands)


def test_pair_hosts_are_adjacent() -> None:
    m1, m2 = _pair(0)
    assert _url(0) in m1
    assert _url(1) in m2
    assert PAYLOAD in m1
    assert PAYLOAD not in m2
    assert "secret.txt" in m2
    assert _URL_RE.findall(m1)[0] != _URL_RE.findall(m2)[0]


if __name__ == "__main__":
    test_farm_is_dual_rail()
    test_pair_hosts_are_adjacent()
    print("ok")
