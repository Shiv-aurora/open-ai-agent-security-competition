"""V11 dual-rail unique hosts."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SDK_ROOT = ROOT / "data" / "unzipped"
sys.path.insert(0, str(SDK_ROOT))
sys.path.insert(0, str(ROOT))

from aicomp_sdk.attacks import AttackRunConfig

from attack import PAYLOAD, AttackAlgorithm


def test_forced_pairs_unique_hosts() -> None:
    algo = AttackAlgorithm(config={"hard_n_cap": 20})
    cands = algo.run(None, AttackRunConfig(time_budget_s=5.0, max_tool_hops=8))
    assert len(cands) == 20
    assert all(len(c.user_messages) == 2 for c in cands)
    texts = [c.user_messages[0] for c in cands]
    assert len(set(texts)) == 20
    assert all(PAYLOAD in t and "http://" in t for t in texts)


if __name__ == "__main__":
    test_forced_pairs_unique_hosts()
    print("ok")
