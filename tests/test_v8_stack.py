"""V10 is one-message; V8 live-argmax stack tests retired."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SDK_ROOT = ROOT / "data" / "unzipped"
sys.path.insert(0, str(SDK_ROOT))
sys.path.insert(0, str(ROOT))

from aicomp_sdk.attacks import AttackRunConfig

from attack import MAX_CHAIN, AttackAlgorithm


def test_v11_allows_two_messages() -> None:
    algo = AttackAlgorithm(config={"hard_n_cap": 5})
    cands = algo.run(None, AttackRunConfig(time_budget_s=5.0, max_tool_hops=8))
    assert MAX_CHAIN == 2
    assert all(len(c.user_messages) == 2 for c in cands)


if __name__ == "__main__":
    test_v11_allows_two_messages()
    print("ok")
