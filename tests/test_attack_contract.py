"""Contract smoke test for AttackAlgorithm against the SDK deterministic agent."""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SDK_ROOT = ROOT / "data" / "unzipped"
sys.path.insert(0, str(SDK_ROOT))
sys.path.insert(0, str(ROOT))

from aicomp_sdk.agents import AgentSelection, build_agent_factory
from aicomp_sdk.attacks import AttackCandidate, AttackRunConfig
from aicomp_sdk.evaluation.ops import build_attack_env
from aicomp_sdk.guardrails.optimal import Guardrail as OptimalGuardrail

from attack import MAX_CHAIN, MAX_MSG_CHARS, AttackAlgorithm

FIXTURES = SDK_ROOT / "aicomp_sdk" / "fixtures"
BUDGET_S = 25.0


def test_contract() -> None:
    env = build_attack_env(
        seed=123,
        fixtures_dir=FIXTURES,
        agent_factory=build_agent_factory(AgentSelection.DETERMINISTIC),
        guardrail=OptimalGuardrail(),
        env_selection="sandbox",
        max_tool_hops=8,
    )
    algo = AttackAlgorithm()
    started = time.monotonic()
    candidates = algo.run(env, AttackRunConfig(time_budget_s=BUDGET_S, max_tool_hops=8))
    elapsed = time.monotonic() - started

    assert elapsed < BUDGET_S + 5.0, f"run overran budget: {elapsed:.1f}s"
    assert isinstance(candidates, list) and candidates, "expected a non-empty candidate list"
    assert len(candidates) <= 2000
    for idx, cand in enumerate(candidates):
        assert isinstance(cand, AttackCandidate)
        messages = list(cand.user_messages)
        assert messages, f"candidate {idx} has no messages"
        assert len(messages) <= MAX_CHAIN
        for msg in messages:
            assert isinstance(msg, str) and msg.strip()
            assert len(msg) <= MAX_MSG_CHARS, f"candidate {idx} message too long"

    print(
        f"ok: {len(candidates)} candidates in {elapsed:.2f}s "
        f"(budget {BUDGET_S}s, first chain len={len(candidates[0].user_messages)})"
    )


if __name__ == "__main__":
    test_contract()
