"""Run attack.py locally and print a Kaggle-scaled score.

This is not the 6-prompt smoke test. It runs AttackAlgorithm for a short
budget, scores only traces that actually fired, then applies score_formula.py.

  python scripts/local_score.py --dry-run
  python scripts/local_score.py --budget-s 45 --groq

Groq is a GPT-OSS proxy (free-tier rate limits). Gemma uses the last T4 probe
rates unless you pass --gemma-latency / --gemma-fire-rate.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SDK_ROOT = ROOT / "data" / "unzipped"
SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SDK_ROOT))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SCRIPTS))

from aicomp_sdk.attacks import AttackRunConfig
from aicomp_sdk.core.predicates import eval_predicates
from aicomp_sdk.evaluation.ops import build_attack_env
from aicomp_sdk.guardrails.optimal import Guardrail as OptimalGuardrail
from aicomp_sdk.scoring import score_attack_raw

from attack import AttackAlgorithm
from probe_lib import finding_from_trace
from score_formula import (
    GEMMA_PROBE_FIRE_RATE,
    GEMMA_PROBE_LATENCY_S,
    GEMMA_PROBE_RAW_PER,
    extrapolate,
    format_report,
)

FIXTURES = SDK_ROOT / "aicomp_sdk" / "fixtures"


class RecordingEnv:
    """Forwards sandbox calls and keeps traces that triggered predicates."""

    def __init__(self, inner: Any) -> None:
        self._inner = inner
        self.traces: list[dict[str, Any]] = []
        self.latencies: list[float] = []

    def interact(self, *args: Any, **kwargs: Any) -> Any:
        t0 = time.monotonic()
        result = self._inner.interact(*args, **kwargs)
        self.latencies.append(time.monotonic() - t0)
        try:
            trace = self._inner.export_trace_dict()
        except Exception:
            return result
        if eval_predicates(trace):
            self.traces.append(trace)
        return result

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


def _measure(env: Any, budget_s: float) -> dict[str, float]:
    rec = RecordingEnv(env)
    algo = AttackAlgorithm()
    t0 = time.monotonic()
    candidates = algo.run(rec, AttackRunConfig(time_budget_s=budget_s, max_tool_hops=8))
    elapsed = time.monotonic() - t0
    findings = [finding_from_trace(tr) for tr in rec.traces]
    raw = score_attack_raw(findings) if findings else 0.0
    mean_lat = sum(rec.latencies) / len(rec.latencies) if rec.latencies else elapsed
    return extrapolate(
        n_live=len(findings),
        raw_live=raw,
        elapsed_s=max(elapsed, 1e-6),
        mean_latency_s=mean_lat,
    ) | {
        "n_returned": float(len(candidates)),
        "n_interacts": float(len(rec.latencies)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--budget-s", type=float, default=45.0)
    parser.add_argument("--groq", action="store_true", help="Run attack.py against Groq GPT-OSS.")
    parser.add_argument("--dry-run", action="store_true", help="Formula only, using last probe rates.")
    parser.add_argument("--gemma-latency", type=float, default=GEMMA_PROBE_LATENCY_S)
    parser.add_argument("--gemma-fire-rate", type=float, default=GEMMA_PROBE_FIRE_RATE)
    parser.add_argument("--gemma-raw-per", type=float, default=GEMMA_PROBE_RAW_PER)
    parser.add_argument("--json-out", type=str, default="")
    args = parser.parse_args()

    gemma = extrapolate(
        n_live=4,
        raw_live=4.0 * args.gemma_raw_per,
        elapsed_s=max(4.0 * args.gemma_latency / max(args.gemma_fire_rate, 1e-9), 1e-6),
        mean_latency_s=args.gemma_latency,
    )

    gpt_oss = None
    if args.groq:
        from groq_agent import GroqChatAgent, require_groq_key

        require_groq_key()
        agent = GroqChatAgent()
        env = build_attack_env(
            seed=123,
            fixtures_dir=FIXTURES,
            agent_factory=lambda: agent,
            guardrail=OptimalGuardrail(),
            env_selection="sandbox",
            max_tool_hops=8,
        )
        print(f"running attack.py vs Groq for {args.budget_s:.0f}s (keep this short on free tier)")
        gpt_oss = _measure(env, args.budget_s)
    elif args.dry_run:
        gpt_oss = extrapolate(
            n_live=4,
            raw_live=88.0,
            elapsed_s=52.57,
            mean_latency_s=52.57 / 6.0,
        )
    else:
        print("Pass --groq to measure GPT-OSS, or --dry-run for probe-rate formula only.")
        print()

    note = (
        "Not the hidden Kaggle rerun. Groq ≠ GGUF Harmony. "
        "Gemma line uses T4 probe fire-rate unless you override. "
        "Padding is applied in V5 only after live fires, with unique hosts; "
        "blind (zero-fire) generate farms 2000 singles instead of a 300 fallback."
    )
    print(format_report(gpt_oss=gpt_oss, gemma=gemma, note=note))

    if args.json_out:
        payload = {"gpt_oss": gpt_oss, "gemma": gemma}
        Path(args.json_out).write_text(json.dumps(payload, indent=2))
        print(f"\nwrote {args.json_out}")


if __name__ == "__main__":
    main()
