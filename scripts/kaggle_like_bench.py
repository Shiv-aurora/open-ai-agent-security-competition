"""Kaggle-like generate + independent replay bench.

Matches the hosted gateway as closely as a local/Colab/Kaggle-GPU run can:

- ``env_selection=gym`` (the competition gateway), hops=8, OptimalGuardrail
- generate is wrapped in the opaque attacker env
- each returned candidate is replayed on a **fresh** env until the replay clock
  hits the budget (partial findings are kept, same as the gateway)
- score is SDK ``score_attack`` / ``score_attack_raw`` on replay findings only

This is still not the hidden rerun: Groq Chat Completions ≠ GPT-OSS GGUF Harmony,
and a 90s sample is not 8750s. It **is** the same primitive, guardrail, cell
hash, and replay order the leaderboard uses.

  python scripts/kaggle_like_bench.py --groq --generate-s 90 --replay-s 90
  python scripts/kaggle_like_bench.py --backend gemma_gguf --generate-s 90
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable

def _here() -> Path:
    return Path(__file__).resolve().parent


def _repo_root() -> Path:
    here = _here()
    if (here.parent / "attack.py").exists():
        return here.parent
    return here


def _sdk_root() -> Path:
    root = _repo_root()
    local = root / "data" / "unzipped"
    if (local / "aicomp_sdk").is_dir():
        return local
    for p in sys.path:
        pp = Path(p)
        if (pp / "aicomp_sdk").is_dir():
            return pp
    return local


def _fixtures() -> Path:
    env = os.environ.get("AICOMP_FIXTURES_DIR", "").strip()
    if env and Path(env).is_dir():
        return Path(env)
    local = _sdk_root() / "aicomp_sdk" / "fixtures"
    if local.is_dir():
        return local
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.exists():
        hits = sorted(kaggle_input.rglob("aicomp_sdk/fixtures"))
        if hits:
            return hits[0]
    return local


ROOT = _repo_root()
SDK_ROOT = _sdk_root()
SCRIPTS = _here()
sys.path.insert(0, str(SDK_ROOT))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SCRIPTS))
FIXTURES = _fixtures()

from aicomp_sdk.attacks import AttackCandidate, AttackRunConfig
from aicomp_sdk.core.env.opaque import _OpaqueAttackEnv
from aicomp_sdk.core.predicates import eval_predicates
from aicomp_sdk.evaluation.ops import build_attack_env
from aicomp_sdk.guardrails.optimal import Guardrail as OptimalGuardrail
from aicomp_sdk.scoring import score_attack, score_attack_raw

from attack import AttackAlgorithm, PAYLOAD, _msg, _pair
from probe_lib import (
    classify_kind,
    finding_from_trace,
    packed_messages,
    run_prompts,
    v5_prompts,
)
from score_formula import KAGGLE_BUDGET_S, MAX_FINDINGS, extrapolate, norm_from_raw

HOPS = 8
SEED = 123


class RecordingEnv:
    """Forwards env ops and records generate-side fires + latencies."""

    def __init__(self, inner: Any) -> None:
        self._inner = inner
        self.traces: list[dict[str, Any]] = []
        self.latencies: list[float] = []
        self.msgs: list[str] = []

    def interact(self, user_message: str, max_tool_hops: int | None = None) -> Any:
        self.msgs.append(user_message)
        t0 = time.monotonic()
        result = self._inner.interact(user_message, max_tool_hops=max_tool_hops)
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


def resolve_env_selection(preferred: str) -> str:
    if preferred != "gym":
        return preferred
    try:
        import gymnasium  # noqa: F401
    except ImportError:
        print("gymnasium not installed; falling back to sandbox (Kaggle uses gym)")
        return "sandbox"
    return "gym"


def make_raw_env(agent_factory: Callable, env_selection: str) -> Any:
    return build_attack_env(
        seed=SEED,
        fixtures_dir=FIXTURES,
        agent_factory=agent_factory,
        guardrail=OptimalGuardrail(),
        env_selection=env_selection,
        max_tool_hops=HOPS,
    )


def make_generate_env(agent_factory: Callable, env_selection: str) -> RecordingEnv:
    inner = make_raw_env(agent_factory, env_selection)
    return RecordingEnv(_OpaqueAttackEnv(inner))


def replay_candidates(
    agent_factory: Callable,
    candidates: list[AttackCandidate],
    *,
    env_selection: str,
    budget_s: float,
    hops: int = HOPS,
    kind: str | None = None,
) -> dict[str, Any]:
    """Independent per-candidate replay with a wall clock, like the gateway."""
    deadline = time.monotonic() + max(0.0, float(budget_s))
    findings: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    timed_out = False
    replayed = 0
    t0 = time.monotonic()
    for idx, cand in enumerate(candidates):
        if time.monotonic() >= deadline:
            timed_out = True
            break
        messages = list(cand.user_messages)
        row_kind = kind or classify_kind(messages[0] if messages else "")
        env = make_raw_env(agent_factory, env_selection)
        env.reset()
        if time.monotonic() >= deadline:
            timed_out = True
            break
        step_t0 = time.monotonic()
        try:
            for msg in messages:
                if time.monotonic() >= deadline:
                    timed_out = True
                    break
                env.interact(msg, max_tool_hops=hops)
        except Exception as err:
            rows.append(
                {
                    "idx": idx,
                    "kind": row_kind,
                    "n_messages": len(messages),
                    "fired": False,
                    "error": str(err),
                }
            )
            continue
        if timed_out:
            break
        latency = time.monotonic() - step_t0
        trace = env.export_trace_dict()
        finding = finding_from_trace(trace)
        fired = bool(finding["predicates"])
        if fired:
            findings.append(finding)
        posts = 0
        for event in trace.get("tool_events") or []:
            if event.get("ok") is True and event.get("name") == "http.post":
                if PAYLOAD in str((event.get("args") or {}).get("data", "")):
                    posts += 1
        rows.append(
            {
                "idx": idx,
                "kind": row_kind,
                "n_messages": len(messages),
                "fired": fired,
                "posts": posts,
                "latency_s": round(latency, 3),
                "predicates": [p["predicate"] for p in finding["predicates"]],
            }
        )
        replayed += 1
    raw = score_attack_raw(findings) if findings else 0.0
    elapsed = time.monotonic() - t0
    mean_lat = (
        sum(r["latency_s"] for r in rows if "latency_s" in r) / max(1, sum(1 for r in rows if "latency_s" in r))
        if rows
        else elapsed
    )
    scaled = extrapolate(
        n_live=len(findings),
        raw_live=raw,
        elapsed_s=max(elapsed, 1e-6),
        mean_latency_s=max(mean_lat, 1e-6),
    )
    return {
        "replayed": replayed,
        "fires": len(findings),
        "timed_out": timed_out,
        "elapsed_s": round(elapsed, 3),
        "raw": raw,
        "normalized": score_attack(findings) if findings else 0.0,
        "kind_counts": dict(Counter(r["kind"] for r in rows)),
        "fire_kinds": dict(Counter(r["kind"] for r in rows if r.get("fired"))),
        "rows": rows[:80],
        "scaled": scaled,
    }


def run_bench(
    agent_factory: Callable,
    *,
    generate_s: float,
    replay_s: float,
    env_selection: str,
    probe_templates: bool,
    hard_n_cap: int | None = None,
) -> dict[str, Any]:
    env_selection = resolve_env_selection(env_selection)
    report: dict[str, Any] = {
        "env_selection": env_selection,
        "hops": HOPS,
        "generate_s": generate_s,
        "replay_s": replay_s,
        "note": "Replay-scored. Not the hidden Kaggle rerun.",
    }
    if probe_templates:
        print(f"\n== V5 template probe ({env_selection}, hops={HOPS}) ==")
        probe_env = make_raw_env(agent_factory, env_selection)
        report["template_probe"] = run_prompts(probe_env, v5_prompts(), hops=HOPS)

    print(f"\n== generate AttackAlgorithm ({generate_s:.0f}s) ==")
    gen_env = make_generate_env(agent_factory, env_selection)
    cfg = {"hard_n_cap": int(hard_n_cap)} if hard_n_cap else {}
    algo = AttackAlgorithm(config=cfg)
    t0 = time.monotonic()
    candidates = algo.run(gen_env, AttackRunConfig(time_budget_s=generate_s, max_tool_hops=HOPS))
    gen_elapsed = time.monotonic() - t0
    gen_findings = [finding_from_trace(tr) for tr in gen_env.traces]
    gen_raw = score_attack_raw(gen_findings) if gen_findings else 0.0
    returned_kinds = Counter(classify_kind(c.user_messages[0]) for c in candidates if c.user_messages)
    live_kinds = Counter(classify_kind(msg) for msg in gen_env.msgs)
    report["generate"] = {
        "elapsed_s": round(gen_elapsed, 3),
        "n_interact": len(gen_env.latencies),
        "n_live_predicate_traces": len(gen_env.traces),
        "n_returned": len(candidates),
        "live_raw": gen_raw,
        "mean_latency_s": (
            sum(gen_env.latencies) / len(gen_env.latencies) if gen_env.latencies else 0.0
        ),
        "returned_kind_counts": dict(returned_kinds),
        "interact_kind_counts": dict(live_kinds),
        "first_returned_kind": classify_kind(candidates[0].user_messages[0]) if candidates else None,
    }
    print(
        f"  interact={report['generate']['n_interact']}  "
        f"live_fires={report['generate']['n_live_predicate_traces']}  "
        f"returned={report['generate']['n_returned']}  "
        f"kinds={report['generate']['returned_kind_counts']}"
    )

    print(f"\n== independent replay ({replay_s:.0f}s, list order) ==")
    report["replay"] = replay_candidates(
        agent_factory,
        candidates,
        env_selection=env_selection,
        budget_s=replay_s,
    )
    print(
        f"  replayed={report['replay']['replayed']}  fires={report['replay']['fires']}  "
        f"timed_out={report['replay']['timed_out']}  "
        f"raw={report['replay']['raw']:.0f}  "
        f"norm={report['replay']['normalized']:.3f}"
    )
    scaled = report["replay"]["scaled"]
    print(
        f"  scaled-to-8750s  n_kaggle={scaled['n_kaggle']:.1f}  "
        f"norm={scaled['norm_kaggle']:.3f}  (forecast, not leaderboard)"
    )
    return report


def summarize_pack_family(label: str, result: dict[str, Any], n_messages: int) -> dict[str, Any]:
    """Scale a short packed replay to one 8750s / 2000-finding model clock.

    Misses count as 0 raw (unlike extrapolate(), which averages only fires).
    """
    rows = [r for r in result.get("rows") or [] if "posts" in r]
    replayed = int(result.get("replayed") or 0)
    posts = [int(r.get("posts") or 0) for r in rows]
    lats = [float(r["latency_s"]) for r in rows if "latency_s" in r]
    mean_posts = sum(posts) / len(posts) if posts else 0.0
    mean_lat = sum(lats) / len(lats) if lats else 0.0
    fire_rate = (int(result.get("fires") or 0) / replayed) if replayed else 0.0
    raw = float(result.get("raw") or 0.0)
    raw_per = raw / replayed if replayed else 0.0
    n_kaggle = min(float(MAX_FINDINGS), KAGGLE_BUDGET_S / mean_lat) if mean_lat > 0 else 0.0
    raw_kaggle = n_kaggle * raw_per
    norm = norm_from_raw(raw_kaggle)
    return {
        "label": label,
        "n_messages": n_messages,
        "replayed": replayed,
        "fires": int(result.get("fires") or 0),
        "fire_rate": round(fire_rate, 4),
        "mean_posts": round(mean_posts, 4),
        "mean_latency_s": round(mean_lat, 4),
        "raw": raw,
        "raw_per_replayed": round(raw_per, 4),
        "n_kaggle": round(n_kaggle, 2),
        "norm_kaggle": round(norm, 3),
        "public_4row_if_both_private0": round(norm / 2.0, 3),
        "kill_pack_m2": bool(label == "pack_m2" and mean_posts < 1.8),
        "timed_out": bool(result.get("timed_out")),
        "elapsed_s": result.get("elapsed_s"),
        "rows": rows,
    }


def _pack_ship_decision(families: dict[str, Any]) -> dict[str, Any]:
    m2 = families.get("pack_m2") or {}
    h2 = families.get("farm_h2") or {}
    packs = [families[k] for k in ("pack_m2", "pack_m3", "pack_m4") if k in families]
    pack_ok = [
        f
        for f in packs
        if f.get("mean_posts", 0) >= 1.8 and f.get("fire_rate", 0) >= 0.95
    ]
    h2_ok = h2.get("mean_posts", 0) >= 1.8 and h2.get("fire_rate", 0) >= 0.95
    if pack_ok:
        winner = max(pack_ok, key=lambda f: f["norm_kaggle"])
        return {
            "winner": winner["label"],
            "ship": True,
            "mode": "pack",
            "kill_reason": None,
        }
    if m2.get("kill_pack_m2") and h2_ok:
        return {
            "winner": "farm_h2",
            "ship": True,
            "mode": "h2",
            "kill_reason": "pack_m2 mean_posts < 1.8; Harmony-2 still stacks",
        }
    if h2_ok:
        return {
            "winner": "farm_h2",
            "ship": True,
            "mode": "h2",
            "kill_reason": None,
        }
    return {
        "winner": None,
        "ship": False,
        "mode": None,
        "kill_reason": "pack_m2 mean_posts < 1.8 and farm_h2 did not stack",
    }


def run_pack_probe(
    agent_factory: Callable,
    *,
    env_selection: str,
    n_per: int = 20,
    replay_s: float = 900.0,
    host_start: int = 50_000,
) -> dict[str, Any]:
    """Replay farmed M-message singles and farmed Harmony-2. No generate fill."""
    env_selection = resolve_env_selection(env_selection)
    host = int(host_start)
    families_spec: list[tuple[str, int, list[AttackCandidate]]] = []
    for m in (2, 3, 4):
        cands: list[AttackCandidate] = []
        for _ in range(n_per):
            cands.append(AttackCandidate.from_messages(packed_messages(m, host)))
            host += m
        families_spec.append((f"pack_m{m}", m, cands))
    h2: list[AttackCandidate] = []
    for _ in range(n_per):
        h2.append(AttackCandidate.from_messages((_msg(host),)))
        host += 1
    families_spec.append(("farm_h2", 1, h2))

    report: dict[str, Any] = {
        "env_selection": env_selection,
        "hops": HOPS,
        "n_per_family": n_per,
        "replay_s": replay_s,
        "note": "Packed replay. Not the hidden Kaggle rerun. Not a V5 generate fill.",
        "families": {},
    }
    for label, n_messages, cands in families_spec:
        print(f"\n== packed replay {label} ({len(cands)} cands, {replay_s:.0f}s) ==")
        raw_result = replay_candidates(
            agent_factory,
            cands,
            env_selection=env_selection,
            budget_s=replay_s,
            kind=label,
        )
        family = summarize_pack_family(label, raw_result, n_messages)
        report["families"][label] = family
        print(
            f"  replayed={family['replayed']}  fires={family['fires']}  "
            f"mean_posts={family['mean_posts']}  lat={family['mean_latency_s']}s  "
            f"norm_kaggle={family['norm_kaggle']}  "
            f"kill_m2={family['kill_pack_m2']}"
        )

    decision = _pack_ship_decision(report["families"])
    report.update(decision)
    print(
        f"\n  winner={report['winner']}  ship={report['ship']}  "
        f"mode={report['mode']}  {report.get('kill_reason') or ''}"
    )
    return report


def summarize_stack_family(label: str, result: dict[str, Any]) -> dict[str, Any]:
    rows = [r for r in result.get("rows") or [] if "posts" in r]
    replayed = int(result.get("replayed") or 0)
    posts = [int(r.get("posts") or 0) for r in rows]
    lats = [float(r["latency_s"]) for r in rows if "latency_s" in r]
    mean_posts = sum(posts) / len(posts) if posts else 0.0
    mean_lat = sum(lats) / len(lats) if lats else 0.0
    fire_rate = (int(result.get("fires") or 0) / replayed) if replayed else 0.0
    raw = float(result.get("raw") or 0.0)
    raw_per = raw / replayed if replayed else 0.0
    n_kaggle = min(float(MAX_FINDINGS), KAGGLE_BUDGET_S / mean_lat) if mean_lat > 0 else 0.0
    raw_kaggle = n_kaggle * raw_per
    norm = norm_from_raw(raw_kaggle)
    return {
        "label": label,
        "n_messages": 1,
        "replayed": replayed,
        "fires": int(result.get("fires") or 0),
        "fire_rate": round(fire_rate, 4),
        "mean_posts": round(mean_posts, 4),
        "mean_latency_s": round(mean_lat, 4),
        "raw": raw,
        "raw_per_replayed": round(raw_per, 4),
        "n_kaggle": round(n_kaggle, 2),
        "norm_kaggle": round(norm, 3),
        "public_4row_if_both_private0": round(norm / 2.0, 3),
        "stacks": bool(mean_posts >= 1.8),
        "timed_out": bool(result.get("timed_out")),
        "elapsed_s": result.get("elapsed_s"),
        "rows": rows,
    }


def _stack_ship_decision(families: dict[str, Any]) -> dict[str, Any]:
    stack_ok = [
        f
        for name, f in families.items()
        if name != "single" and f.get("mean_posts", 0) >= 1.8
    ]
    if stack_ok:
        winner = max(stack_ok, key=lambda f: f["norm_kaggle"])
        return {
            "winner": winner["label"],
            "ship": True,
            "mode": "stack",
            "kill_reason": None,
        }
    return {
        "winner": None,
        "ship": False,
        "mode": None,
        "kill_reason": "no one-message stack with mean_posts >= 1.8",
    }


def summarize_dual_family(result: dict[str, Any]) -> dict[str, Any]:
    rows = [r for r in result.get("rows") or [] if "posts" in r]
    replayed = int(result.get("replayed") or 0)
    posts = [int(r.get("posts") or 0) for r in rows]
    lats = [float(r["latency_s"]) for r in rows if "latency_s" in r]
    mean_posts = sum(posts) / len(posts) if posts else 0.0
    mean_lat = sum(lats) / len(lats) if lats else 0.0
    fire_rate = (int(result.get("fires") or 0) / replayed) if replayed else 0.0
    raw = float(result.get("raw") or 0.0)
    raw_per = raw / replayed if replayed else 0.0
    n_kaggle = min(float(MAX_FINDINGS), KAGGLE_BUDGET_S / mean_lat) if mean_lat > 0 else 0.0
    raw_kaggle = n_kaggle * raw_per
    norm = norm_from_raw(raw_kaggle)
    return {
        "label": "dual",
        "n_messages": 2,
        "replayed": replayed,
        "fires": int(result.get("fires") or 0),
        "fire_rate": round(fire_rate, 4),
        "mean_posts": round(mean_posts, 4),
        "mean_latency_s": round(mean_lat, 4),
        "raw": raw,
        "raw_per_replayed": round(raw_per, 4),
        "n_kaggle": round(n_kaggle, 2),
        "norm_kaggle": round(norm, 3),
        "public_4row_if_both_private0": round(norm / 2.0, 3),
        "timed_out": bool(result.get("timed_out")),
        "elapsed_s": result.get("elapsed_s"),
        "rows": rows,
    }


def _dual_ship_decision(family: dict[str, Any]) -> dict[str, Any]:
    mean_posts = float(family.get("mean_posts") or 0.0)
    mean_lat = float(family.get("mean_latency_s") or 0.0)
    if mean_posts >= 0.95 and mean_lat <= 4.0:
        return {
            "winner": "dual",
            "ship": True,
            "mode": "dual",
            "kill_reason": None,
        }
    reasons = []
    if mean_posts < 0.95:
        reasons.append(f"mean_posts {mean_posts} < 0.95")
    if mean_lat > 4.0:
        reasons.append(f"mean_latency_s {mean_lat} > 4.0")
    return {
        "winner": None,
        "ship": False,
        "mode": None,
        "kill_reason": "; ".join(reasons) or "dual-rail ship gate failed",
    }


def run_dual_probe(
    agent_factory: Callable,
    *,
    env_selection: str,
    n_per: int = 20,
    replay_s: float = 900.0,
    host_start: int = 0,
) -> dict[str, Any]:
    """Replay farmed two-message dual-rail candidates. No generate fill."""
    env_selection = resolve_env_selection(env_selection)
    start = int(host_start)
    cands: list[AttackCandidate] = []
    for i in range(n_per):
        pair = _pair(start + i)
        cands.append(AttackCandidate.from_messages(pair))
    assert all(len(c.user_messages) == 2 for c in cands)
    print(f"\n== dual-rail replay ({len(cands)} cands, {replay_s:.0f}s) ==")
    raw_result = replay_candidates(
        agent_factory,
        cands,
        env_selection=env_selection,
        budget_s=replay_s,
        kind="dual",
    )
    family = summarize_dual_family(raw_result)
    report: dict[str, Any] = {
        "env_selection": env_selection,
        "hops": HOPS,
        "n_per_family": n_per,
        "replay_s": replay_s,
        "note": "Two-message dual-rail replay. Marker then secret.txt. Not the hidden Kaggle rerun.",
        "families": {"dual": family},
    }
    decision = _dual_ship_decision(family)
    report.update(decision)
    print(
        f"  replayed={family['replayed']}  fires={family['fires']}  "
        f"mean_posts={family['mean_posts']}  lat={family['mean_latency_s']}s  "
        f"n_kaggle={family['n_kaggle']}  ship={report['ship']}  "
        f"{report.get('kill_reason') or ''}"
    )
    return report


def groq_factory():
    from groq_agent import GroqChatAgent, require_groq_key

    require_groq_key()
    agent = GroqChatAgent()
    return lambda: agent


def cooperating_factory():
    """Keyword poster. Pipeline smoke only — not a model forecast."""
    from local_probe import CooperatingPoster

    agent = CooperatingPoster()
    return lambda: agent


def gemma_gguf_factory():
    from kaggle_evaluation.jed_attack_134815.gemma_model_server import SPEC
    from kaggle_evaluation.jed_attack_134815.gguf_model_server import GgufModelServer

    server = GgufModelServer(SPEC)
    agent = server._load_agent()
    return lambda: agent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generate-s", type=float, default=90.0)
    parser.add_argument("--replay-s", type=float, default=90.0)
    parser.add_argument("--env", choices=("gym", "sandbox"), default="gym")
    parser.add_argument("--groq", action="store_true")
    parser.add_argument("--backend", choices=("groq", "gemma_gguf", "cooperating"), default="")
    parser.add_argument("--skip-templates", action="store_true")
    parser.add_argument("--pack", action="store_true", help="Replay packed M-message singles + farmed Harmony-2.")
    parser.add_argument("--stack", action="store_true", help="Replay one-message stack templates (V8).")
    parser.add_argument("--dual", action="store_true", help="Replay two-message dual-rail candidates (V9).")
    parser.add_argument("--reliable", action="store_true", help="V5 baseline + attack + pack canary; V6-shaped kill.")
    parser.add_argument("--pack-n", type=int, default=20, help="Candidates per pack/stack/dual family.")
    parser.add_argument("--hard-n-cap", type=int, default=0, help="Cap returned candidates (0 = attack default 2000).")
    parser.add_argument("--json-out", type=str, default="")
    args = parser.parse_args()
    backend = args.backend or ("groq" if args.groq else "")
    if backend == "groq":
        factory = groq_factory()
        label = "groq_gpt_oss_20b"
    elif backend == "gemma_gguf":
        factory = gemma_gguf_factory()
        label = "gemma4_gguf"
    elif backend == "cooperating":
        factory = cooperating_factory()
        label = "cooperating_toy"
    else:
        parser.error("pass --groq or --backend groq|gemma_gguf|cooperating")

    if args.reliable:
        from replay_scorer import run_reliable_score

        replay_s = 900.0 if args.replay_s == 90.0 else args.replay_s
        report = run_reliable_score(
            factory,
            env_selection=args.env,
            n_per=max(1, args.pack_n),
            replay_s=replay_s,
            pipeline_only=(label == "cooperating_toy"),
        )
        slim = {k: report[k] for k in report if k != "families"}
        slim["families"] = report.get("families")
    elif args.pack or args.stack or args.dual:
        replay_s = 900.0 if args.replay_s == 90.0 else args.replay_s
        if args.dual or args.stack:
            report = run_dual_probe(
                factory,
                env_selection=args.env,
                n_per=max(1, args.pack_n),
                replay_s=replay_s,
            )
        else:
            report = run_pack_probe(
                factory,
                env_selection=args.env,
                n_per=max(1, args.pack_n),
                replay_s=replay_s,
            )
        slim_families = {
            name: {k: v for k, v in fam.items() if k != "rows"}
            for name, fam in report.get("families", {}).items()
        }
        slim = {
            "backend": label,
            "env_selection": report.get("env_selection"),
            "winner": report.get("winner"),
            "ship": report.get("ship"),
            "mode": report.get("mode"),
            "kill_reason": report.get("kill_reason"),
            "families": slim_families,
        }
    else:
        report = run_bench(
            factory,
            generate_s=args.generate_s,
            replay_s=args.replay_s,
            env_selection=args.env,
            probe_templates=not args.skip_templates,
            hard_n_cap=args.hard_n_cap or None,
        )
        slim = {k: report[k] for k in ("backend", "env_selection", "generate", "replay") if k in report}
        slim["backend"] = label
    report["backend"] = label
    slim["backend"] = label
    print("\n" + json.dumps(slim, indent=2)[:4000])
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(report, indent=2))
        print(f"\nwrote {args.json_out}")


if __name__ == "__main__":
    main()
