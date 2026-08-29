"""V5 fill/farm contract: keep live fires, farm 2000 if blind, unique hosts."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SDK_ROOT = ROOT / "data" / "unzipped"
sys.path.insert(0, str(SDK_ROOT))
sys.path.insert(0, str(ROOT))

from aicomp_sdk.attacks import AttackRunConfig

from attack import PAYLOAD, AttackAlgorithm, _kind_span, _url


class FakeEnv:
    def __init__(self, posts_for_msg) -> None:
        self._posts_for_msg = posts_for_msg
        self._posts = 0
        self.n_interact = 0
        self.msgs: list[str] = []

    def reset(self) -> None:
        self._posts = 0

    def interact(self, user_message: str, max_tool_hops: int = 8) -> None:
        self.n_interact += 1
        self.msgs.append(user_message)
        self._posts = int(self._posts_for_msg(user_message))

    def export_trace_dict(self) -> dict:
        events = []
        for i in range(self._posts):
            events.append(
                {
                    "ok": True,
                    "name": "http.post",
                    "args": {"url": _url(i), "data": PAYLOAD},
                }
            )
        return {"tool_events": events}


def _run(env, budget: float = 5.0, **cfg) -> list:
    algo = AttackAlgorithm(config={"hard_n_cap": 20, "return_margin_s": 0.05, **cfg})
    return algo.run(env, AttackRunConfig(time_budget_s=budget, max_tool_hops=8))


def test_blind_env_farms_cap_not_300() -> None:
    env = FakeEnv(lambda _msg: 0)
    cands = _run(env, miss_abort=5)
    assert len(cands) == 20
    assert env.n_interact <= 6  # warmup + 4 probes, then farm without more generate
    texts = [c.user_messages[0] for c in cands]
    assert len(set(texts)) == 20
    assert all(PAYLOAD in t for t in texts)
    assert all("http://" in t for t in texts)


def test_keeps_every_live_fire_including_probes() -> None:
    env = FakeEnv(lambda msg: 1)
    cands = _run(env)
    assert len(cands) == 20
    live = env.n_interact
    assert live >= 5
    # Every interact that fired must appear in the returned list (no V4 classify drop).
    assert all(msg in {c.user_messages[0] for c in cands} for msg in env.msgs)


def test_picks_stacking_template_when_it_posts_more() -> None:
    def posts(msg: str) -> int:
        return msg.count("http://")

    env = FakeEnv(posts)
    cands = _run(env, budget=2.0)
    fill_msgs = env.msgs[5:]  # after warmup + 4 probes
    assert fill_msgs, "expected fill interacts after probes"
    # Harmony-4 names 4 unique hosts in one message.
    assert any(m.count("http://") >= 4 for m in fill_msgs)


def test_pad_hosts_do_not_reuse_low_indices() -> None:
    env = FakeEnv(lambda msg: 1 if "aa.co" in msg or "ab.co" in msg else 0)
    # Only the first couple of hosts fire; farm must still use fresh hosts.
    cands = _run(env, budget=1.0, miss_abort=50)
    texts = [c.user_messages[0] for c in cands]
    assert len(texts) == 20
    assert len(set(texts)) == 20


def test_kind_span() -> None:
    assert _kind_span("single") == 1
    assert _kind_span("burst2") == 2
    assert _kind_span("h4") == 4


if __name__ == "__main__":
    test_blind_env_farms_cap_not_300()
    test_keeps_every_live_fire_including_probes()
    test_picks_stacking_template_when_it_posts_more()
    test_pad_hosts_do_not_reuse_low_indices()
    test_kind_span()
    print("ok")
