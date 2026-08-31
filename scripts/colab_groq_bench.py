#!/usr/bin/env python3
"""Run the reliable replay scorer on Colab (Groq) or any machine with GROQ_API_KEY.

This does **not** git-clone GitHub. The clone was missing the scorer and would
silently run the old generate+replay bench. Upload or Drive-mount this repo,
then:

  GROQ_API_KEY=gsk_... python scripts/colab_groq_bench.py

Or call the scorer directly:

  python scripts/replay_scorer.py --groq --env gym --n 20 --replay-s 400 --json-out groq_score.json

Groq Chat Completions ≠ Kaggle GPT-OSS GGUF Harmony. Use it as a second model
vote, not as the Gemma T4 number. Ship decisions for Gemma still need the
Kaggle GPU probe (`kaggle kernels push -p kaggle_probe`).
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def find_root() -> Path:
    env = os.environ.get("JED_ROOT", "").strip()
    candidates = []
    if env:
        candidates.append(Path(env))
    candidates.extend(
        [
            Path(__file__).resolve().parent.parent,
            Path.cwd(),
            Path("/content"),
            Path("/content/Untitled"),
            Path("/content/jed-bench"),
        ]
    )
    for root in candidates:
        if (root / "attack.py").is_file() and (root / "scripts" / "replay_scorer.py").is_file():
            return root
    raise SystemExit(
        "Need this repo in the runtime (attack.py + scripts/replay_scorer.py). "
        "Upload the folder or set JED_ROOT. Do not clone GitHub unless that "
        "remote already has replay_scorer.py."
    )


def sh(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.check_call(cmd)


def main() -> None:
    if not os.environ.get("GROQ_API_KEY", "").strip():
        raise SystemExit("GROQ_API_KEY is not set.")
    root = find_root()
    os.chdir(root)
    sys.path.insert(0, str(root / "data" / "unzipped"))
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(root / "scripts"))
    sh([sys.executable, "-m", "pip", "install", "-q", "openai", "gymnasium", "pydantic"])
    out = Path("/content/groq_score.json") if Path("/content").exists() else root / "groq_score.json"
    sh(
        [
            sys.executable,
            "scripts/replay_scorer.py",
            "--groq",
            "--env",
            "gym",
            "--n",
            "20",
            "--replay-s",
            "400",
            "--json-out",
            str(out),
        ]
    )
    print("done", out)


if __name__ == "__main__":
    main()
