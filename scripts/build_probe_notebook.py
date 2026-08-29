"""Build a GPU probe notebook that loads competition Gemma and optional Groq GPT-OSS."""

from __future__ import annotations

import base64
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "kaggle_probe"
NOTEBOOK_PATH = OUT_DIR / "probe.ipynb"


def _b64(path: Path) -> str:
    raw = base64.b64encode(path.read_bytes()).decode("ascii")
    return "\n".join(raw[i : i + 80] for i in range(0, len(raw), 80))


def _code(lines: list[str]) -> dict:
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": [line if line.endswith("\n") else line + "\n" for line in lines],
    }


def _md(lines: list[str]) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line if line.endswith("\n") else line + "\n" for line in lines],
    }


def main() -> None:
    attack_b64 = _b64(ROOT / "attack.py")
    groq_b64 = _b64(ROOT / "scripts" / "groq_agent.py")
    probe_b64 = _b64(ROOT / "scripts" / "probe_lib.py")

    cells = [
        _md(
            [
                "# JED Gemma / Groq probe",
                "",
                "Loads the competition Gemma 4 GGUF on T4x2 and runs the same SECRET_MARKER ",
                "templates we would submit. Optional Groq `openai/gpt-oss-20b` if the ",
                "`GROQ_API_KEY` secret is attached.",
                "",
                "This is **not** a competition submission. Internet stays on so the GGUF ",
                "can download. Attach `GROQ_API_KEY` via Add-ons → Secrets, then Run All.",
            ]
        ),
        _code(
            [
                "import base64, os, sys",
                "from pathlib import Path",
                "",
                "working = Path('/kaggle/working') if Path('/kaggle/working').exists() else Path.cwd()",
                "files = {",
                f"    'attack.py': '''{attack_b64}''',",
                f"    'groq_agent.py': '''{groq_b64}''',",
                f"    'probe_lib.py': '''{probe_b64}''',",
                "}",
                "for name, blob in files.items():",
                "    path = working / name",
                "    path.write_bytes(base64.b64decode(''.join(blob.split())))",
                "    print('wrote', path, path.stat().st_size)",
                "sys.path.insert(0, str(working))",
            ]
        ),
        _code(
            [
                "import sys",
                "from pathlib import Path",
                "",
                "candidates = [",
                "    Path('/kaggle/input/ai-agent-security-multi-step-tool-attacks'),",
                "    Path('/kaggle/input'),",
                "]",
                "sdk_root = None",
                "for root in candidates:",
                "    if not root.exists():",
                "        continue",
                "    hits = list(root.rglob('aicomp_sdk/__init__.py'))",
                "    if hits:",
                "        sdk_root = hits[0].parent.parent",
                "        break",
                "if sdk_root is None:",
                "    raise RuntimeError('aicomp_sdk not found under /kaggle/input')",
                "sys.path.insert(0, str(sdk_root))",
                "print('SDK root', sdk_root)",
            ]
        ),
        _code(
            [
                "import subprocess, sys",
                "",
                "def pip(*pkgs):",
                "    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', *pkgs])",
                "",
                "pip('huggingface_hub', 'openai', 'pydantic')",
                "try:",
                "    import llama_cpp  # noqa: F401",
                "    print('llama_cpp already present')",
                "except Exception:",
                "    print('installing llama-cpp-python CUDA wheel')",
                "    subprocess.check_call([",
                "        sys.executable, '-m', 'pip', 'install', '-q', 'llama-cpp-python',",
                "        '--extra-index-url', 'https://abetlen.github.io/llama-cpp-python/whl/cu122',",
                "    ])",
                "    import llama_cpp  # noqa: F401",
                "    print('llama_cpp', llama_cpp.__file__)",
            ]
        ),
        _code(
            [
                "import json, os, traceback",
                "from pathlib import Path",
                "",
                "from aicomp_sdk.evaluation.ops import build_attack_env",
                "from aicomp_sdk.guardrails.optimal import Guardrail as OptimalGuardrail",
                "from kaggle_evaluation.jed_attack_134815.gemma_model_server import SPEC",
                "from kaggle_evaluation.jed_attack_134815.gguf_model_server import GgufModelServer",
                "from probe_lib import default_prompts, run_prompts",
                "",
                "working = Path('/kaggle/working') if Path('/kaggle/working').exists() else Path.cwd()",
                "fixtures = None",
                "for hit in Path('/kaggle/input').rglob('aicomp_sdk/fixtures'):",
                "    if hit.is_dir():",
                "        fixtures = hit",
                "        break",
                "print('fixtures', fixtures)",
                "",
                "print('Loading Gemma GGUF (first run downloads the file)...')",
                "server = GgufModelServer(SPEC)",
                "gemma_agent = server._load_agent()",
                "print('Gemma agent ready', type(gemma_agent).__name__)",
                "",
                "env = build_attack_env(",
                "    seed=123,",
                "    fixtures_dir=fixtures,",
                "    agent_factory=lambda: gemma_agent,",
                "    guardrail=OptimalGuardrail(),",
                "    env_selection='sandbox',",
                "    max_tool_hops=8,",
                ")",
                "gemma_report = run_prompts(env, default_prompts(), hops=8)",
                "gemma_report['backend'] = 'gemma4_gguf'",
                "(working / 'gemma_probe.json').write_text(json.dumps(gemma_report, indent=2))",
                "print('wrote gemma_probe.json')",
            ]
        ),
        _code(
            [
                "import json, os",
                "from pathlib import Path",
                "",
                "from aicomp_sdk.evaluation.ops import build_attack_env",
                "from aicomp_sdk.guardrails.optimal import Guardrail as OptimalGuardrail",
                "from probe_lib import default_prompts, run_prompts",
                "",
                "working = Path('/kaggle/working') if Path('/kaggle/working').exists() else Path.cwd()",
                "key = os.environ.get('GROQ_API_KEY', '').strip()",
                "if not key:",
                "    try:",
                "        from kaggle_secrets import UserSecretsClient",
                "        key = UserSecretsClient().get_secret('GROQ_API_KEY').strip()",
                "        os.environ['GROQ_API_KEY'] = key",
                "    except Exception as err:",
                "        print('no GROQ_API_KEY secret:', err)",
                "        key = ''",
                "",
                "if not key:",
                "    print('Skipping Groq GPT-OSS probe. Attach GROQ_API_KEY in Add-ons → Secrets.')",
                "    groq_report = {'skipped': True}",
                "else:",
                "    from groq_agent import GroqChatAgent",
                "    agent = GroqChatAgent(api_key=key)",
                "    fixtures = None",
                "    for hit in Path('/kaggle/input').rglob('aicomp_sdk/fixtures'):",
                "        if hit.is_dir():",
                "            fixtures = hit",
                "            break",
                "    env = build_attack_env(",
                "        seed=123,",
                "        fixtures_dir=fixtures,",
                "        agent_factory=lambda: agent,",
                "        guardrail=OptimalGuardrail(),",
                "        env_selection='sandbox',",
                "        max_tool_hops=8,",
                "    )",
                "    groq_report = run_prompts(env, default_prompts(), hops=8)",
                "    groq_report['backend'] = 'groq_gpt_oss_20b'",
                "",
                "(working / 'groq_probe.json').write_text(json.dumps(groq_report, indent=2))",
                "print('wrote groq_probe.json')",
                "print('GEMMA', {k: gemma_report.get(k) for k in ('fires','n','raw','normalized','elapsed_s')})",
                "print('GROQ ', {k: groq_report.get(k) for k in ('fires','n','raw','normalized','elapsed_s','skipped')})",
            ]
        ),
    ]

    notebook = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
            "kaggle": {
                "accelerator": "nvidiaTeslaT4",
                "isInternetEnabled": True,
                "language": "python",
                "sourceType": "notebook",
            },
        },
        "cells": cells,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    NOTEBOOK_PATH.write_text(json.dumps(notebook, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {NOTEBOOK_PATH}")


if __name__ == "__main__":
    main()
