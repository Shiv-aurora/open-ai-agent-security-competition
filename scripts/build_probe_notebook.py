"""Build a T4 probe notebook: Gemma GGUF + optional Groq, gym env, generate+replay."""

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
    files_b64 = {
        "attack.py": _b64(ROOT / "attack.py"),
        "groq_agent.py": _b64(ROOT / "scripts" / "groq_agent.py"),
        "probe_lib.py": _b64(ROOT / "scripts" / "probe_lib.py"),
        "score_formula.py": _b64(ROOT / "scripts" / "score_formula.py"),
        "kaggle_like_bench.py": _b64(ROOT / "scripts" / "kaggle_like_bench.py"),
        "replay_scorer.py": _b64(ROOT / "scripts" / "replay_scorer.py"),
    }
    file_literal = ",\n".join(
        f"    {json.dumps(name)}: '''{blob}'''" for name, blob in files_b64.items()
    )

    cells = [
        _md(
            [
                "# JED reliable replay scorer (Gemma GGUF)",
                "",
                "Gateway-shaped gym replay: **V5 singles baseline**, current `attack.py`, ",
                "and a pack-4 canary. Misses count as 0. Ship only if the pessimistic ",
                "K=1 / extra-message tax still matches V5 (the V6 failure mode).",
                "",
                "Internet on, T4, not a submission. Groq is skipped unless a secret is attached.",
            ]
        ),
        _code(
            [
                "import base64, sys",
                "from pathlib import Path",
                "",
                "working = Path('/kaggle/working') if Path('/kaggle/working').exists() else Path.cwd()",
                f"files = {{\n{file_literal}\n}}",
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
                "sdk_root = None",
                "for root in [Path('/kaggle/input/ai-agent-security-multi-step-tool-attacks'), Path('/kaggle/input')]:",
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
                "pip('huggingface_hub', 'openai', 'pydantic', 'gymnasium')",
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
                "import json, os, sys",
                "from pathlib import Path",
                "",
                "working = Path('/kaggle/working') if Path('/kaggle/working').exists() else Path.cwd()",
                "sys.path.insert(0, str(working))",
                "",
                "from kaggle_like_bench import gemma_gguf_factory",
                "from replay_scorer import run_reliable_score",
                "",
                "print('Loading Gemma GGUF...')",
                "factory = gemma_gguf_factory()",
                "print('Gemma ready')",
                "gemma = run_reliable_score(",
                "    factory,",
                "    env_selection='gym',",
                "    n_per=30,",
                "    replay_s=900.0,",
                "    include_pack_canary=True,",
                "    pack_n=10,",
                ")",
                "gemma['backend'] = 'gemma4_gguf'",
                "(working / 'gemma_score.json').write_text(json.dumps(gemma, indent=2))",
                "print('wrote gemma_score.json')",
                "print('VERDICT', gemma.get('verdict'), 'ship', gemma.get('ship'), gemma.get('reason'))",
            ]
        ),
        _code(
            [
                "import json, os, sys",
                "from pathlib import Path",
                "",
                "working = Path('/kaggle/working') if Path('/kaggle/working').exists() else Path.cwd()",
                "sys.path.insert(0, str(working))",
                "key = os.environ.get('GROQ_API_KEY', '').strip()",
                "if not key:",
                "    try:",
                "        from kaggle_secrets import UserSecretsClient",
                "        key = UserSecretsClient().get_secret('GROQ_API_KEY').strip()",
                "        os.environ['GROQ_API_KEY'] = key",
                "    except Exception as err:",
                "        print('no GROQ_API_KEY:', err)",
                "        key = ''",
                "",
                "if not key:",
                "    print('Skipping Groq. Attach GROQ_API_KEY in Add-ons → Secrets.')",
                "    groq = {'skipped': True}",
                "else:",
                "    from kaggle_like_bench import groq_factory",
                "    from replay_scorer import run_reliable_score",
                "    factory = groq_factory()",
                "    groq = run_reliable_score(",
                "        factory,",
                "        env_selection='gym',",
                "        n_per=8,",
                "        replay_s=240.0,",
                "        include_pack_canary=True,",
                "        pack_n=4,",
                "    )",
                "    groq['backend'] = 'groq_gpt_oss_20b'",
                "",
                "(working / 'groq_score.json').write_text(json.dumps(groq, indent=2))",
                "print('wrote groq_score.json')",
                "def _brief(d):",
                "    if not isinstance(d, dict) or d.get('skipped'):",
                "        return d",
                "    fam = {}",
                "    for name, row in (d.get('families') or {}).items():",
                "        fam[name] = {",
                "            'mean_posts': row.get('mean_posts'),",
                "            'fire_rate': row.get('fire_rate'),",
                "            'mean_latency_s': row.get('mean_latency_s'),",
                "            'n_kaggle': row.get('n_kaggle'),",
                "            'displayed_both_private0': row.get('displayed_both_private0'),",
                "            'pess_displayed': row.get('pess_displayed'),",
                "            'n_messages': row.get('n_messages'),",
                "        }",
                "    return {",
                "        'backend': d.get('backend'),",
                "        'verdict': d.get('verdict'),",
                "        'ship': d.get('ship'),",
                "        'reason': d.get('reason'),",
                "        'families': fam,",
                "    }",
                "print('GEMMA', json.dumps(_brief(gemma)))",
                "print('GROQ ', json.dumps(_brief(groq)))",
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
