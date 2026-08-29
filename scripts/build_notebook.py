"""Build the Kaggle submission notebook with attack.py embedded as base64."""

from __future__ import annotations

import base64
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ATTACK_PATH = ROOT / "attack.py"
OUT_DIR = ROOT / "kaggle_kernel"
NOTEBOOK_PATH = OUT_DIR / "submit.ipynb"
NOTEBOOK_COPY = ROOT / "notebooks" / "submit.ipynb"


def main() -> None:
    source = ATTACK_PATH.read_bytes()
    b64 = base64.b64encode(source).decode("ascii")
    # Keep lines short enough for the notebook editor.
    wrapped = "\n".join(b64[i : i + 80] for i in range(0, len(b64), 80))

    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# JED Attack V5\n",
                "\n",
                "Writes `attack.py` then starts the competition inference server. "
                "On a normal commit it only emits a placeholder `submission.csv`; "
                "the hosted rerun (`KAGGLE_IS_COMPETITION_RERUN`) runs the real evaluation.\n",
            ],
        },
        {
            "cell_type": "code",
            "metadata": {},
            "execution_count": None,
            "outputs": [],
            "source": [
                "import base64\n",
                "import os\n",
                "from pathlib import Path\n",
                "\n",
                f"_ATTACK_B64 = \"\"\"{wrapped}\"\"\"\n",
                "\n",
                "working = Path(\"/kaggle/working\") if Path(\"/kaggle/working\").exists() else Path.cwd()\n",
                "attack_path = working / \"attack.py\"\n",
                "attack_path.write_bytes(base64.b64decode(_ATTACK_B64))\n",
                "print(f\"wrote {attack_path} ({attack_path.stat().st_size} bytes)\")\n",
            ],
        },
        {
            "cell_type": "code",
            "metadata": {},
            "execution_count": None,
            "outputs": [],
            "source": [
                "import os\n",
                "from pathlib import Path\n",
                "\n",
                "working = Path(\"/kaggle/working\") if Path(\"/kaggle/working\").exists() else Path.cwd()\n",
                "\n",
                "if os.getenv(\"KAGGLE_IS_COMPETITION_RERUN\"):\n",
                "    from kaggle_evaluation.jed_attack_134815.jed_attack_inference_server import (\n",
                "        JEDAttackInferenceServer,\n",
                "    )\n",
                "\n",
                "    JEDAttackInferenceServer().run()\n",
                "else:\n",
                "    submission = working / \"submission.csv\"\n",
                "    submission.write_text(\n",
                "        \"Id,Score\\n\"\n",
                "        \"gpt_oss_public,0.0\\n\"\n",
                "        \"gpt_oss_private,0.0\\n\"\n",
                "        \"gemma_public,0.0\\n\"\n",
                "        \"gemma_private,0.0\\n\"\n",
                "    )\n",
                "    print(f\"placeholder {submission} (not a competition rerun)\")\n",
            ],
        },
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
                "isInternetEnabled": False,
                "language": "python",
                "sourceType": "notebook",
            },
        },
        "cells": cells,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    NOTEBOOK_COPY.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(notebook, indent=1) + "\n"
    NOTEBOOK_PATH.write_text(payload, encoding="utf-8")
    NOTEBOOK_COPY.write_text(payload, encoding="utf-8")
    print(f"wrote {NOTEBOOK_PATH}")
    print(f"wrote {NOTEBOOK_COPY}")


if __name__ == "__main__":
    main()
