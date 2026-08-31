# Kaggle probe and submit

Account: `shivaurora`. Competition:
`ai-agent-security-multi-step-tool-attacks`. Deadline: 2026-09-01.

## Hard rules

- T4 via `"machine_shape": "NvidiaTeslaT4"` in kernel metadata only.
  **Never** `kaggle kernels push --accelerator`.
- Submit kernels: internet **off**. Probe kernels: internet **on**.
- Do **not** overwrite kernels `jed-attack-v4` … `jed-attack-v11`.
  Push a new slug (`jed-attack-v12`, …) for the next attack.
- Probe kernel slug is `shivaurora/jed-gemma-groq-probe` (reused on purpose).
- Code-competition submit from a **full path** to `submission.csv` returned
  HTTP 400. `cd` into the kernel output dir and `-f submission.csv` worked.

## Probe (Gemma T4 llama.cpp)

```bash
python3 scripts/build_probe_notebook.py
kaggle kernels push -p kaggle_probe
```

Metadata: GPU on, internet on, `NvidiaTeslaT4`. The probe is a vote, not the
hosted number. Groq Chat Completions ≠ GPT-OSS GGUF Harmony.

Colab Groq (do not git-clone; the clone was stale):

```bash
GROQ_API_KEY=… python3 scripts/colab_groq_bench.py
```

Gateway-shaped local replay:

```bash
python3 scripts/replay_scorer.py --backend gemma_gguf --env gym --n 30
python3 scripts/kaggle_like_bench.py --reliable --backend cooperating
```

## Submit

`scripts/build_notebook.py` reads `VERSION` from `attack.py` and embeds the
file as base64 in `kaggle_kernel/submit.ipynb` (copy in `notebooks/submit.ipynb`).

```bash
python3 scripts/build_notebook.py
# set a NEW kernel id in kaggle_kernel/kernel-metadata.json
kaggle kernels push -p kaggle_kernel
# confirm the kernel is T4, internet off
kaggle kernels output -k shivaurora/jed-attack-vN -p /tmp/jed-vN-out
cd /tmp/jed-vN-out
kaggle competitions submit \
  -c ai-agent-security-multi-step-tool-attacks \
  -k shivaurora/jed-attack-vN \
  -f submission.csv \
  -m "…" \
  -v 1
```

On a normal commit the notebook only writes a placeholder csv. The hosted
rerun (`KAGGLE_IS_COMPETITION_RERUN`) runs generate + replay.

## Predict before you push

```bash
python3 scripts/public_solver.py --predict \
  --gemma-k 1 --gemma-lat <measured> \
  --oss-k 1 --oss-lat <n_messages * 3.17>
```

Do not ship if pessimistic D < V5. Do not hardcode a Gemma-friendly
multi-message template onto every model (V6 pack, V8 Harmony, V9 dual-rail).
