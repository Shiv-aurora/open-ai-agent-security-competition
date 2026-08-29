# Submission history

Kaggle: `ai-agent-security-multi-step-tool-attacks` (`shivaurora`).

| Version | Kernel | Submit | Public |
|---|---|---|---|
| v1 | `shivaurora/jed-high-yield-attack-v1` | `55779300` | **0.30** |
| v3 | same kernel lineage | `55797613` | **24.25** |

v3 replaces the explorer with live-validated unique-host `http.post` of
`SECRET_MARKER`. OptimalGuardrail checks the URL, not `data`. Keep only traces
that actually posted the sentinel. Conservative replay sizing; fallback 480.
Public 24.25 is about 0.09 × 269 singles.
