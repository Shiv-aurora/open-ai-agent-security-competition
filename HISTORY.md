# Submission history

Kaggle: `ai-agent-security-multi-step-tool-attacks` (`shivaurora`).

| Version | Kernel | Submit | Public |
|---|---|---|---|
| v1 | `shivaurora/jed-high-yield-attack-v1` | `55779300` | **0.30** |
| v3 | same kernel lineage | `55797613` | **24.25** |
| V4 | `shivaurora/jed-attack-v4` | `55872908` | pending |

V4 uses short unique hosts (`aa.co`). Fast path burst-8, slow path Harmony-10.
Classify/warmup fires are discarded. Live fires are padded to 2000 with
unvalidated burst-8; zero fires fall back to 300. Groq V4 stacked ~1.6 posts
per interact (27.6 raw/fire), not 8 or 10.
