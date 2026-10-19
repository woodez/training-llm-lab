# Experiment log

Mirrored to Notion: POC of training open LLM models → 🧪 Experiment Runs.

Environment: Apple M3, 16GB · macOS 26 · Python 3.13.0 (arm64) · mlx 0.32.4 · mlx-lm 0.32.2

## Session 0 — Setup & baseline (2026-10-18)

### First generation

`mlx_lm.generate --model mlx-community/Qwen3-1.7B-4bit --prompt "Explain LoRA fine-tuning in two sentences. /no_think" --max-tokens 200`

- Prompt: 23 tokens @ 19.3 tok/s · Generation: 68 tokens @ 86.9 tok/s · Peak memory: 1.174 GB
- Model expanded LoRA as "Low-Rank Approximation" (wrong: it's "Adaptation").

### Inference benchmark (`scripts/bench_inference.py`, max 256 tokens, thinking off)

| model | prompt tok/s | generation tok/s | tokens generated | peak memory (GB) | disk |
|---|---|---|---|---|---|
| Qwen3-0.6B-4bit | 471.4 | 220.0 | 107 | 0.55 | 335 MB |
| Qwen3-1.7B-4bit | 333.0 | 87.6 | 106 | 1.19 | 938 MB |
| Qwen3-4B-4bit | 76.7 | 37.2 | 120 | 2.44 | 2.1 GB |

All three stopped on end-of-sequence before 256 tokens.

### "Before" snapshot: untrained Qwen3-1.7B (`scripts/ask_baseline.py`)

19 questions, each in a fresh context, thinking off, greedy decoding, max 400 tokens.
The questions and answers are private (git-ignored in `Q&A.md` and `eval/results/`); only aggregates are kept here.

| set | n | score (0–2 each) | correct | hallucinated |
|---|---|---|---|---|
| personal / factual | 9 | 0 / 18 | 0 | 7 (2 admitted not knowing) |
| economics (course notes, provisional) | 10 | 12 / 20 | 5 (+2 generic) | 3 |
| **total** | 19 | **12 / 38 (avg 0.63)** | 5 | 10 |

Notes: missed even public facts (PLTR → "Palo Alto Software"); textbook economics is fine, but course-specific examples
(roommate, GO train, SMB for public goods) failed; answers run long (250–300 words) with heavy formatting.
