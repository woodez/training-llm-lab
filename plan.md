# Lab Plan: Fine-Tuning a Small Local LLM on Apple M3 (16GB)

A hands-on lab for learning how to fine-tune a small language model on your own
notes and documents, entirely on a MacBook with an Apple M3 and 16GB of unified
memory, using Apple's **MLX** framework (`mlx-lm`) and **LoRA / QLoRA**.

- **Format:** 5 sessions (~3–4 hours each), sized for a few weekends.
- **Audience:** new to ML training — each step includes a short "concept" note.
- **Rule of the lab:** nothing is installed or run without an explicit go-ahead
  per session (see `CLAUDE.md`).

---

## 1. Goal & honest framing

**Goal:** take a small open model, adapt it to *your* notes/documents, and
measure — with numbers, not vibes — whether it got better.

**The key thing to know up front:** fine-tuning (especially LoRA) is very good at
teaching a model *style, format, tone, and habits* (e.g. "answer like my notes
do", "always use this structure", "know my vocabulary"). It is **unreliable at
injecting new facts** — a fine-tuned small model will often confidently
hallucinate details that sound like your notes. The standard technique for
"answer questions using my documents" is **retrieval (RAG)**: look up relevant
chunks and paste them into the prompt.

So this lab deliberately compares four setups on the same held-out questions:

| Setup | What it is |
|---|---|
| **A. Base** | Off-the-shelf model, no help |
| **B. Fine-tuned** | Base + your LoRA adapter |
| **C. Base + RAG** | Base model, relevant note chunks pasted into the prompt |
| **D. Fine-tuned + RAG** | Adapter *and* retrieval |

You'll finish knowing first-hand when fine-tuning helps and when it doesn't.

---

## 2. Hardware budget (M3, 16GB unified memory)

On Apple Silicon, CPU and GPU share the same RAM. macOS + your apps use some of
it; plan on **~10–11GB realistically available** for training. Close browsers
and other heavy apps during runs.

### Recommended models (MLX 4-bit builds from `mlx-community` on Hugging Face)

| Role | Model | Weights on disk (approx.) | Training fit on 16GB |
|---|---|---|---|
| Fast iteration | `mlx-community/Qwen3-0.6B-4bit` | ~0.4GB | Easy — use for debugging |
| **Main model** | `mlx-community/Qwen3-1.7B-4bit` | ~1GB | Comfortable |
| Helper (not trained) | `mlx-community/Qwen3-4B-4bit` | ~2.3GB | Used only for inference: generating Q&A pairs (Session 2) and optional judging (Session 4) |

> Numbers are rough; the real figure that matters is the **peak memory**
> `mlx_lm.lora` prints during training. Record it for every run.

> Qwen3 models have a "thinking" mode that emits `<think>…</think>` blocks.
> For this lab, disable it (add `/no_think` to the prompt, or use the chat
> template with thinking disabled) so outputs and training data stay simple.

### Memory knobs (in order of impact)

1. **Model size / quantization** — 4-bit base weights = "QLoRA". Biggest lever.
2. **`--batch-size`** — start at 1–2. Halve it first when you run out of memory.
3. **`--max-seq-length`** — 512–1024 for this lab. Long chunks cost memory.
4. **`--num-layers`** — how many transformer layers get LoRA adapters (e.g. 8
   or 16 instead of all). Fewer = less memory, less capacity.
5. **`--grad-checkpoint`** — trades extra compute for much lower memory.

**What OOM looks like on a Mac:** the run crashes with a Metal allocation
error, or the whole machine slows to a crawl as it swaps. Open
**Activity Monitor → Memory** and watch "Memory Pressure"; yellow/red means back
off one of the knobs above.

---

## 3. Repo layout

```
training-llm-lab/
├── plan.md                 # this file
├── requirements.txt        # pinned deps
├── configs/                # one YAML per training run
│   └── run01_qwen1.7b_r8.yaml
├── data/
│   ├── raw/                # your original notes (GIT-IGNORED, private)
│   ├── interim/            # extracted text + chunks (GIT-IGNORED)
│   ├── processed/          # train.jsonl / valid.jsonl / test.jsonl (GIT-IGNORED)
│   └── warmup/             # public warm-up dataset (generated, git-ignored)
├── eval/
│   ├── heldout_questions.jsonl   # questions never used in training
│   └── results/            # per-setup answers + scores
├── scripts/
│   ├── bench_inference.py  # tokens/sec + memory check
│   ├── prep_warmup.py      # public dataset -> chat JSONL
│   ├── extract_text.py     # notes/PDFs -> plain text
│   ├── chunk.py            # text -> chunks
│   ├── make_qa.py          # chunks -> Q&A pairs (local model and/or manual)
│   ├── split.py            # -> train/valid/test
│   ├── rag.py              # simple BM25 retrieval over chunks
│   └── evaluate.py         # run setups A–D on held-out questions
├── adapters/               # LoRA adapter outputs (GIT-IGNORED)
├── fused/                  # merged models (GIT-IGNORED)
└── experiments/
    └── log.md              # one row per run (config, memory, time, loss, notes)
```

**`.gitignore` additions:** `data/raw/`, `data/interim/`, `data/processed/`,
`data/warmup/`, `adapters/`, `fused/`, `*.safetensors`, `*.gguf`.

**Privacy:** your notes, generated Q&A pairs (which quote your notes), and
trained adapters (which can leak memorized text) all stay **local and out of
git**. Everything runs offline after the model download.

---

## 4. Sessions

### Session 0 — Setup & baseline (≈2–3h)

**Goal:** a working MLX environment and a feel for how fast the M3 is.

Steps:
1. Confirm Python 3.13 is available (`python3.13 --version`; install via
   Homebrew or pyenv if not).
2. Create and activate the venv:
   ```bash
   python3.13 -m venv .venv
   source .venv/bin/activate
   pip install --upgrade pip
   pip install mlx-lm
   pip freeze > requirements.txt
   ```
   - Verify MLX supports 3.13 on the current release
     (`python -c "import mlx.core as mx; print(mx.default_device())"` should
     print a GPU device). If wheels aren't available, stop and decide together.
3. Update `.gitignore` with the entries from §3.
4. First inference:
   ```bash
   mlx_lm.generate --model mlx-community/Qwen3-1.7B-4bit \
     --prompt "Explain LoRA fine-tuning in two sentences. /no_think" \
     --max-tokens 200
   ```
5. Write `scripts/bench_inference.py`: load 0.6B / 1.7B / 4B, generate 256
   tokens, record **tokens/sec** and **peak memory** (`mx.get_peak_memory()`,
   or `mx.metal.get_peak_memory()` on older MLX).
6. Interactive poke: `mlx_lm.chat --model mlx-community/Qwen3-1.7B-4bit`. Ask it
   questions about topics in your notes and save a few answers — this is your
   "before" snapshot.

> **Concept — quantization:** model weights are normally 16-bit numbers.
> 4-bit quantization stores them in a quarter of the space with a small quality
> loss. That's why a 1.7B-parameter model fits in ~1GB.

**Deliverable:** venv + `requirements.txt` committed, benchmark table in
`experiments/log.md`, "before" answers saved.

---

### Session 1 — Warm-up fine-tune on a public dataset (≈3h)

**Goal:** learn the training loop on data where nothing private is at stake and
mistakes are cheap.

Steps:
1. `scripts/prep_warmup.py`: download a small instruct dataset
   (e.g. `databricks/databricks-dolly-15k`, take ~1,000 rows) and convert to
   MLX's chat JSONL format:
   ```json
   {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
   ```
   Write `data/warmup/train.jsonl` (~900) and `valid.jsonl` (~100).
2. First LoRA run on the **0.6B** model (fast feedback):
   ```bash
   mlx_lm.lora --model mlx-community/Qwen3-0.6B-4bit \
     --train --data data/warmup \
     --fine-tune-type lora \
     --iters 300 --batch-size 2 --num-layers 8 \
     --learning-rate 1e-4 --max-seq-length 512 \
     --steps-per-eval 50 --val-batches 20 \
     --mask-prompt \
     --adapter-path adapters/warmup01
   ```
3. Watch the output: training loss, validation loss, tokens/sec, peak memory.
4. Generate with the adapter and compare to base:
   ```bash
   mlx_lm.generate --model mlx-community/Qwen3-0.6B-4bit \
     --adapter-path adapters/warmup01 --prompt "..."
   ```
5. **Deliberately break things** to learn what they look like:
   - Learning rate 10× too high (`1e-3`) → loss spikes / NaN / gibberish.
   - Way too many iterations on tiny data → train loss ↓, val loss ↑
     (overfitting).
   - Batch size / seq length too large → see what OOM looks like.
6. Repeat the good config on **1.7B**, move it into `configs/` as YAML
   (`mlx_lm.lora -c configs/warmup_1.7b.yaml`) so runs are reproducible.

> **Concept — LoRA:** instead of updating all ~1.7B weights, LoRA freezes them
> and trains small "adapter" matrices (a few million parameters) added to
> selected layers. **Rank** controls adapter size/capacity. **QLoRA** = LoRA on
> top of a quantized (4-bit) base — what we're doing.
>
> **Concept — train vs. validation loss:** loss measures how surprised the
> model is by the correct next token (lower = better). Train loss is on data it
> learns from; validation loss is on data it never trains on. When val loss
> stops falling and starts rising while train loss keeps falling, the model is
> **memorizing** instead of **generalizing** — stop there.
>
> **Concept — `--mask-prompt`:** only compute loss on the assistant's answer,
> not the user's question. You want it to learn to *answer*, not to *ask*.

**Deliverable:** 4–6 runs logged in `experiments/log.md`; you can read a loss
curve and recognize overfitting, divergence, and OOM.

---

### Session 2 — Turn your notes into a dataset (≈3–4h)

**Goal:** a clean, split, private dataset built from your notes, plus a
held-out evaluation set. *This is where most of the quality comes from.*

Steps:
1. **Collect:** copy notes into `data/raw/` (Markdown, text, PDFs). Start with a
   focused topic area rather than everything — e.g. 20–100 documents.
2. **Extract:** `scripts/extract_text.py` → plain text in `data/interim/`.
   Markdown/text pass through; PDFs via `pypdf` (added to requirements when we
   get there). Spot-check output for garbage (headers, page numbers, broken
   tables).
3. **Chunk:** `scripts/chunk.py` — split into ~300–500-token chunks along
   headings/paragraphs, with a little overlap. Keep the source file + heading
   for each chunk (used later by RAG and for tracing answers).
4. **Make training pairs** — `scripts/make_qa.py`, combining two sources:
   - **Hand-written (gold):** you write 30–50 Q&A pairs in your own voice.
     Small but highest quality.
   - **Synthetic:** a local model (e.g. Qwen3-4B-4bit via `mlx_lm`) reads each
     chunk and proposes 2–5 question/answer pairs grounded in that chunk.
     Slow-ish on an M3 but fully offline.
   - Target **~300–1,500 pairs** total. Quality > quantity.
5. **Clean:** dedupe near-identical questions, drop answers that aren't
   supported by their chunk, drop very long examples (> max seq length). Read
   a random sample of 30 by eye.
6. **Hold out an eval set first:** `eval/heldout_questions.jsonl` — 40–60
   questions with reference answers and the source chunk id. Mix:
   - questions directly answerable from one chunk,
   - questions needing two chunks,
   - a few questions your notes *don't* answer (does the model admit it?).

   Split by **document** where possible, so eval questions don't come from
   chunks the model was trained on — otherwise you're testing memorization.
7. **Split** the rest: `scripts/split.py` → `data/processed/{train,valid,test}.jsonl`
   at ~80/10/10, in the same chat format as Session 1.

> **Concept — data leakage:** if eval questions (or near-paraphrases) appear in
> training, scores look great and mean nothing. Keep the held-out set sealed.
>
> **Concept — how much data?** For style/format, a few hundred good examples is
> often enough. For "knowing" content, more data helps, but returns are
> limited — which is exactly what Session 4 will show.

**Deliverable:** `data/processed/*.jsonl`, `eval/heldout_questions.jsonl`, a
short note in `experiments/log.md` on dataset size and composition.

---

### Session 3 — Fine-tune on your data & run experiments (≈3–4h)

**Goal:** a good adapter, and a feel for which knobs matter.

Steps:
1. **Baseline run** (Qwen3-1.7B-4bit), e.g. `configs/run01_qwen1.7b_r8.yaml`:
   ```yaml
   model: mlx-community/Qwen3-1.7B-4bit
   train: true
   data: data/processed
   fine_tune_type: lora
   num_layers: 16
   batch_size: 2
   iters: 600            # roughly 1–3 passes over the data; adjust to dataset size
   learning_rate: 1.0e-4
   max_seq_length: 1024
   grad_checkpoint: true
   mask_prompt: true
   steps_per_eval: 50
   val_batches: 25
   save_every: 100
   adapter_path: adapters/run01
   lora_parameters:
     rank: 8
     scale: 20.0
     dropout: 0.0
   ```
   Run: `mlx_lm.lora -c configs/run01_qwen1.7b_r8.yaml`
2. **Test loss:** `mlx_lm.lora --model ... --adapter-path adapters/run01 --data data/processed --test`
3. **Experiments — change one thing at a time:**

   | Run | Change vs. baseline | Question it answers |
   |---|---|---|
   | run02 | rank 16 | Does more adapter capacity help? |
   | run03 | rank 32 | …or does it just overfit faster? |
   | run04 | LR 5e-5 | Is the baseline LR too aggressive? |
   | run05 | LR 2e-4 | …or too timid? |
   | run06 | 2× iters | Where does val loss bottom out? |
   | run07 | 0.6B model | How much does size matter? |

   Use **0.6B** to sanity-check any new config before spending time on 1.7B.
4. **Log every run** in `experiments/log.md`:

   | run | model | rank | LR | iters | best val loss (step) | test loss | peak mem | time | notes |
   |---|---|---|---|---|---|---|---|---|---|

5. Pick the best checkpoint by **validation loss + a quick read of 10 sample
   answers** (low loss doesn't always mean nicer answers).

> **Concept — one variable at a time:** if you change rank *and* LR together
> and it improves, you don't know why. Ablations are slow but they're how you
> actually learn.
>
> **Concept — checkpoints:** `save_every` writes intermediate adapters. If the
> best val loss was at step 300 of 600, use that one.

**Deliverable:** 6–7 logged runs, a chosen "best" adapter, and 2–3 sentences
on what mattered most.

---

### Session 4 — Evaluate, compare with RAG, package (≈3–4h)

**Goal:** an honest answer to "did fine-tuning help, and with what?"

Steps:
1. **Simple RAG:** `scripts/rag.py` — BM25 keyword search over the chunks
   from Session 2 (`rank-bm25`, lightweight, no extra models). Top 3 chunks get
   pasted into the prompt: *"Answer using only the context below. If the answer
   isn't there, say so."*
2. **Run all four setups** (A Base, B Fine-tuned, C Base+RAG, D FT+RAG) on every
   held-out question with `scripts/evaluate.py`. Save raw answers to
   `eval/results/`.
3. **Score** each answer 0/1/2 (wrong / partial / correct) plus a
   **hallucination flag** (states something not in your notes):
   - Manually score at least 20 questions yourself — this is ground truth.
   - Optionally, use a larger local model as a judge for the rest, then check
     how well it agrees with your manual scores.
4. **Summarize:**

   | Setup | Avg score | % correct | Hallucination rate | "Says I don't know" on unanswerable |
   |---|---|---|---|---|
   | A Base | | | | |
   | B FT | | | | |
   | C RAG | | | | |
   | D FT+RAG | | | | |

   Typical outcome to expect (but verify!): RAG wins on facts; fine-tuning
   wins on tone/format and makes RAG answers read more like you; FT alone
   hallucinates more confidently.
5. **Package the best model:**
   ```bash
   mlx_lm.fuse --model mlx-community/Qwen3-1.7B-4bit \
     --adapter-path adapters/runXX --save-path fused/my-notes-1.7b
   ```
   Use with `mlx_lm.generate --model fused/my-notes-1.7b` or `mlx_lm.server`
   (OpenAI-compatible API on localhost). Optional: export to GGUF/Ollama —
   check current `mlx_lm.fuse --export-gguf` support for the model family
   (it may require de-quantizing first); otherwise just keep the adapter.
6. **Write-up** (`experiments/summary.md`): what worked, what didn't, the
   results table, and what you'd try next.

> **Concept — why manual scoring:** automatic metrics (loss, BLEU, etc.) don't
> capture "is this answer right and useful". A small, careful human-scored set
> beats a large sloppy automated one.

**Deliverable:** results table, fused model, `experiments/summary.md`.

---

## 5. Troubleshooting cheatsheet

| Symptom | Likely cause | Fix |
|---|---|---|
| Metal / allocation error, or machine swapping | Out of memory | ↓ batch size → ↓ max seq length → ↓ num layers → `--grad-checkpoint` → smaller model |
| Loss is `nan` or jumps up | LR too high | ÷ LR by 2–10 |
| Loss barely moves | LR too low / too few iters / data format wrong | ↑ LR, check a JSONL row by eye |
| Val loss rises while train loss falls | Overfitting | Fewer iters, earlier checkpoint, more/diverse data, lower rank |
| Outputs repeat or ramble | Too few EOS examples / overfit / bad prompt template | Check chat template, `--mask-prompt`, earlier checkpoint |
| `<think>` blocks everywhere | Qwen3 thinking mode | Disable thinking in prompts and in training data |
| Model invents facts "from your notes" | Expected with FT alone | That's what RAG (setup C/D) is for |
| Very slow training | Long sequences / big model / other apps using GPU | Shorter chunks, 0.6B for iteration, close apps, keep laptop plugged in |

---

## 6. Stretch goals (after the core lab)

- **DPO / preference tuning:** collect "chosen vs. rejected" answer pairs from
  your eval and nudge the model toward your preferred style.
- **Better retrieval:** swap BM25 for an embedding model, compare.
- **Full fine-tune vs. LoRA** on the 0.6B model (`--fine-tune-type full`) —
  feel the memory difference.
- **Tiny model from scratch:** a nanoGPT-style ~10–30M-param model on a small
  text corpus, to see pretraining mechanics.
- **PyTorch/MPS comparison:** same run with Hugging Face `peft`/`trl` on MPS;
  compare speed and memory.
- **Tool-calling adapter:** teach the model to emit structured JSON tool calls
  for the `fi-agent` idea in `CLAUDE.md`.

---

## 7. Glossary

- **Parameter:** a learned number in the model. "1.7B" = 1.7 billion of them.
- **Token:** a word piece; models read and write tokens, ~¾ of a word each.
- **Quantization (4-bit):** storing weights with fewer bits to save memory.
- **Fine-tuning:** continuing training of a pre-trained model on new data.
- **LoRA:** training small adapter matrices instead of all weights.
- **QLoRA:** LoRA on a quantized base model.
- **Rank (r):** size of the LoRA adapter; higher = more capacity, more memory.
- **Learning rate (LR):** how big each update step is.
- **Iteration / step:** one batch processed and one weight update.
- **Epoch:** one full pass over the training data.
- **Batch size:** examples processed per step.
- **Loss:** how wrong the model's predictions are; lower is better.
- **Overfitting:** memorizing training data at the cost of generalization.
- **Adapter:** the small file of LoRA weights produced by training.
- **Fuse:** merging an adapter permanently into the base weights.
- **RAG:** Retrieval-Augmented Generation — fetching relevant text and putting
  it in the prompt.
- **Held-out set:** examples never seen during training, used only to evaluate.

---

## 8. Session checklist

- [ ] **S0** venv, `mlx-lm`, benchmark, "before" answers
- [ ] **S1** warm-up LoRA runs, break-it experiments, YAML config
- [ ] **S2** notes → chunks → Q&A, held-out eval set, splits
- [ ] **S3** baseline + 6 ablation runs, best adapter chosen
- [ ] **S4** RAG baseline, 4-way eval, fused model, summary
