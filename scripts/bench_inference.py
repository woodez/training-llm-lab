"""Benchmark inference speed and peak memory for the lab's models.

Usage:
    python scripts/bench_inference.py [--max-tokens 256] [model ...]
"""

import argparse
import gc

import mlx.core as mx
from mlx_lm import load, stream_generate

DEFAULT_MODELS = [
    "mlx-community/Qwen3-0.6B-4bit",
    "mlx-community/Qwen3-1.7B-4bit",
    "mlx-community/Qwen3-4B-4bit",
]

PROMPT = "Write a short paragraph explaining how a neural network learns from data."


def bench(model_name: str, max_tokens: int) -> dict:
    mx.reset_peak_memory()
    model, tokenizer = load(model_name)
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": PROMPT}],
        add_generation_prompt=True,
        enable_thinking=False,
    )

    last = None
    for last in stream_generate(model, tokenizer, prompt, max_tokens=max_tokens):
        pass

    result = {
        "model": model_name.split("/")[-1],
        "prompt_tps": last.prompt_tps,
        "generation_tps": last.generation_tps,
        "generated_tokens": last.generation_tokens,
        "peak_memory_gb": last.peak_memory,
    }

    del model, tokenizer
    gc.collect()
    mx.clear_cache()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("models", nargs="*", default=DEFAULT_MODELS)
    parser.add_argument("--max-tokens", type=int, default=256)
    args = parser.parse_args()

    results = [bench(m, args.max_tokens) for m in args.models]

    print("| model | prompt tok/s | generation tok/s | tokens generated | peak memory (GB) |")
    print("|---|---|---|---|---|")
    for r in results:
        print(
            f"| {r['model']} | {r['prompt_tps']:.1f} | {r['generation_tps']:.1f} "
            f"| {r['generated_tokens']} | {r['peak_memory_gb']:.2f} |"
        )


if __name__ == "__main__":
    main()
