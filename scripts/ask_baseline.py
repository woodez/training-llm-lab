"""Ask a model each question from a Q&A file, one fresh context per question.

Q&A file format (blocks separated by blank lines):
    - "Question\\nExpected answer"     -> question with a reference answer
    - "1. Question (may wrap lines)"  -> numbered question, no reference answer

Usage:
    python scripts/ask_baseline.py Q&A.md eval/results/baseline_1.7b.jsonl \\
        [--model mlx-community/Qwen3-1.7B-4bit] [--max-tokens 400]
"""

import argparse
import json
import re
from pathlib import Path

from mlx_lm import generate, load

NUMBERED = re.compile(r"^\d+\.\s+")


def parse_questions(path: Path) -> list[dict]:
    blocks = [b.strip() for b in path.read_text().split("\n\n") if b.strip()]
    items = []
    for block in blocks:
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if NUMBERED.match(lines[0]):
            question = NUMBERED.sub("", " ".join(lines))
            items.append({"question": question, "reference": None})
        else:
            reference = " ".join(lines[1:]) or None
            items.append({"question": lines[0], "reference": reference})
    return items


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("questions", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--model", default="mlx-community/Qwen3-1.7B-4bit")
    parser.add_argument("--max-tokens", type=int, default=400)
    args = parser.parse_args()

    items = parse_questions(args.questions)
    model, tokenizer = load(args.model)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        for i, item in enumerate(items, 1):
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": item["question"]}],
                add_generation_prompt=True,
                enable_thinking=False,
            )
            answer = generate(model, tokenizer, prompt, max_tokens=args.max_tokens)
            record = {"id": i, "model": args.model, **item, "answer": answer.strip()}
            f.write(json.dumps(record) + "\n")
            print(f"[{i}/{len(items)}] {item['question'][:60]}")


if __name__ == "__main__":
    main()
