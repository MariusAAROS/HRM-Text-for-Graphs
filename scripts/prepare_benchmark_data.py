"""Build HRM-Text SFT JSONL train sets for the MMLU / GSM8k / MATH ablations.

Prompts are built with the SAME helpers `evaluation/benchmarks.py` uses at eval
time, so the training distribution matches the benchmark prompt format exactly.
Responses are "direct" (final answer only, no chain of thought), matching the
`condition: "direct"` used by the GraphQA / KQAPro / MetaQA ablations.

Emits the format `scripts/prepare_sft_data.py` consumes:

    {"instruction": "<full prompt>", "response": "<expected output>",
     "condition": "direct"}

Run on a LOGIN node (downloads from the HF hub). Usage:
    python scripts/prepare_benchmark_data.py --benchmark gsm8k \
        --output data/gsm8k/hrm-text/train.jsonl
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Iterator

from datasets import load_dataset, get_dataset_config_names

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.benchmarks import MCQDoc, _format_mcq  # noqa: E402
from utils.functions import last_boxed_only_string  # noqa: E402

Pair = tuple[str, str]

# Splits that hold the training data for each benchmark. MMLU has no plain
# "train": `auxiliary_train` is its 99.8k-example training pool.
DEFAULT_SPLIT = {"gsm8k": "train", "math": "train", "mmlu": "auxiliary_train"}


def build_gsm8k(split: str) -> Iterator[Pair]:
    # Eval prompt is the raw question (see benchmarks.GSM8k); the direct answer
    # is the integer after "####", which GSM8k._extract_answer int-parses.
    for item in load_dataset("openai/gsm8k", "main", split=split):
        yield item["question"], item["answer"].split("####")[-1].strip().replace(",", "")


def build_math(split: str) -> Iterator[Pair]:
    # benchmarks.MATH scores with math_verify against last_boxed_only_string(solution),
    # which returns the bare \boxed{...} contents -- so that is the direct response.
    for subset in get_dataset_config_names("EleutherAI/hendrycks_math"):
        for item in load_dataset("EleutherAI/hendrycks_math", subset, split=split):
            boxed = last_boxed_only_string(item["solution"])
            if boxed is not None:
                yield item["problem"], boxed


def build_mmlu(split: str) -> Iterator[Pair]:
    # _format_mcq(include_gold=False) is exactly the per-document unit that
    # benchmarks.MMLU concatenates to build its 5-shot eval prompt.
    for row in load_dataset("cais/mmlu", "all", split=split):
        doc = MCQDoc(row["question"], row["choices"], row["answer"])
        text, _ = _format_mcq(doc, include_gold=False)
        yield text, chr(ord("A") + doc.gold_index)


BUILDERS = {"gsm8k": build_gsm8k, "math": build_math, "mmlu": build_mmlu}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", required=True, choices=sorted(BUILDERS))
    ap.add_argument("--output", required=True, help="Output JSONL path.")
    ap.add_argument("--split", default=None, help=f"Defaults: {DEFAULT_SPLIT}")
    ap.add_argument("--condition", default="direct",
                    help="Condition label written to every record.")
    ap.add_argument("--limit", type=int, default=None, help="Optional cap on samples.")
    args = ap.parse_args()

    split = args.split or DEFAULT_SPLIT[args.benchmark]
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    n, skipped = 0, 0
    with open(out, "w", encoding="utf-8") as f:
        for instruction, response in BUILDERS[args.benchmark](split):
            if args.limit is not None and n >= args.limit:
                break
            # prepare_sft_data.py drops empty responses anyway; drop them here so
            # the reported count matches what gets tokenized.
            if not response.strip():
                skipped += 1
                continue
            f.write(json.dumps({
                "instruction": instruction,
                "response": response,
                "condition": args.condition,
            }, ensure_ascii=False) + "\n")
            n += 1

    print(f"{args.benchmark}[{split}] -> {out}  ({n} samples, {skipped} skipped)")


if __name__ == "__main__":
    main()
