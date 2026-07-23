"""Convert GraphQA baseline data into HRM-Text SFT JSONL.

Reads the GraphQA baseline JSON files (list of objects with `question` and
`answer` fields) and emits JSONL where each line is:

    {"instruction": "<full prompt>", "response": "<expected output>",
     "condition": "direct"}

The `question` field already contains the fully-framed prompt (schema,
edge list, "Q: ...", trailing "A: "), so it is used verbatim as the
instruction. The resulting JSONL is directly consumable by
`scripts/prepare_sft_data.py`.

Usage:
    python scripts/format_graphqa_to_hrm.py \
        --input-dir data/graphqa/baseline \
        --output-dir data/graphqa/hrm-text
"""
import argparse
import json
from pathlib import Path


def convert_file(src: Path, dst: Path, condition: str) -> int:
    with open(src, encoding="utf-8") as f:
        rows = json.load(f)

    dst.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(dst, "w", encoding="utf-8") as out:
        for r in rows:
            record = {
                "instruction": r["question"],
                "response": r["answer"],
                "condition": condition,
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", default="data/graphqa/baseline",
                    help="Root of the GraphQA baseline data.")
    ap.add_argument("--output-dir", default="data/graphqa/hrm-text",
                    help="Root of the emitted HRM-Text JSONL data.")
    ap.add_argument("--condition", default="direct",
                    help="Condition label written to every record.")
    args = ap.parse_args()

    in_root = Path(args.input_dir)
    out_root = Path(args.output_dir)

    src_files = sorted(in_root.rglob("*.json"))
    if not src_files:
        raise SystemExit(f"No .json files found under {in_root}")

    total = 0
    for src in src_files:
        rel = src.relative_to(in_root).with_suffix(".jsonl")
        dst = out_root / rel
        n = convert_file(src, dst, args.condition)
        total += n
        print(f"{src} -> {dst}  ({n} samples)")

    print(f"Done. Converted {total} samples across {len(src_files)} files.")


if __name__ == "__main__":
    main()
