"""Pointer chasing: a controlled-depth positive control for the recursion study.

Each sample is a directed graph in which every node has exactly one successor, laid out as a
single random N-cycle (so there are no short cycles, and "follow k edges" cannot be shortcut by
reducing k modulo a small cycle length for k < N). The question asks for the node reached after
k hops from a start node, so the answer needs k sequential lookups: depth is set by k alone.

The prompt mirrors the GraphQA hrm-text phrasing (edge list, "Q: ... A: ", answer "17."), and
edges are listed in shuffled order so that the answer cannot be read off the layout.

With --queries Q > 1, each sample asks for Q distinct start nodes (same k) and the response lists
the Q answers in order ("3, 11, 7, 0."). This multiplies the supervised answer tokens per sample:
with one query, only ~2 of ~425 tokens carry loss and a from-scratch model never left chance (v1).

Output: hrm-text JSONL (instruction / response / condition), plus `k`, `starts`, `answers` and
`graph_id` for the per-k breakdown (scripts/summarize_per_group.py --field k [--per-query]).

Usage:
    python scripts/make_pointer_chasing.py --out data/pointer/hrm-text/standard                 # v1
    python scripts/make_pointer_chasing.py --out data/pointer2/hrm-text/standard --N 16 \
        --ks 1,2,3,4,6,8 --queries 4 --train 30000 --eval 600                                  # v2
"""
import argparse
import json
import os
import random


def make_sample(rng: random.Random, n: int, k: int, queries: int = 1) -> tuple[dict, tuple]:
    order = list(range(n))
    rng.shuffle(order)
    succ = {order[i]: order[(i + 1) % n] for i in range(n)}
    edges = list(succ.items())
    rng.shuffle(edges)
    starts = rng.sample(range(n), queries)

    answers = []
    for start in starts:
        node = start
        for _ in range(k):
            node = succ[node]
        answers.append(node)

    nodes = ", ".join(str(i) for i in range(n - 1)) + f", and {n - 1}"
    if queries == 1:
        question = f"Q: Start at node {starts[0]} and follow {k} edges. Which node do you reach?\nA: "
    else:
        listed = ", ".join(str(a) for a in starts[:-1]) + f" and {starts[-1]}"
        question = (f"Q: Start at each of the nodes {listed} in turn and follow {k} edges. "
                    "Which node do you reach from each?\nA: ")
    instruction = (
        "In a directed graph, (i->j) means that there is a directed edge from node i to node j. "
        f"G describes a graph among nodes {nodes}.\n"
        "The edges in G are: " + " ".join(f"({a}->{b})" for a, b in edges) + ".\n" + question
    )
    key = (tuple(order[order.index(0):] + order[:order.index(0)]), tuple(starts), k)  # cycle up to rotation
    return {"instruction": instruction, "response": ", ".join(str(a) for a in answers) + ".", "condition": "direct",
            "k": k, "starts": starts, "answers": answers}, key


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/pointer/hrm-text/standard")
    ap.add_argument("--N", type=int, default=32)
    ap.add_argument("--ks", default="1,2,3,4,6,8,12,16")
    ap.add_argument("--train", type=int, default=12000)
    ap.add_argument("--eval", type=int, default=800, help="Samples per eval split (balanced over k).")
    ap.add_argument("--queries", type=int, default=1, help="Start nodes asked per graph (same k).")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    ks = [int(k) for k in args.ks.split(",")]
    assert max(ks) < args.N, "k must stay below the cycle length"
    rng = random.Random(args.seed)
    seen: set = set()
    os.makedirs(args.out, exist_ok=True)

    for split, size in (("test", args.eval), ("val", args.eval), ("train", args.train)):
        rows = []
        while len(rows) < size:
            k = ks[len(rows) % len(ks)]  # balanced over k
            row, key = make_sample(rng, args.N, k, args.queries)
            if key in seen:
                continue
            seen.add(key)
            rows.append(row)
        rng.shuffle(rows)
        with open(os.path.join(args.out, f"{split}.jsonl"), "w") as f:
            for i, row in enumerate(rows):
                f.write(json.dumps(row | {"graph_id": f"{split}-{i}"}) + "\n")
        print(f"{split}: {len(rows)} samples -> {args.out}/{split}.jsonl")


if __name__ == "__main__":
    main()
