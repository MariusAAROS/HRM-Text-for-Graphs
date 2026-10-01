"""Dataset statistics for GraphQA, MetaQA (Gold-1/5/10, Retrieved-1) and KQA-Pro.

Every number in the generated LaTeX tables is computed here from the files under data/:
the `baseline/standard/{train,val,test}.json` splits (Meta-ICL and HRM-Text share the same
questions) and the two source knowledge bases. Lengths are counted with the HRM-Text-1B
tokenizer on the raw prompt (`question`) and target (`answer`) strings.

Usage:  python report/dataset_stats.py
Writes: report/dataset_stats.tex        (full-width `table*`, one column per dataset)
        report/dataset_stats_column.tex (single-column `table`, one row per dataset)
"""

import json
from collections import Counter
from pathlib import Path

import numpy as np
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "report"
SPLITS = ["train", "val", "test"]
FACTS_HEADER = "The facts are:\n"

# (column label, directory under data/). Order = column order in the table.
DATASETS = [
    ("GraphQA", "graphqa"),
    ("Gold-1", "metaqa-gold-1"),
    ("Gold-5", "metaqa-gold-5"),
    ("Gold-10", "metaqa-gold-10"),
    ("Retrieved-1", "metaqa-retrieved-1"),
    ("KQA-Pro", "kqapro"),
]
METAQA = [label for label, d in DATASETS if d.startswith("metaqa")]

tokenizer = Tokenizer.from_file(hf_hub_download("sapientinc/HRM-Text-1B", "tokenizer.json"))


def load_splits(name):
    return {
        sp: json.loads((DATA_DIR / name / "baseline" / "standard" / f"{sp}.json").read_text())
        for sp in SPLITS
    }


def n_tokens(texts):
    encs = tokenizer.encode_batch(texts, add_special_tokens=False)
    return np.array([len(e.ids) for e in encs])


def answer_items(answer):
    """MetaQA set answers are ' | '-joined and end with a period."""
    return [a.strip() for a in answer.rstrip().removesuffix(".").split(" | ")]


def answer_in_context(record):
    """Fraction of the gold answer entities that appear as an entity in the given facts."""
    facts = record["question"].split(FACTS_HEADER, 1)[1].rsplit("\nQ: ", 1)[0]
    items = answer_items(record["answer"])
    return np.mean([f", {a})" in facts or f"({a}, " in facts for a in items])


def dataset_stats(name):
    splits = load_splits(name)
    records = [r for sp in SPLITS for r in splits[sp]]
    stats = {
        "n": {sp: len(splits[sp]) for sp in SPLITS},
        "nodes": np.array([int(r["nnodes"]) for r in records]),
        "edges": np.array([int(r["nedges"]) for r in records]),
        "in_tok": n_tokens([r["question"] for r in records]),
        "out_tok": n_tokens([r["answer"] for r in records]),
        "tasks": Counter(r["task"] for r in records),
        "generators": Counter(r["algorithm"] for r in records),
        # KQA-Pro pairs most questions with two contexts, so count distinct questions too.
        "unique": {sp: len({r["question"].rsplit("\nQ: ", 1)[1] for r in splits[sp]})
                   for sp in SPLITS},
    }
    if "nsteps" in records[0]:
        stats["steps"] = np.array([int(r["nsteps"]) for r in records])
    if "nanswers" in records[0]:
        stats["answers"] = np.array([int(r["nanswers"]) for r in records])
        stats["recall"] = np.mean([answer_in_context(r) for r in records])
    else:
        stats["answers"] = np.ones(len(records), dtype=int)
    stats["empty_ctx"] = np.mean(stats["edges"] == 0)
    return stats


def metaqa_kb():
    triples = [l.rstrip("\n").split("|") for l in open(DATA_DIR / "metaqa" / "kb.txt")]
    entities = {t[0] for t in triples} | {t[2] for t in triples}
    return {"entities": len(entities), "relations": len({t[1] for t in triples}),
            "triples": len(triples)}


def kqapro_kb():
    kb = json.loads((DATA_DIR / "kqapro" / "kb.json").read_text())
    ents = kb["entities"].values()
    return {"entities": len(kb["entities"]), "concepts": len(kb["concepts"]),
            "relations": len({r["predicate"] for e in ents for r in e["relations"]}),
            "attributes": len({a["key"] for e in ents for a in e["attributes"]})}


def graphqa_ood_counts():
    ood = DATA_DIR / "graphqa" / "baseline" / "ood"
    return {sp: len(json.loads((ood / f"{sp}.json").read_text())) for sp in SPLITS}


# ----------------------------------------------------------------------------- formatting
def k(x):
    """Compact thousands for KB sizes: 43234 -> 43k."""
    return f"{x / 1000:.0f}k" if x >= 10_000 else f"{x:,}"


def pm(a, digits=1):
    return f"${a.mean():.{digits}f}\\pm{a.std():.{digits}f}$"


def row(label, cells):
    return f"{label} & " + " & ".join(cells) + r" \\"


def build_wide(S, kb_meta, kb_kqa, ood):
    labels = [label for label, _ in DATASETS]
    g, kq = S["GraphQA"], S["KQA-Pro"]
    per = lambda f: [f(S[l]) for l in labels]
    na = lambda f, key: (lambda s: f(s) if key in s else "--")
    kq_gold = 100 * kq["generators"]["gold_1hop"] / sum(kq["generators"].values())

    L = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{5pt}",
        r"\resizebox{\textwidth}{!}{%",  # needs \usepackage{graphicx}
        r"\begin{tabular}{@{}l c cccc c@{}}",
        r"\toprule",
        r" & & \multicolumn{4}{c}{\textbf{MetaQA}} & \\",
        r"\cmidrule(lr){3-6}",
        row("", [r"\textbf{GraphQA}", "Gold-1", "Gold-5", "Gold-10", "Retrieved-1",
                 r"\textbf{KQA-Pro}"]),
        r"\midrule",
        row("Knowledge base", ["Synthetic", r"\multicolumn{4}{c}{WikiMovies}", "Wikidata"]),
        row("KB entities", ["--", rf"\multicolumn{{4}}{{c}}{{{k(kb_meta['entities'])}}}",
                            k(kb_kqa["entities"])]),
        row("Context", ["Whole graph", r"\multicolumn{3}{c}{Gold subgraph}", "Retrieved",
                        "Mixed"]),
        row("Categories", [f"{len(g['tasks'])} tasks", r"\multicolumn{4}{c}{1-, 2-, 3-hop}",
                           f"{len(kq['tasks'])} types"]),
        r"\midrule",
        *[row(f"{sp.capitalize()}", per(lambda s, sp=sp: f"{s['n'][sp]:,}"))
          for sp in SPLITS],
        r"\midrule",
        row("Context nodes", per(lambda s: pm(s["nodes"], 0))),
        row("Context edges", per(lambda s: pm(s["edges"], 0))),
        row("Reasoning steps", per(na(lambda s: pm(s["steps"]), "steps"))),
        row("Answers", per(lambda s: f"{s['answers'].mean():.0f}")),
        row(r"Answer in context (\%)",
            per(na(lambda s: f"{100 * s['recall']:.1f}", "recall"))),
        r"\midrule",
        row("Input tokens", per(lambda s: pm(s["in_tok"], 0))),
        row("Input tokens (max)", per(lambda s: f"{s['in_tok'].max():,}")),
        row("Output tokens", per(lambda s: pm(s["out_tok"], 0))),
        r"\bottomrule",
        r"\end{tabular}%",
        r"}",
        r"\caption{Dataset statistics. Train/val/test give the number of questions; the "
        r"remaining rows are computed over all three splits and report mean$\pm$std. "
        rf"GraphQA graphs come from {len(g['generators'])} random-graph generators; the "
        rf"WikiMovies KB has {kb_meta['relations']} relations and {k(kb_meta['triples'])} "
        rf"triples. Context nodes and edges are the entities and "
        r"facts serialized in the prompt (for GraphQA, the whole graph). Reasoning steps are "
        r"hops for MetaQA and KoPL program length for KQA-Pro. Tokens are counted with the "
        r"HRM-Text tokenizer. "
        r"Most KQA-Pro questions appear twice, once with a gold 1-hop context and once with a "
        rf"retrieved 2-hop one ({kq_gold:.0f}\% gold; {kq['unique']['train']:,}/"
        rf"{kq['unique']['val']:,}/{kq['unique']['test']:,} distinct questions); "
        rf"{100 * kq['empty_ctx']:.1f}\% of its contexts are empty. "
        r"The Meta-ICL version of each dataset adds 32 demonstrations per question. GraphQA "
        rf"also has a leave-one-task-out split ({ood['train']:,}/{ood['val']:,}/"
        rf"{ood['test']:,} questions).}}",
        r"\label{tab:dataset_stats}",
        r"\end{table*}",
    ]
    return "\n".join(L)


def build_column(S):
    """One row per dataset, so the table fits a single column."""
    L = [
        r"\begin{table}[t]",
        r"\centering",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{2.5pt}",
        r"\begin{tabular}{@{}l rrr rr r@{}}",
        r"\toprule",
        r" & \multicolumn{3}{c}{\textbf{Questions}} & \multicolumn{2}{c}{\textbf{Context}}"
        r" & \\",
        r"\cmidrule(lr){2-4}\cmidrule(lr){5-6}",
        row("Dataset", ["Train", "Val", "Test", "Nodes", "Edges", "Tokens"]),
        r"\midrule",
    ]
    for label, _ in DATASETS:
        s = S[label]
        if label == METAQA[0]:
            L.append(r"MetaQA & & & & & & \\")
        name = rf"\hspace{{0.6em}}{label}" if label in METAQA else label
        L.append(row(name, [
            f"{s['n']['train']:,}", f"{s['n']['val']:,}", f"{s['n']['test']:,}",
            f"{s['nodes'].mean():.0f}", f"{s['edges'].mean():.0f}",
            f"{s['in_tok'].mean():,.0f}",
        ]))
    L += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{Dataset statistics: number of questions per split, mean number of "
        r"context nodes and edges serialized in the prompt, and mean prompt length in "
        r"HRM-Text tokens. Gold-$k$ questions have exactly $k$ answers; all other questions "
        r"have one.}",
        r"\label{tab:dataset_stats_column}",
        r"\end{table}",
    ]
    return "\n".join(L)


if __name__ == "__main__":
    S = {label: dataset_stats(name) for label, name in DATASETS}
    kb_meta, kb_kqa, ood = metaqa_kb(), kqapro_kb(), graphqa_ood_counts()

    for label, s in S.items():
        print(f"{label:>11}  n={s['n']}  nodes={s['nodes'].mean():.1f}  "
              f"edges={s['edges'].mean():.1f}  in_tok={s['in_tok'].mean():.0f} "
              f"(max {s['in_tok'].max()})  out_tok={s['out_tok'].mean():.1f}  "
              f"recall={s.get('recall', float('nan')):.3f}  empty={s['empty_ctx']:.3f}")
    print("MetaQA KB:", kb_meta, "\nKQA-Pro KB:", kb_kqa, "\nGraphQA OOD:", ood)

    (OUT_DIR / "dataset_stats.tex").write_text(build_wide(S, kb_meta, kb_kqa, ood) + "\n")
    (OUT_DIR / "dataset_stats_column.tex").write_text(build_column(S) + "\n")
    print("wrote", OUT_DIR / "dataset_stats.tex", "and", OUT_DIR / "dataset_stats_column.tex")
