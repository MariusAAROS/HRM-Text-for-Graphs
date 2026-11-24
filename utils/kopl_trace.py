"""
KoPL Program Tracing on Prompt Graphs
=====================================

Runs a KQA Pro KoPL program (``function`` / ``inputs`` / ``dependencies``) over
the facts written in a prompt and records which facts each step relies on. The
result is the evidence subgraph needed to answer the question: the facts to
highlight when plotting question entities, answer and reasoning edges.

Unlike MetaQA, a KQA Pro answer is often not a graph node (relation labels,
qualifier values, counts, yes/no), so a question→answer shortest path does not
capture the reasoning. Following the program does.

The executor is approximate: values are compared with lenient parsers
(numbers ignore units, dates compare on their common Y/M/D prefix), and
constraints the prompt cannot check are skipped: ``FilterConcept`` passes its
input through when no ``instance of`` fact matches, and ``And`` returns the
other branch when one branch is empty.

Usage
-----
    from utils.kopl_trace import parse_facts, trace_kopl

    facts = parse_facts(sample["question"])            # prompt text
    trace = trace_kopl(original["program"], facts, original["answer"])
    trace["evidence"]       # facts used by the program
    trace["answer_kind"]    # node / relation / qualifier / attribute / count / boolean
"""

import re
from collections import defaultdict, namedtuple

# ----------------------------------------------------------------------
# Prompt parsing
# ----------------------------------------------------------------------

# (s, p, o) [k1: v1; k2: v2] — qualifiers are optional
FACT_RE = re.compile(r"^\((.+?), (.+?), (.+)\)(?: \[(.*)\])?$")

Fact = namedtuple("Fact", "s p o quals")


def parse_facts(prompt):
    """Parse every fact line of a KQA prompt into Fact(s, p, o, quals), quals = ((k, v), ...)."""
    facts = []
    for line in prompt.split("\n"):
        m = FACT_RE.match(line.strip())
        if not m:
            continue
        s, p, o, q = m.groups()
        quals = tuple(tuple(kv.split(": ", 1)) for kv in q.split("; ") if ": " in kv) if q else ()
        facts.append(Fact(s, p, o, quals))
    return facts


def fact_str(f):
    return f"({f.s}, {f.p}, {f.o})"


# ----------------------------------------------------------------------
# Value comparison (KoPL uses ISO dates and unit-bearing numbers)
# ----------------------------------------------------------------------

_NUM_RE = re.compile(r"^-?\d+(?:\.\d+)?(?:e[+-]?\d+)?", re.IGNORECASE)
_DATE_RE = re.compile(r"^(-?\d+)(?:[/-](\d+))?(?:[/-](\d+))?$")


def _num(v):
    m = _NUM_RE.match(v.strip())
    return float(m.group()) if m else None


def _year(v):
    m = re.match(r"^-?\d+", v.strip())
    return int(m.group()) if m else None


def _date(v):
    m = _DATE_RE.match(v.strip())
    return tuple(int(x) for x in m.groups() if x is not None) if m else None


PARSERS = {"str": str.strip, "num": _num, "year": _year, "date": _date}


def _match(value, target, op="=", kind="str"):
    a, b = PARSERS[kind](value), PARSERS[kind](target)
    if a is None or b is None:
        return False
    if kind == "date":
        n = min(len(a), len(b))
        a, b = a[:n], b[:n]
    return {"=": a == b, "!=": a != b, "<": a < b, ">": a > b}[op]


def _loose_eq(value, target):
    """Equality across formats: exact string, then date, then number."""
    return any(_match(value, target, "=", kind) for kind in ("str", "date", "num"))


def _sort_key(value):
    d = _date(value)
    return d if d is not None and len(d) > 1 else (_num(value),)


# ----------------------------------------------------------------------
# Executor
# ----------------------------------------------------------------------
# Entity steps return {entity: (prov, last)}: `prov` is every fact justifying
# the entity, `last` the facts added by the latest step (what QFilter* prunes).
# Value steps return [(value, prov, detail)].

EMPTY = frozenset()

FINAL_KIND = {
    "What": "node", "SelectBetween": "node", "SelectAmong": "node",
    "QueryRelation": "relation", "QueryRelationQualifier": "qualifier",
    "QueryAttrQualifier": "qualifier", "QueryAttr": "attribute",
    "QueryAttrUnderCondition": "attribute", "Count": "count",
    "VerifyStr": "boolean", "VerifyNum": "boolean", "VerifyYear": "boolean", "VerifyDate": "boolean",
}

FILTER_KIND = {"FilterStr": "str", "FilterNum": "num", "FilterYear": "year", "FilterDate": "date",
               "QFilterStr": "str", "QFilterNum": "num", "QFilterYear": "year", "QFilterDate": "date",
               "VerifyStr": "str", "VerifyNum": "num", "VerifyYear": "year", "VerifyDate": "date"}


class _Executor:
    def __init__(self, facts):
        self.nodes = {f.s for f in facts} | {f.o for f in facts}
        self.by_s, self.by_o = defaultdict(list), defaultdict(list)
        for f in facts:
            self.by_s[f.s].append(f)
            self.by_o[f.o].append(f)

    # -- entity steps --------------------------------------------------

    def FindAll(self, inputs):
        return {e: (EMPTY, EMPTY) for e in self.nodes}

    def Find(self, inputs):
        return {inputs[0]: (EMPTY, EMPTY)} if inputs[0] in self.nodes else {}

    def FilterConcept(self, inputs, ents):
        out = {}
        for e, (prov, last) in ents.items():
            hit = next((f for f in self.by_s[e] if f.p == "instance of" and f.o == inputs[0]), None)
            if hit:
                out[e] = (prov | {hit}, last)
        return out or ents  # no type facts in the prompt: cannot filter

    def _filter(self, kind, inputs, ents):
        key, target = inputs[0], inputs[1]
        op = inputs[2] if len(inputs) > 2 else "="
        out = {}
        for e, (prov, _) in ents.items():
            hits = frozenset(f for f in self.by_s[e] if f.p == key and _match(f.o, target, op, kind))
            if hits:
                out[e] = (prov | hits, hits)
        return out

    def _qfilter(self, kind, inputs, ents):
        qkey, target = inputs[0], inputs[1]
        op = inputs[2] if len(inputs) > 2 else "="
        out = {}
        for e, (prov, last) in ents.items():
            keep = frozenset(f for f in last
                             if any(k == qkey and _match(v, target, op, kind) for k, v in f.quals))
            if keep:
                out[e] = ((prov - last) | keep, keep)
        return out

    def Relate(self, inputs, ents):
        rel, direction = inputs
        out = {}
        for e, (prov, _) in ents.items():
            if direction == "forward":
                hits = [(f, f.o) for f in self.by_s[e] if f.p == rel]
            else:
                hits = [(f, f.s) for f in self.by_o[e] if f.p == rel]
            for f, t in hits:
                p0, l0 = out.get(t, (EMPTY, EMPTY))
                out[t] = (p0 | prov | {f}, l0 | {f})
        return out

    def And(self, inputs, a, b):
        if not a or not b:  # constraint the prompt cannot check: skip it
            return a or b
        return {e: (a[e][0] | b[e][0], a[e][1] | b[e][1]) for e in a.keys() & b.keys()}

    def Or(self, inputs, a, b):
        out = dict(a)
        for e, (prov, last) in b.items():
            p0, l0 = out.get(e, (EMPTY, EMPTY))
            out[e] = (p0 | prov, l0 | last)
        return out

    # -- value steps ---------------------------------------------------

    def What(self, inputs, ents):
        return [(e, prov, "graph node") for e, (prov, _) in ents.items()]

    def Count(self, inputs, ents):
        prov = frozenset().union(*(p for p, _ in ents.values()))
        return [(str(len(ents)), prov, f"count of {len(ents)} entities")]

    def QueryAttr(self, inputs, ents):
        return [(f.o, prov | {f}, f"value of {fact_str(f)}")
                for e, (prov, _) in ents.items() for f in self.by_s[e] if f.p == inputs[0]]

    def QueryAttrUnderCondition(self, inputs, ents):
        key, qkey, qval = inputs
        return [(f.o, prov | {f}, f"value of {fact_str(f)} with {qkey}: {qval}")
                for e, (prov, _) in ents.items() for f in self.by_s[e]
                if f.p == key and any(k == qkey and _loose_eq(v, qval) for k, v in f.quals)]

    def QueryAttrQualifier(self, inputs, ents):
        key, value, qkey = inputs
        return [(v, prov | {f}, f"qualifier '{qkey}' of {fact_str(f)}")
                for e, (prov, _) in ents.items() for f in self.by_s[e]
                if f.p == key and _loose_eq(f.o, value) for k, v in f.quals if k == qkey]

    def _between(self, a, b, rel=None):
        """Facts linking an entity of `a` to an entity of `b` (either direction), optionally with label `rel`."""
        for e, (pa, _) in a.items():
            for f in self.by_s[e] + self.by_o[e]:
                other = f.o if f.s == e else f.s
                if other in b and other != e and (rel is None or f.p == rel):
                    yield f, pa | b[other][0] | {f}

    def QueryRelation(self, inputs, a, b):
        return [(f.p, prov, f"relation label of {fact_str(f)}") for f, prov in self._between(a, b)]

    def QueryRelationQualifier(self, inputs, a, b):
        rel, qkey = inputs
        return [(v, prov, f"qualifier '{qkey}' of {fact_str(f)}")
                for f, prov in self._between(a, b, rel) for k, v in f.quals if k == qkey]

    def _select(self, key, largest, ents, what):
        cands = [(f, e, prov | {f}) for e, (prov, _) in ents.items()
                 for f in self.by_s[e] if f.p == key and _num(f.o) is not None]
        if not cands:
            return []
        f, e, _ = (max if largest else min)(cands, key=lambda c: _sort_key(c[0].o))
        prov = frozenset().union(*(p for _, _, p in cands))  # every compared value is evidence
        return [(e, prov, f"{what} '{key}' over {len(cands)} values")]

    def SelectBetween(self, inputs, a, b):
        return self._select(inputs[0], inputs[1] == "greater", self.Or([], a, b), "compare")

    def SelectAmong(self, inputs, ents):
        return self._select(inputs[0], inputs[1] == "largest", ents, "arg-" + inputs[1])

    def _verify(self, kind, inputs, values):
        op = inputs[1] if len(inputs) > 1 else "="
        prov = frozenset().union(*(p for _, p, _ in values))
        ok = any(_match(v, inputs[0], op, kind) for v, _, _ in values)
        return [("yes" if ok else "no", prov, f"check {op} {inputs[0]}")]

    def run_step(self, fn, inputs, args):
        if fn.startswith("QFilter"):
            return self._qfilter(FILTER_KIND[fn], inputs, *args)
        if fn.startswith("Filter") and fn != "FilterConcept":
            return self._filter(FILTER_KIND[fn], inputs, *args)
        if fn.startswith("Verify"):
            return self._verify(FILTER_KIND[fn], inputs, *args)
        if not hasattr(self, fn):
            raise NotImplementedError(f"KoPL function {fn!r}")
        return getattr(self, fn)(inputs, *args)


def program_str(program, sep=" → "):
    return sep.join(f"{fn}({', '.join(inp)})" for fn, inp in zip(program["function"], program["inputs"]))


def trace_kopl(program, facts, answer=None):
    """
    Run a KoPL program over prompt facts and return the evidence it uses.

    Returns a dict with:
        "question_entities" : names passed to Find (topic entities)
        "evidence"          : sorted list of Fact used to reach the answer
        "answer"            : gold answer (or the traced one if `answer` is None)
        "answer_kind"       : node / relation / qualifier / attribute / count / boolean
        "answer_traced"     : whether the gold answer is among the traced values
        "answer_detail"     : where the answer comes from, e.g. "qualifier 'start time' of (A, employer, B)"
        "program_str"       : compact program, "Find(X) → Relate(r, forward) → ..."
    """
    ex = _Executor(facts)
    results = []
    for fn, inputs, deps in zip(program["function"], program["inputs"], program["dependencies"]):
        results.append(ex.run_step(fn, inputs, [results[d] for d in deps]))

    values = results[-1]
    if isinstance(values, dict):  # program ends on an entity set
        values = ex.What([], values)
    matched = [v for v in values if answer is not None and _loose_eq(v[0], answer)]
    chosen = matched or values
    evidence = frozenset().union(*(p for _, p, _ in chosen))

    return {
        "question_entities": [inp[0] for fn, inp in zip(program["function"], program["inputs"]) if fn == "Find"],
        "evidence": sorted(evidence),
        "answer": answer if answer is not None else (values[0][0] if values else None),
        "answer_kind": FINAL_KIND.get(program["function"][-1], "node"),
        "answer_traced": bool(matched),
        "answer_detail": chosen[0][2] if chosen else "not found in prompt",
        "program_str": program_str(program),
    }
