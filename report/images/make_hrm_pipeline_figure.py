"""Pipeline illustration: GraphQA graph -> text serialization -> HRM H/L recurrence.

Faithful to the code paths it depicts:
  * prompt text: a verbatim GraphQA test sample (data/graphqa/hrm-text/standard/test.jsonl);
  * token layout: scripts/prepare_sft_data.py  (<|im_start|> <cond> instr <|im_end|> resp <|box_end|>);
  * attention: models/flash_attention_prefixlm_v2.py (prefix bidirectional, response causal);
  * recurrence: models/baselines/hrm_nocarry_bp_warmup.py (inject_x=False, default H=2, L=3):
        z_H <- x~, z_L <- z_L_init
        for i in 1..H:  for k in cycle i:  z_L <- f_L(z_L + z_H)
                        z_H <- f_H(z_H + z_L)
        logits = LMHead(z_H)
  * gradient window: truncated BPTT after warm-up, bp=5 -> H_bp=min(H, bp-1)=2, L_bp=3
    (last 3 of the H*L=6 L-steps, i.e. only cycle 2).

Usage:
    python report/images/make_hrm_pipeline_figure.py
Writes report/images/hrm_pipeline.{pdf,png,svg}.
"""
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle

OUT_DIR = Path(__file__).resolve().parent

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "savefig.dpi": 300,
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
})

# ----------------------------------------------------------------------------- palette
C_H = "#c0392b"       # high-level module / z_H  (paper accent)
C_H_BG = "#fbeceb"
C_L = "#2c6fbb"       # low-level module / z_L
C_L_BG = "#e9f1fa"
C_X = "#1e8449"       # token embedding x~
C_X_BG = "#e8f5ec"
C_INK = "#222222"
C_MUTED = "#7f8c8d"
C_GRID = "#d5d8dc"
C_PANEL = "#f7f7f8"

# Serialized-text segment colors
C_SCHEMA = "#7f8c8d"
C_NODES = "#8e44ad"
C_EDGES = "#d68910"
C_QUERY = "#2c3e50"
C_RESP = C_H

# ----------------------------------------------------------------------------- canvas
W_IN, H_IN = 7.0, 4.75
U = 20.0                                  # drawing units per inch
fig = plt.figure(figsize=(W_IN, H_IN))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W_IN * U)
ax.set_ylim(0, H_IN * U)
ax.set_aspect("equal")
ax.axis("off")

FS = 7.0       # base font size (pt)


def rbox(x, y, w, h, fc, ec, lw=0.8, ls="-", r=0.8, z=2, hatch=None, alpha=1.0):
    p = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                       fc=fc, ec=ec, lw=lw, ls=ls, zorder=z, hatch=hatch, alpha=alpha)
    ax.add_patch(p)
    return p


def arrow(p0, p1, color=C_INK, lw=0.9, z=3, style="-|>", ms=6, ls="-", cs="arc3,rad=0", shrinkA=0, shrinkB=0):
    a = FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=ms, color=color, lw=lw,
                        zorder=z, linestyle=ls, connectionstyle=cs, shrinkA=shrinkA, shrinkB=shrinkB,
                        joinstyle="miter", capstyle="butt")
    ax.add_patch(a)
    return a


def line(xs, ys, color=C_INK, lw=0.9, z=3, ls="-"):
    ax.plot(xs, ys, color=color, lw=lw, zorder=z, ls=ls, solid_capstyle="butt")


def plus(x, y, color, r=1.15, z=5):
    ax.add_patch(Circle((x, y), r, fc="white", ec=color, lw=0.8, zorder=z))
    ax.plot([x - 0.6 * r, x + 0.6 * r], [y, y], color=color, lw=0.8, zorder=z + 1)
    ax.plot([x, x], [y - 0.6 * r, y + 0.6 * r], color=color, lw=0.8, zorder=z + 1)


def txt(x, y, s, size=FS, color=C_INK, ha="left", va="center", z=6, **kw):
    return ax.text(x, y, s, fontsize=size, color=color, ha=ha, va=va, zorder=z, **kw)


def panel_label(x, y, letter, title):
    txt(x, y, f"({letter})", size=FS + 1.5, weight="bold")
    txt(x + 8.0, y, title, size=FS + 1.5)


# ============================================================================ panel (a)
panel_label(1.5, 92, "a", "Graph-to-text serialization")

# ---- graph (verbatim test sample): edges (0,5)(1,2)(1,3)(2,3)(2,5)(4,5); Q: shortest path 3 -> 4; A: 3
pos = {1: (5.8, 81.0), 3: (5.8, 65.5), 2: (13.8, 73.25), 5: (22.2, 73.25), 0: (30.2, 81.0), 4: (30.2, 65.5)}
edges = [(0, 5), (1, 2), (1, 3), (2, 3), (2, 5), (4, 5)]
path = {(2, 3), (2, 5), (4, 5)}
rbox(1.5, 59.5, 33.0, 28.0, C_PANEL, C_GRID, lw=0.6, z=1)
txt(2.8, 85.5, r"graph $G=(V,E)$", size=FS - 0.8, color=C_MUTED, style="italic")
for u, v in edges:
    on = (min(u, v), max(u, v)) in path
    line([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]],
         color=C_EDGES if on else "#9aa3a8", lw=1.8 if on else 1.0, z=2)
for n, (x, y) in pos.items():
    special = n in (3, 4)
    ax.add_patch(Circle((x, y), 2.4, fc=C_NODES if special else "white", ec=C_NODES, lw=1.0, zorder=4))
    txt(x, y, str(n), size=FS, color="white" if special else C_NODES, ha="center", va="center", weight="bold")
txt(18.0, 61.6, r"shortest path 3$\rightarrow$4 (length 3)", size=FS - 1.6, color=C_MUTED, ha="center", style="italic")

# ---- serialize arrow
arrow((35.6, 73.25), (41.8, 73.25), lw=1.1, ms=8)
txt(38.7, 75.6, "serialize", size=FS - 1.6, color=C_MUTED, ha="center", style="italic")

# ---- serialized prompt (instruction + response, as in the JSONL)
TX0, TY0, TW, TH = 43.0, 59.5, 67.5, 28.0
rbox(TX0, TY0, TW, TH, "white", C_GRID, lw=0.6, z=1)
MONO_FS = 4.95
mono = dict(family="DejaVu Sans Mono", size=MONO_FS)
rows = [
    ("schema", C_SCHEMA, ["In an undirected graph, (i,j) means that node i and",
                          "node j are connected with an undirected edge."]),
    (r"$V$", C_NODES, ["G describes a graph among nodes 0, 1, 2, 3, 4, and 5."]),
    (r"$E$", C_EDGES, ["The edges in G are: (0, 5) (1, 2) (1, 3) (2, 3) (2, 5) (4, 5)."]),
    ("query", C_QUERY, ["Q: What is the length of the shortest path from node 3 to node 4?"]),
]
y = TY0 + TH - 3.6
dy = 3.25
gap = 0.9
for tag, col, lines in rows:
    y_top = y
    for s in lines:
        ax.text(TX0 + 8.0, y, s, color=col, ha="left", va="center", zorder=6, **mono)
        y -= dy
    ax.add_patch(Rectangle((TX0 + 6.4, y + dy - 1.1), 0.55, y_top - (y + dy) + 2.2, fc=col, ec="none", zorder=5))
    txt(TX0 + 5.6, (y_top + y + dy) / 2, tag, size=FS - 1.5, color=col, ha="right", style="italic")
    y -= gap
# "A: " closes the instruction; the response "3." is the training target
ax.text(TX0 + 8.0, y, "A: ", color=C_QUERY, ha="left", va="center", zorder=6, **mono)
ax.text(TX0 + 8.0 + 3 * 0.6 * MONO_FS / 72 * U, y, "3.", color=C_RESP, ha="left", va="center", zorder=6,
        weight="bold", **mono)
ax.add_patch(Rectangle((TX0 + 6.4, y - 1.1), 0.55, 2.2, fc=C_RESP, ec="none", zorder=5))
txt(TX0 + 5.6, y, "target", size=FS - 1.5, color=C_RESP, ha="right", style="italic")

# ---- prefix-LM attention mask (one packed document)
MS = 1.72
P, R = 8, 4
N = P + R
MX0 = 139.0 - N * MS
MY0 = 64.0
txt(MX0 + N * MS / 2, 88.4, "prefix-LM attention", size=FS - 1.2, ha="center", color=C_INK)
for q in range(N):
    for k in range(N):
        allow = (k <= q) or (k < P)
        resp_row = q >= P
        fc = (C_H if resp_row else C_X) if allow else "white"
        ax.add_patch(Rectangle((MX0 + k * MS, MY0 + (N - 1 - q) * MS), MS, MS, fc=fc,
                               ec="white" if allow else C_GRID, lw=0.3, zorder=3, alpha=0.85 if allow else 1.0))
ax.add_patch(Rectangle((MX0, MY0), N * MS, N * MS, fc="none", ec=C_MUTED, lw=0.6, zorder=4))
txt(MX0 + N * MS / 2, MY0 + N * MS + 1.3, "keys", size=FS - 2.2, color=C_MUTED, ha="center")
txt(MX0 - 0.7, MY0 + N * MS / 2, "queries", size=FS - 2.2, color=C_MUTED, ha="right", rotation=90)
ax.add_patch(Rectangle((MX0 - 0.3, MY0 - 2.7), 1.4, 1.4, fc=C_X, ec="none", alpha=0.85, zorder=4))
txt(MX0 + 1.5, MY0 - 2.0, "prefix: bidir.", size=FS - 2.3, color=C_INK)
ax.add_patch(Rectangle((MX0 - 0.3, MY0 - 4.9), 1.4, 1.4, fc=C_H, ec="none", alpha=0.85, zorder=4))
txt(MX0 + 1.5, MY0 - 4.2, "response: causal", size=FS - 2.3, color=C_INK)

# ---- token sequence (scripts/prepare_sft_data.py layout)
SY, SH = 50.0, 4.6
tok_cells = [
    ("<|im_start|>", "sp"), ("<direct>", "sp"), ("In", "i"), ("an", "i"), ("undirected", "i"), ("graph", "i"),
    (",", "i"), ("…", "dots"), ("(", "i"), ("4", "i"), (",", "i"), ("5", "i"), (").", "i"),
    ("Q", "i"), (":", "i"), ("…", "dots"), ("node", "i"), ("4", "i"), ("?", "i"), ("A", "i"), (":", "i"),
    ("<|im_end|>", "sp"), ("3", "r"), (".", "r"), ("<|box_end|>", "spr"),
]
mono_t = dict(family="DejaVu Sans Mono", size=4.3)
cw_char = 0.6 * 4.3 / 72 * U
x = 2.0
spans = {}
for s, kind in tok_cells:
    w = max(2.6, len(s) * cw_char + 1.2)
    if kind == "dots":
        txt(x + w / 2, SY + SH / 2, s, size=FS, color=C_MUTED, ha="center")
    else:
        resp = kind in ("r", "spr")
        fc = C_H_BG if resp else C_X_BG
        ec = C_H if resp else C_X
        rbox(x, SY, w, SH, fc, ec, lw=0.6 if kind in ("i", "r") else 0.9, r=0.5, z=3)
        ax.text(x + w / 2, SY + SH / 2, s, ha="center", va="center", zorder=6,
                color=ec if kind.startswith("sp") else C_INK, **mono_t)
    grp = "r" if kind in ("r", "spr") else "p"
    spans.setdefault(grp, [x, x + w])
    spans[grp][1] = x + w
    x += w + 0.45
txt(x + 0.8, SY + SH / 2, "tokenize, then pack\n" r"($\leq$4096 tokens / step)", size=FS - 1.8,
    color=C_MUTED, ha="left", linespacing=1.3)


def brace_below(x0, x1, yb, label, color):
    line([x0, x0, x1, x1], [yb + 0.9, yb, yb, yb + 0.9], color=color, lw=0.7)
    txt((x0 + x1) / 2, yb - 1.7, label, size=FS - 1.5, color=color, ha="center")


brace_below(spans["p"][0], spans["p"][1], SY - 1.0, "instruction prefix (no loss)", C_X)
brace_below(spans["r"][0], spans["r"][1], SY - 1.0, "response (CE loss)", C_H)

# ============================================================================ panel (b)
panel_label(1.5, 41.0, "b", r"HRM recurrence, unrolled ($H{=}2$ H-cycles $\times$ $L{=}3$ L-steps, default config)")

YH, YL = 28.5, 12.0        # z_H rail, z_L rail
BW, BH = 6.6, 6.2          # block size
RP = 1.15                  # adder radius
YTOP = 35.4

# Embedding (all tokens, teacher-forced)
EX, EY, EW, EH = 2.0, YH - 3.1, 9.0, 6.2
XE = EX + EW / 2
rbox(EX, EY, EW, EH, C_X_BG, C_X, lw=0.9, z=4)
txt(XE, EY + EH / 2 + 0.9, "Embed", size=FS - 0.5, ha="center", color=C_X, weight="bold")
txt(XE, EY + EH / 2 - 1.5, "token ids", size=FS - 2.2, ha="center", color=C_X)
arrow((XE, SY - 1.0), (XE, EY + EH), color=C_X, lw=0.9)

# z_L init
LX, LY, LW, LH = 2.0, YL - 3.1, 9.0, 6.2
rbox(LX, LY, LW, LH, "white", C_L, lw=0.9, ls=(0, (2, 1)), z=4)
txt(LX + LW / 2, LY + LH / 2 + 0.9, r"$z_L^{\mathrm{init}}$", size=FS, ha="center", color=C_L)
txt(LX + LW / 2, LY + LH / 2 - 1.6, "fixed vector", size=FS - 2.2, ha="center", color=C_L)
txt(XE, (YH + YL) / 2, r"$\tilde{x}$ enters only" "\n" r"as $z_H^0$ (no" "\n" r"re-injection)", size=FS - 2.4,
    color=C_MUTED, ha="center", va="center", style="italic", linespacing=1.3)

CW, CGAP, CX0 = 46.0, 4.0, 16.0
grad = {"H": {1: True, 2: True}, "L": {1: False, 2: True}}   # bp=5 -> H_bp=2, L_bp=3
k_global = 0
zh_src = (EX + EW, C_X)     # where the current z_H rail starts, and its color
zl_src = LX + LW            # where the current z_L rail starts
for i in (1, 2):
    cx = CX0 + (i - 1) * (CW + CGAP)
    rbox(cx, 4.2, CW, YTOP - 4.2, "#fbfbfc", C_GRID, lw=0.7, ls=(0, (4, 2)), z=1)
    txt(cx + 1.2, YTOP - 1.6, f"H-cycle {i}", size=FS - 0.5, color=C_MUTED, weight="bold")

    zh_col = zh_src[1]
    hx_plus = cx + 36.6
    # z_H rail, held fixed during the L-steps of this cycle
    line([zh_src[0], hx_plus - RP], [YH, YH], color=zh_col, lw=1.2)
    zh_lab = r"$z_H^{0}{=}\tilde{x}$" if i == 1 else r"$z_H^{1}$"
    txt(cx + 1.2, YH + 2.0, zh_lab, size=FS, color=zh_col, ha="left")
    txt(cx + 1.2, YL + 2.0, r"$z_L^{0}$" if i == 1 else r"$z_L^{3}$", size=FS, color=C_L, ha="left")

    # L-steps: z_L <- f_L(z_L + z_H)
    for j in range(3):
        k_global += 1
        px = cx + 6.0 + j * 10.0
        bx = px + RP + 0.8
        arrow((zl_src, YL), (px - RP, YL), color=C_L, lw=1.2, ms=5)
        ax.add_patch(Circle((px, YH), 0.55, fc=zh_col, ec="none", zorder=5))
        arrow((px, YH), (px, YL + RP), color=zh_col, lw=0.9, ms=5)
        plus(px, YL, C_L, r=RP)
        arrow((px + RP, YL), (bx, YL), color=C_L, lw=1.0, ms=4)
        g = grad["L"][i]
        rbox(bx, YL - BH / 2, BW, BH, C_L if g else C_L_BG, C_L, lw=0.9, ls="-" if g else (0, (2, 1)), z=4)
        txt(bx + BW / 2, YL, r"$f_L$", size=FS + 1.5, color="white" if g else C_L, ha="center")
        zl_src = bx + BW
        if j < 2:
            txt(zl_src + 1.0, YL - 3.0, rf"$z_L^{{{k_global}}}$", size=FS - 1.0, color=C_L, ha="center")

    # last z_L of the cycle -> up into the H adder (and on to the next cycle)
    line([zl_src, hx_plus], [YL, YL], color=C_L, lw=1.2)
    arrow((hx_plus, YL), (hx_plus, YH - RP), color=C_L, lw=0.9, ms=5)
    txt(hx_plus + 0.8, (YH + YL) / 2, rf"$z_L^{{{k_global}}}$", size=FS, color=C_L, ha="left")
    if i == 1:
        ax.add_patch(Circle((hx_plus, YL), 0.55, fc=C_L, ec="none", zorder=5))
    zl_src = hx_plus

    # H-step: z_H <- f_H(z_H + z_L)
    plus(hx_plus, YH, C_H, r=RP)
    hbx = hx_plus + RP + 0.8
    arrow((hx_plus + RP, YH), (hbx, YH), color=C_H, lw=1.0, ms=4)
    g = grad["H"][i]
    rbox(hbx, YH - BH / 2, BW, BH, C_H if g else C_H_BG, C_H, lw=0.9, ls="-" if g else (0, (2, 1)), z=4)
    txt(hbx + BW / 2, YH, r"$f_H$", size=FS + 1.5, color="white" if g else C_H, ha="center")
    zh_src = (hbx + BW, C_H)

# final z_H -> LM head
HX, HY, HW, HH = 117.0, YH - 3.1, 11.0, 6.2
line([zh_src[0], HX], [YH, YH], color=C_H, lw=1.2)
txt((zh_src[0] + HX) / 2, YH + 2.0, r"$z_H^{2}$", size=FS, color=C_H, ha="center")
rbox(HX, HY, HW, HH, "white", C_INK, lw=0.9, z=4)
txt(HX + HW / 2, HY + HH / 2 + 0.9, "LM head", size=FS - 0.5, ha="center", weight="bold")
txt(HX + HW / 2, HY + HH / 2 - 1.5, r"$W_{\mathrm{out}}\,z_H^{H}$", size=FS - 1.6, ha="center")
arrow((HX + HW, YH), (HX + HW + 3.0, YH), lw=0.9, ms=5)
ax.text(HX + HW + 3.5, YH, "3.", color=C_RESP, ha="left", va="center", zorder=6, weight="bold",
        family="DejaVu Sans Mono", size=FS)

# update rules
eq = (r"$z_L^{k} = f_L\!\left(z_L^{k-1} + z_H^{i-1}\right)$" "\n"
      r"$z_H^{i} = f_H\!\left(z_H^{i-1} + z_L^{iL}\right)$")
txt(116.5, 17.0, eq, size=FS - 0.4, ha="left", va="center", linespacing=1.8)
txt(116.5, 8.6, "$k$: L-step (global over cycles)\n$i$: H-cycle", size=FS - 2.2, color=C_MUTED,
    ha="left", va="center", linespacing=1.4)

# legend (bottom)
LGY = 1.7
lx = 16.0
rbox(lx, LGY - 1.0, 3.2, 2.0, C_L, C_L, r=0.4, z=4)
rbox(lx + 3.6, LGY - 1.0, 3.2, 2.0, C_H, C_H, r=0.4, z=4)
txt(lx + 7.6, LGY, "gradient (truncated BPTT after warm-up, bp = 5)", size=FS - 1.8, color=C_INK)
lx2 = 58.0
rbox(lx2, LGY - 1.0, 3.2, 2.0, C_L_BG, C_L, r=0.4, ls=(0, (2, 1)), z=4)
txt(lx2 + 4.0, LGY, "no_grad (gets gradient in the full-BP arm)", size=FS - 1.8, color=C_INK)
txt(94.0, LGY, r"$f_L, f_H$: 6-layer Transformers, weights shared across steps", size=FS - 1.8,
    color=C_MUTED)

for ext in ("pdf", "png", "svg"):
    fig.savefig(OUT_DIR / f"hrm_pipeline.{ext}", dpi=300)
print("wrote", *(OUT_DIR / f"hrm_pipeline.{e}" for e in ("pdf", "png", "svg")))
