"""Generate the two hand-made figures for the CS336 assignment-1 blog series.

The other eight figures in the series are W&B exports copied in from the
assignment repo; these two are computed here. Run from the repo root:

    python3 scripts/make_cs336_figures.py
"""

import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"

OUT = pathlib.Path(__file__).resolve().parent.parent / "posts"

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "text.color": INK,
        "axes.labelcolor": INK2,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.edgecolor": AXIS,
        "grid.color": GRID,
    }
)


def strip(ax, left=True, bottom=True):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side, keep in (("left", left), ("bottom", bottom)):
        ax.spines[side].set_visible(keep)
        if keep:
            ax.spines[side].set_linewidth(0.8)
    ax.tick_params(length=0, labelsize=9)


# ---------------------------------------------------------------- figure 1
# GPT-2 XL per-block FLOP share vs. context length.
#   QKVO projections: 8 * seq * d^2
#   attention score/value matmuls: 4 * seq^2 * d
#   SwiGLU FFN: 6 * seq * d * d_ff
d, d_ff = 1600, 4288
seq = np.logspace(np.log2(512), np.log2(32768), 400, base=2)

proj = 8 * seq * d**2
attn = 4 * seq**2 * d
ffn = 6 * seq * d * d_ff
total = proj + attn + ffn

series = [
    ("SwiGLU FFN", 100 * ffn / total, BLUE),
    ("Attention scores + ×V", 100 * attn / total, ORANGE),
    ("QKVO projections", 100 * proj / total, AQUA),
]

fig, ax = plt.subplots(figsize=(8.2, 4.8), dpi=200)
ax.set_axisbelow(True)
ax.grid(axis="y", linewidth=0.7)

for label, y, color in series:
    ax.plot(seq, y, color=color, linewidth=2, solid_capstyle="round", label=label)

# direct labels at the right end (relief for the sub-3:1 aqua, and identity
# never rests on color alone)
for label, y, color in series:
    ax.annotate(
        f"{label}\n{y[-1]:.0f}%",
        xy=(seq[-1], y[-1]),
        xytext=(8, 0),
        textcoords="offset points",
        va="center",
        ha="left",
        fontsize=9,
        color=INK2,
        linespacing=1.35,
    )

# the crossover: attention overtakes the FFN
i = int(np.argmin(np.abs(100 * attn / total - 100 * ffn / total)))
ax.plot(
    [seq[i]], [100 * attn[i] / total[i]], "o", ms=8, color=INK, zorder=6,
    markeredgecolor=SURFACE, markeredgewidth=2,
)
ax.annotate(
    f"attention overtakes the FFN\nat seq ≈ {int(round(seq[i] / 100) * 100):,}",
    xy=(seq[i], 100 * attn[i] / total[i]),
    xytext=(7900, 20.5),
    textcoords="data",
    ha="left",
    va="center",
    fontsize=9,
    color=INK2,
    linespacing=1.35,
    arrowprops=dict(arrowstyle="-", color=AXIS, linewidth=0.9,
                    shrinkA=2, shrinkB=6),
)

ticks = [512, 1024, 2048, 4096, 8192, 16384, 32768]
ax.set_xscale("log", base=2)
ax.set_xticks(ticks)
ax.set_xticklabels([f"{t:,}" for t in ticks])
ax.set_xlim(480, 33000)
ax.set_ylim(0, 80)
ax.set_yticks([0, 20, 40, 60, 80])
ax.set_yticklabels(["0%", "20%", "40%", "60%", "80%"])
ax.set_xlabel("context length (tokens)", fontsize=10)
ax.set_ylabel("share of per-block forward FLOPs", fontsize=10)
ax.set_title(
    "Where a GPT-2 XL block spends its FLOPs, by context length",
    fontsize=12.5,
    color=INK,
    loc="left",
    pad=14,
)
ax.legend(frameon=False, fontsize=9, loc="upper left", bbox_to_anchor=(-0.02, -0.16),
          labelcolor=INK2, handlelength=1.6, ncols=3, columnspacing=2.2,
          borderpad=0, handletextpad=0.6)
strip(ax)
fig.subplots_adjust(left=0.09, right=0.775, top=0.88, bottom=0.24)
fig.savefig(OUT / "llm-from-scratch-2-flops/figures/attention_share_vs_context.png")
plt.close(fig)


# ---------------------------------------------------------------- figure 2
# Ablation ranking: how much validation loss each component is worth.
BASELINE = 1.564
rows = [
    ("no SwiGLU gate (plain SiLU FFN)", 1.642),
    ("no RMSNorm anywhere", 1.715),
    ("post-norm instead of pre-norm", 1.769),
    ("no position embedding (NoPE)", 1.802),
]
labels = [r[0] for r in rows]
deltas = [r[1] - BASELINE for r in rows]
finals = [r[1] for r in rows]

fig, ax = plt.subplots(figsize=(8.2, 3.6), dpi=200)
ax.set_axisbelow(True)
ax.grid(axis="x", linewidth=0.7)

y = np.arange(len(rows))
ax.barh(y, deltas, height=0.5, color=BLUE, zorder=3)

for yi, dv, fv in zip(y, deltas, finals):
    ax.annotate(
        f"+{dv:.2f}   (val loss {fv:.3f})",
        xy=(dv, yi),
        xytext=(8, 0),
        textcoords="offset points",
        va="center",
        ha="left",
        fontsize=9,
        color=INK2,
    )

ax.set_yticks(y)
ax.set_yticklabels(labels, fontsize=10, color=INK2)
ax.set_xlim(0, 0.36)
ax.set_xticks([0, 0.05, 0.10, 0.15, 0.20, 0.25])
ax.set_xlabel("increase in final validation loss vs. the 1.564 baseline", fontsize=10)
ax.set_title(
    "What each architecture choice is worth on TinyStories",
    fontsize=12.5,
    color=INK,
    loc="left",
    pad=14,
)
strip(ax, left=False)
fig.subplots_adjust(left=0.30, right=0.98, top=0.83, bottom=0.19)
fig.savefig(OUT / "llm-from-scratch-3-training-dynamics/figures/ablation_ranking.png")
plt.close(fig)

print("wrote both figures into posts/*/figures/")
