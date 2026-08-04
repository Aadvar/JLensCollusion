"""
fig.py — Ablation figure in the NARCBench visual style.

Horizontal grouped bars, solid = one setting, hatched = the other,
one panel per model, value labels at bar ends.

Edit RESULTS as each model finishes. Only llama3.1-8b-it has real numbers;
the rest are None and render as an empty panel with 'pending'.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# NUMBERS.  transfer = train on committee domains -> test on procurement
#           indist   = COVERT vs HONEST, clean honests, same layer
# Layer 6 (k=8) for llama: the layer where workspace exceeded random by >3 SD.
# ---------------------------------------------------------------------------

ARMS = ["full", "workspace", "residue_k", "random_k"]

RESULTS = {
    "llama3.1-8b-it": {
        "layer": 6, "k": 8,
        "transfer": {"full": 0.811, "workspace": 0.779,
                     "residue_k": 0.545, "random_k": 0.604},
        "indist":   {"full": 0.970, "workspace": 0.736,
                     "residue_k": 0.750, "random_k": 0.727},
    },
    "qwen3.5-27b":    None,
    "gemma-3-27b-it": None,
    "gpt-oss-20b":    None,
}

COLORS = {
    "llama3.1-8b-it": "#2E7CF6",
    "qwen3.5-27b":    "#7B3FE4",
    "gemma-3-27b-it": "#2FA37C",
    "gpt-oss-20b":    "#4B4DED",
}

LABEL = {"full": "Full activations", "workspace": "Workspace (J-space)",
         "residue_k": "Complement (matched k)", "random_k": "Random subspace"}

# ---------------------------------------------------------------------------

fig, axes = plt.subplots(2, 2, figsize=(13, 7.2))
fig.patch.set_facecolor("white")

for ax, (model, res) in zip(axes.flat, RESULTS.items()):
    color = COLORS[model]
    y = np.arange(len(ARMS))[::-1]
    h = 0.36

    if res is None:
        ax.text(0.5, 0.5, "pending", ha="center", va="center",
                fontsize=13, color="#999", transform=ax.transAxes)
        ax.set_yticks(y); ax.set_yticklabels([LABEL[a] for a in ARMS], fontsize=9)
        ax.set_xlim(0, 1.12); ax.set_xticks([0, .25, .5, .75, 1.0])
        ax.set_xticklabels(["0.00", "0.25", "0.50", "0.75", "1.00"], fontsize=9)
    else:
        tr = [res["transfer"][a] for a in ARMS]
        ind = [res["indist"][a] for a in ARMS]

        ax.barh(y + h/2, tr, h, color=color, edgecolor=color, zorder=3)
        ax.barh(y - h/2, ind, h, facecolor="white", edgecolor=color,
                hatch="///", linewidth=1.1, zorder=3)

        for yy, v in zip(y + h/2, tr):
            ax.text(v + .012, yy, f"{v:.2f}", va="center", fontsize=8.5)
        for yy, v in zip(y - h/2, ind):
            ax.text(v + .012, yy, f"{v:.2f}", va="center", fontsize=8.5)

        ax.axvline(0.5, color="#bbb", lw=.8, ls=":", zorder=1)
        ax.set_yticks(y); ax.set_yticklabels([LABEL[a] for a in ARMS], fontsize=9)
        ax.set_xlim(0, 1.12); ax.set_xticks([0, .25, .5, .75, 1.0])
        ax.set_xticklabels(["0.00", "0.25", "0.50", "0.75", "1.00"], fontsize=9)
        ax.text(0.99, 1.04, f"layer {res['layer']}, k={res['k']}",
                transform=ax.transAxes, ha="right", fontsize=8.5, color="#666")

    ax.set_xlabel("AUROC", fontsize=10)
    ax.set_ylabel(model, fontsize=11, fontweight="bold", labelpad=12)
    ax.grid(axis="x", color="#e8e8e8", lw=.7, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color("#ccc")
    ax.tick_params(length=0)

# legend
from matplotlib.patches import Patch
fig.legend(handles=[
    Patch(facecolor="#555", edgecolor="#555", label="Transfer (committee → procurement)"),
    Patch(facecolor="white", edgecolor="#555", hatch="///", label="In-distribution")],
    loc="lower center", ncol=2, frameon=False, fontsize=10,
    bbox_to_anchor=(0.5, -0.005))

fig.suptitle("Collusion signal by activation subspace",
             fontsize=13, fontweight="bold", y=0.98)
fig.text(0.5, 0.937, "Same probe, same labels, same CV splits — only the input "
         "subspace varies. Workspace and complement are dimension-matched.",
         ha="center", fontsize=9, color="#666")

plt.tight_layout(rect=[0, 0.035, 1, 0.925])
plt.savefig("/mnt/user-data/outputs/ablation_figure.png", dpi=200,
            facecolor="white", bbox_inches="tight")
print("saved")
