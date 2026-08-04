"""
fig_null.py — The honest figure for a null result.

Workspace minus random, across layers, three models, with the +/-1 SD band of
the random-draw distribution. If J-space carried special signal, the lines
would sit consistently above zero. They don't.

Fixed k=64 for every model and layer, so the comparison is capacity-matched.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

LAYERS = [2, 4, 6, 8, 10, 12, 15, 20, 25, 30]

DATA = {
    "llama3.1-8b-it (32 blocks)": {
        "c": "#2E7CF6",
        "ws":   [0.527, 0.709, 0.779, 0.787, 0.846, 0.834, 0.861, 0.697, 0.605, 0.664],
        "rnd":  [0.599, 0.657, 0.739, 0.740, 0.844, 0.872, 0.858, 0.812, 0.801, 0.759],
        "sd":   [0.081, 0.077, 0.072, 0.088, 0.056, 0.086, 0.085, 0.096, 0.119, 0.112],
        "res":  [0.521, 0.693, 0.688, 0.750, 0.803, 0.863, 0.594, 0.748, 0.688, 0.783],
        "full": [0.725, 0.766, 0.811, 0.916, 0.965, 0.969, 0.979, 0.953, 0.965, 0.957],
    },
    "qwen3.5-27b (64 blocks)": {
        "c": "#7B3FE4",
        "ws":   [0.607, 0.670, 0.773, 0.895, 0.744, 0.723, 0.588, 0.689, 0.775, 0.830],
        "rnd":  [0.613, 0.716, 0.712, 0.752, 0.710, 0.710, 0.688, 0.780, 0.859, 0.968],
        "sd":   [0.113, 0.079, 0.085, 0.081, 0.091, 0.097, 0.104, 0.126, 0.080, 0.046],
        "res":  [0.721, 0.680, 0.715, 0.697, 0.631, 0.529, 0.627, 0.740, 0.918, 0.990],
        "full": [0.680, 0.840, 0.848, 0.879, 0.842, 0.857, 0.822, 0.965, 0.982, 1.000],
    },
    "gemma-3-27b-it (62 blocks)": {
        "c": "#2FA37C",
        "ws":   [0.527, 0.551, 0.584, 0.490, 0.506, 0.574, 0.652, 0.777, 0.867, 0.963],
        "rnd":  [0.664, 0.667, 0.682, 0.632, 0.606, 0.637, 0.649, 0.757, 0.857, 0.844],
        "sd":   [0.060, 0.061, 0.058, 0.079, 0.078, 0.107, 0.080, 0.084, 0.083, 0.109],
        "res":  [0.504, 0.658, 0.510, 0.588, 0.615, 0.807, 0.646, 0.488, 0.807, 0.930],
        "full": [0.605, 0.629, 0.707, 0.719, 0.646, 0.693, 0.699, 0.924, 0.977, 0.994],
    },
}

fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2),
                         gridspec_kw={"width_ratios": [1.15, 1]})
fig.patch.set_facecolor("white")

# ---- LEFT: workspace advantage over random ------------------------------
ax = axes[0]
x = np.arange(len(LAYERS))
band = np.mean([d["sd"] for d in DATA.values()], axis=0)
ax.fill_between(x, -band, band, color="#ddd", alpha=.55, zorder=1,
                label="±1 SD of random draws")
ax.axhline(0, color="#555", lw=1.1, zorder=2)

for name, d in DATA.items():
    gap = np.array(d["ws"]) - np.array(d["rnd"])
    ax.plot(x, gap, "-o", color=d["c"], ms=5, lw=1.9, zorder=3, label=name)

ax.set_xticks(x); ax.set_xticklabels(LAYERS)
ax.set_xlabel("layer", fontsize=10)
ax.set_ylabel("AUROC(workspace) − AUROC(random)", fontsize=10)
ax.set_title("No consistent J-space advantage", fontsize=11.5, fontweight="bold")
ax.set_ylim(-0.28, 0.28)
ax.grid(color="#eee", lw=.7, zorder=0); ax.set_axisbelow(True)
for s in ("top", "right"): ax.spines[s].set_visible(False)
ax.legend(fontsize=8.3, frameon=False, loc="lower right")
ax.text(.02, .96, "above 0 = workspace beats random", transform=ax.transAxes,
        fontsize=8, color="#777", va="top")

# ---- RIGHT: absolute AUROC by arm ---------------------------------------
ax = axes[1]
arms = ["full", "ws", "res", "rnd"]
labels = ["Full\nactivations", "Workspace\n(J-space)", "Complement\n(k=64)", "Random\n(k=64)"]
w = 0.24
for i, (name, d) in enumerate(DATA.items()):
    vals = [np.mean(d[a][:6]) for a in arms]     # mean over layers 2-12
    errs = [0, 0, 0, np.mean(d["sd"][:6])]
    pos = np.arange(4) + (i - 1) * w
    ax.bar(pos, vals, w, color=d["c"], edgecolor=d["c"], zorder=3,
           yerr=errs, error_kw=dict(lw=1, capsize=3, ecolor="#444"))
    for p, v in zip(pos, vals):
        ax.text(p, v + .015, f"{v:.2f}", ha="center", fontsize=7.4)

ax.axhline(0.5, color="#bbb", ls=":", lw=.9, zorder=1)
ax.set_xticks(np.arange(4)); ax.set_xticklabels(labels, fontsize=8.8)
ax.set_ylabel("AUROC (mean, layers 2–12)", fontsize=10)
ax.set_ylim(0, 1.05)
ax.set_title("Only full activations separate reliably",
             fontsize=11.5, fontweight="bold")
ax.grid(axis="y", color="#eee", lw=.7, zorder=0); ax.set_axisbelow(True)
for s in ("top", "right"): ax.spines[s].set_visible(False)

fig.suptitle("Collusion signal is not concentrated in the verbalizable workspace",
             fontsize=13, fontweight="bold", y=1.0)
fig.text(0.5, 0.945,
         "Cross-domain transfer (committee deliberation → sealed-bid procurement). "
         "Same probe, labels, and scenario-grouped splits; only the input subspace varies. "
         "k=64 for all models and layers.",
         ha="center", fontsize=8.6, color="#666")

plt.tight_layout(rect=[0, 0, 1, 0.925])
plt.savefig("/mnt/user-data/outputs/null_result_figure.png", dpi=200,
            facecolor="white", bbox_inches="tight")
print("saved")
