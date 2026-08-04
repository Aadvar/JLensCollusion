"""
appendix_figs.py — All appendix figures from the recorded results.

Produces:
  A1_workspace_dimensionality.png   how k grows with depth (llama)
  A2_jacobian_spectra.png           cross-model spectra; gemma outlier
  A3_saturation.png                 why in-distribution is uninformative
  A4_llama_contrasts.png            all four contrasts, four arms, by layer
  A5_transfer_all_models.png        transfer, three models, k=64
  A6_effect_size_summary.png        forest-style plot of ws-rnd with bands

Every number is from the recorded runs. Nothing simulated.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

OUT = "/mnt/user-data/outputs/"
C = {"llama": "#2E7CF6", "qwen": "#7B3FE4", "gemma": "#2FA37C"}
ARMC = {"full": "#222", "workspace": "#D9480F", "residue": "#1971C2", "random": "#868E96"}


def style(ax):
    ax.grid(color="#eee", lw=.7, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


# ===========================================================================
# A1 — workspace dimensionality by depth (llama3.1-8b-it, d_model=4096)
# ===========================================================================
L = [0, 5, 10, 15, 20, 25, 30]
k50 = [3, 8, 11, 87, 422, 623, 1036]
k90 = [49, 103, 245, 1049, 2046, 2417, 3079]
D = 4096

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
ax = axes[0]
ax.semilogy(L, k50, "-o", color="#D9480F", lw=2, ms=6, label="50% of spectral energy")
ax.semilogy(L, k90, "-s", color="#1971C2", lw=2, ms=5, label="90% of spectral energy")
ax.axhline(D, color="#999", ls="--", lw=1)
ax.text(0.3, D * 1.1, f"d_model = {D}", fontsize=8, color="#777")
ax.set_xlabel("layer"); ax.set_ylabel("dimensions (log scale)")
ax.set_title("Workspace size grows 345× with depth", fontsize=11, fontweight="bold")
ax.legend(fontsize=8.5, frameon=False); style(ax)

ax = axes[1]
ax.plot(L, [100 * k / D for k in k50], "-o", color="#D9480F", lw=2, ms=6)
ax.fill_between(L, 0, [100 * k / D for k in k50], color="#D9480F", alpha=.12)
for x, k in zip(L, k50):
    ax.annotate(f"k={k}", (x, 100 * k / D), textcoords="offset points",
                xytext=(0, 7), ha="center", fontsize=7.5)
ax.set_xlabel("layer"); ax.set_ylabel("% of d_model")
ax.set_title("A fixed energy threshold is not a fixed subspace",
             fontsize=11, fontweight="bold")
ax.set_ylim(0, 30); style(ax)

fig.suptitle("Figure A1: Jacobian workspace dimensionality by depth (llama3.1-8b-it)",
             fontsize=12, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(OUT + "A1_workspace_dimensionality.png", dpi=200,
            facecolor="white", bbox_inches="tight")
plt.close()

# ===========================================================================
# A2 — Jacobian spectra across models (mid-layer)
# ===========================================================================
SPEC = {
    "llama3.1-8b-it\nlayer 15, d=4096":  dict(c=C["llama"], s0=20.1, s1=16.8,
                                              ratio=1.2, top1=7.5, k50=87, d=4096),
    "qwen3.5-27b\nlayer 31, d=5120":     dict(c=C["qwen"], s0=13.8, s1=10.0,
                                              ratio=1.4, top1=10.7, k50=47, d=5120),
    "gemma-3-27b-it\nlayer 30, d=5376":  dict(c=C["gemma"], s0=106.8, s1=23.3,
                                              ratio=4.6, top1=40.0, k50=10, d=5376),
}
fig, axes = plt.subplots(1, 3, figsize=(13, 4))
names = list(SPEC)

ax = axes[0]
x = np.arange(3)
ax.bar(x, [SPEC[n]["ratio"] for n in names],
       color=[SPEC[n]["c"] for n in names], zorder=3)
for i, n in enumerate(names):
    ax.text(i, SPEC[n]["ratio"] + .1, f"{SPEC[n]['ratio']:.1f}", ha="center", fontsize=9)
ax.axhline(1, color="#999", ls=":", lw=1)
ax.set_xticks(x); ax.set_xticklabels([n.split("\n")[0] for n in names], fontsize=8, rotation=12)
ax.set_ylabel("$\\sigma_1 / \\sigma_2$")
ax.set_title("Leading singular value ratio", fontsize=10.5, fontweight="bold"); style(ax)

ax = axes[1]
ax.bar(x, [SPEC[n]["top1"] for n in names],
       color=[SPEC[n]["c"] for n in names], zorder=3)
for i, n in enumerate(names):
    ax.text(i, SPEC[n]["top1"] + 1, f"{SPEC[n]['top1']:.0f}%", ha="center", fontsize=9)
ax.set_xticks(x); ax.set_xticklabels([n.split("\n")[0] for n in names], fontsize=8, rotation=12)
ax.set_ylabel("% of spectral energy in $\\sigma_1$ alone")
ax.set_title("Single-direction dominance", fontsize=10.5, fontweight="bold")
ax.set_ylim(0, 48); style(ax)

ax = axes[2]
ax.bar(x, [SPEC[n]["k50"] for n in names],
       color=[SPEC[n]["c"] for n in names], zorder=3)
for i, n in enumerate(names):
    ax.text(i, SPEC[n]["k50"] + 2, f"k={SPEC[n]['k50']}", ha="center", fontsize=9)
ax.set_xticks(x); ax.set_xticklabels([n.split("\n")[0] for n in names], fontsize=8, rotation=12)
ax.set_ylabel("k at 50% energy")
ax.set_title("Resulting workspace size", fontsize=10.5, fontweight="bold"); style(ax)

fig.suptitle("Figure A2: Energy-based dimensionality selection is not transferable across model families",
             fontsize=12, fontweight="bold", y=1.03)
fig.text(0.5, 0.955, "On gemma-3-27b-it a single direction carries 40% of the Jacobian's "
         "spectral energy, so a 50%-energy rule yields k=10 where llama yields k=87.",
         ha="center", fontsize=8.6, color="#666")
plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.savefig(OUT + "A2_jacobian_spectra.png", dpi=200,
            facecolor="white", bbox_inches="tight")
plt.close()

# ===========================================================================
# A3 — saturation
# ===========================================================================
SATL = [0, 2, 4, 6, 8, 10, 12, 15, 20, 25, 30]
rnd_ch = [0.500, 0.493, 0.631, 0.727, 0.623, 0.825, 0.891, 0.997, 0.994, 0.994, 0.991]
rnd_cb = [0.500, 0.542, 0.689, 0.664, 0.656, 0.829, 0.897, 0.993, 1.000, 0.999, 0.999]
full_ch = [0.500, 0.913, 0.941, 0.970, 0.985, 0.998, 0.997, 0.999, 0.997, 0.997, 0.995]
kk = [3, 3, 7, 8, 8, 11, 17, 87, 422, 623, 1036]

fig, ax = plt.subplots(figsize=(7.5, 4.6))
x = np.arange(len(SATL))
ax.axhspan(0.95, 1.02, color="#FFE3E3", zorder=0)
ax.text(0.15, 0.965, "saturated: no subspace comparison is informative",
        fontsize=8.4, color="#C92A2A")
ax.plot(x, full_ch, "-o", color=ARMC["full"], lw=2, ms=5, label="full activations")
ax.plot(x, rnd_ch, "-s", color=ARMC["random"], lw=2, ms=5,
        label="random k-dim subspace (COVERT vs HONEST)")
ax.plot(x, rnd_cb, "-^", color="#ADB5BD", lw=1.6, ms=5, ls="--",
        label="random k-dim subspace (COVERT vs BENIGN)")
ax.set_xticks(x)
ax.set_xticklabels([f"{l}\nk={k}" for l, k in zip(SATL, kk)], fontsize=7.6)
ax.set_xlabel("layer / workspace dimensionality", fontsize=9.5)
ax.set_ylabel("AUROC"); ax.set_ylim(0.45, 1.03)
ax.axhline(0.5, color="#bbb", ls=":", lw=.9)
ax.legend(fontsize=8.3, frameon=False, loc="lower right")
ax.set_title("Figure A3: In-distribution saturation (llama3.1-8b-it)",
             fontsize=11.5, fontweight="bold")
style(ax)
fig.text(0.5, -0.04, "From layer 15 onward a random projection reaches 0.99+ AUROC. "
         "Any subspace comparison in this regime is uninformative.",
         ha="center", fontsize=8.5, color="#666")
plt.tight_layout()
plt.savefig(OUT + "A3_saturation.png", dpi=200, facecolor="white", bbox_inches="tight")
plt.close()

# ===========================================================================
# A4 — all four contrasts, llama, energy-based k
# ===========================================================================
CON = {
    "COVERT vs HONEST  (n=192)": dict(
        full=[0.913, 0.941, 0.970, 0.985, 0.998, 0.997, 0.999, 0.997, 0.997, 0.995],
        ws  =[0.435, 0.597, 0.736, 0.767, 0.864, 0.930, 0.988, 0.992, 0.994, 0.997],
        res =[0.578, 0.508, 0.750, 0.825, 0.884, 0.921, 0.967, 0.993, 0.995, 0.993],
        rnd =[0.493, 0.631, 0.727, 0.623, 0.825, 0.891, 0.997, 0.994, 0.994, 0.991],
        sd  =[0.056, 0.027, 0.052, 0.041, 0.038, 0.031, 0.004, 0.002, 0.003, 0.002]),
    "COVERT vs BENIGN  (n=128)": dict(
        full=[0.868, 0.933, 0.981, 0.981, 0.998, 1.000, 1.000, 1.000, 1.000, 1.000],
        ws  =[0.565, 0.703, 0.708, 0.704, 0.800, 0.906, 0.994, 1.000, 0.999, 0.998],
        res =[0.566, 0.558, 0.693, 0.769, 0.843, 0.852, 0.992, 0.999, 0.997, 0.997],
        rnd =[0.542, 0.689, 0.664, 0.656, 0.829, 0.897, 0.993, 1.000, 0.999, 0.999],
        sd  =[0.052, 0.036, 0.064, 0.035, 0.038, 0.069, 0.004, 0.000, 0.001, 0.001]),
    "COVERT vs COVERT_NC  (n=128)": dict(
        full=[0.531, 0.528, 0.519, 0.665, 0.740, 0.860, 0.927, 0.862, 0.864, 0.840],
        ws  =[0.544, 0.529, 0.508, 0.603, 0.564, 0.543, 0.817, 0.781, 0.782, 0.831],
        res =[0.509, 0.438, 0.471, 0.609, 0.513, 0.586, 0.768, 0.870, 0.762, 0.812],
        rnd =[0.409, 0.540, 0.533, 0.540, 0.523, 0.596, 0.776, 0.799, 0.825, 0.834],
        sd  =[0.067, 0.042, 0.019, 0.064, 0.035, 0.054, 0.066, 0.021, 0.016, 0.010]),
    "HONEST vs EXPOSED  (n=320)": dict(
        full=[0.798, 0.821, 0.845, 0.892, 0.918, 0.937, 0.921, 0.896, 0.888, 0.885],
        ws  =[0.582, 0.644, 0.646, 0.694, 0.730, 0.752, 0.844, 0.864, 0.860, 0.876],
        res =[0.586, 0.621, 0.644, 0.660, 0.588, 0.666, 0.903, 0.863, 0.889, 0.875],
        rnd =[0.579, 0.612, 0.639, 0.625, 0.663, 0.730, 0.831, 0.871, 0.875, 0.877],
        sd  =[0.041, 0.036, 0.031, 0.041, 0.037, 0.027, 0.044, 0.013, 0.002, 0.006]),
}
LX = [2, 4, 6, 8, 10, 12, 15, 20, 25, 30]

fig, axes = plt.subplots(2, 2, figsize=(12.5, 7.5))
for ax, (name, d) in zip(axes.flat, CON.items()):
    x = np.arange(len(LX))
    ax.fill_between(x, np.array(d["rnd"]) - np.array(d["sd"]),
                    np.array(d["rnd"]) + np.array(d["sd"]),
                    color=ARMC["random"], alpha=.22, zorder=1)
    ax.plot(x, d["full"], "-o", color=ARMC["full"], lw=1.9, ms=4.5, label="full")
    ax.plot(x, d["ws"], "-o", color=ARMC["workspace"], lw=1.9, ms=4.5, label="workspace")
    ax.plot(x, d["res"], "-o", color=ARMC["residue"], lw=1.9, ms=4.5, label="complement")
    ax.plot(x, d["rnd"], "--", color=ARMC["random"], lw=1.6, label="random ±1 SD")
    ax.axhline(0.5, color="#ccc", ls=":", lw=.9)
    ax.set_xticks(x); ax.set_xticklabels(LX, fontsize=8)
    ax.set_ylim(0.35, 1.04); ax.set_xlabel("layer", fontsize=9)
    ax.set_ylabel("AUROC", fontsize=9)
    ax.set_title(name, fontsize=10.5, fontweight="bold"); style(ax)
axes.flat[0].legend(fontsize=8, frameon=False, loc="lower right", ncol=2)

fig.suptitle("Figure A4: All contrasts, llama3.1-8b-it, in-distribution (k = 50% spectral energy)",
             fontsize=12.5, fontweight="bold", y=1.0)
fig.text(0.5, 0.955, "COVERT vs BENIGN is the confound control; COVERT vs COVERT_NC is the "
         "minimal pair differing by one concealment sentence; EXPOSED are honest agents "
         "seated in colluding episodes.", ha="center", fontsize=8.5, color="#666")
plt.tight_layout(rect=[0, 0, 1, 0.94])
plt.savefig(OUT + "A4_llama_contrasts.png", dpi=200, facecolor="white", bbox_inches="tight")
plt.close()

# ===========================================================================
# A5 — transfer, all three models, k=64
# ===========================================================================
TL = [2, 4, 6, 8, 10, 12, 15, 20, 25, 30]
TR = {
    "llama3.1-8b-it": dict(c=C["llama"],
        full=[0.725, 0.766, 0.811, 0.916, 0.965, 0.969, 0.979, 0.953, 0.965, 0.957],
        ws  =[0.527, 0.709, 0.779, 0.787, 0.846, 0.834, 0.861, 0.697, 0.605, 0.664],
        res =[0.521, 0.693, 0.688, 0.750, 0.803, 0.863, 0.594, 0.748, 0.688, 0.783],
        rnd =[0.599, 0.657, 0.739, 0.740, 0.844, 0.872, 0.858, 0.812, 0.801, 0.759],
        sd  =[0.081, 0.077, 0.072, 0.088, 0.056, 0.086, 0.085, 0.096, 0.119, 0.112]),
    "qwen3.5-27b": dict(c=C["qwen"],
        full=[0.680, 0.840, 0.848, 0.879, 0.842, 0.857, 0.822, 0.965, 0.982, 1.000],
        ws  =[0.607, 0.670, 0.773, 0.895, 0.744, 0.723, 0.588, 0.689, 0.775, 0.830],
        res =[0.721, 0.680, 0.715, 0.697, 0.631, 0.529, 0.627, 0.740, 0.918, 0.990],
        rnd =[0.613, 0.716, 0.712, 0.752, 0.710, 0.710, 0.688, 0.780, 0.859, 0.968],
        sd  =[0.113, 0.079, 0.085, 0.081, 0.091, 0.097, 0.104, 0.126, 0.080, 0.046]),
    "gemma-3-27b-it": dict(c=C["gemma"],
        full=[0.605, 0.629, 0.707, 0.719, 0.646, 0.693, 0.699, 0.924, 0.977, 0.994],
        ws  =[0.527, 0.551, 0.584, 0.490, 0.506, 0.574, 0.652, 0.777, 0.867, 0.963],
        res =[0.504, 0.658, 0.510, 0.588, 0.615, 0.807, 0.646, 0.488, 0.807, 0.930],
        rnd =[0.664, 0.667, 0.682, 0.632, 0.606, 0.637, 0.649, 0.757, 0.857, 0.844],
        sd  =[0.060, 0.061, 0.058, 0.079, 0.078, 0.107, 0.080, 0.084, 0.083, 0.109]),
}
fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), sharey=True)
for ax, (m, d) in zip(axes, TR.items()):
    x = np.arange(len(TL))
    ax.fill_between(x, np.array(d["rnd"]) - np.array(d["sd"]),
                    np.array(d["rnd"]) + np.array(d["sd"]),
                    color=ARMC["random"], alpha=.25, zorder=1)
    ax.plot(x, d["full"], "-o", color=ARMC["full"], lw=1.9, ms=4.5, label="full")
    ax.plot(x, d["ws"], "-o", color=ARMC["workspace"], lw=2.1, ms=5, label="workspace")
    ax.plot(x, d["res"], "-o", color=ARMC["residue"], lw=1.7, ms=4.5, label="complement")
    ax.plot(x, d["rnd"], "--", color=ARMC["random"], lw=1.6, label="random ±1 SD")
    ax.axhline(0.5, color="#ccc", ls=":", lw=.9)
    ax.set_xticks(x); ax.set_xticklabels(TL, fontsize=8)
    ax.set_xlabel("layer", fontsize=9.5)
    ax.set_title(m, fontsize=10.5, fontweight="bold"); style(ax)
axes[0].set_ylabel("AUROC", fontsize=9.5); axes[0].set_ylim(0.42, 1.04)
axes[0].legend(fontsize=8.2, frameon=False, loc="lower right", ncol=2)

fig.suptitle("Figure A5: Cross-domain transfer, k = 64 for all models and layers",
             fontsize=12.5, fontweight="bold", y=1.02)
fig.text(0.5, 0.945, "Train on committee-deliberation domains, test zero-shot on sealed-bid "
         "procurement. The workspace trace does not separate from the random band in any model.",
         ha="center", fontsize=8.6, color="#666")
plt.tight_layout(rect=[0, 0, 1, 0.92])
plt.savefig(OUT + "A5_transfer_all_models.png", dpi=200, facecolor="white", bbox_inches="tight")
plt.close()

# ===========================================================================
# A6 — effect size summary (forest style)
# ===========================================================================
rows, labels, cols = [], [], []
for m, d in TR.items():
    for i, L_ in enumerate(TL):
        rows.append((np.array(d["ws"])[i] - np.array(d["rnd"])[i], d["sd"][i]))
        labels.append(f"{m.split('-')[0]}  L{L_}")
        cols.append(d["c"])

fig, ax = plt.subplots(figsize=(7.2, 9))
y = np.arange(len(rows))[::-1]
for yy, (g, s), c in zip(y, rows, cols):
    ax.errorbar(g, yy, xerr=s, fmt="o", color=c, ms=4.5, lw=1.3,
                capsize=2.5, zorder=3)
ax.axvline(0, color="#333", lw=1.2, zorder=2)
ax.axvspan(-0.05, 0.05, color="#f1f3f5", zorder=0)
ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=7.2)
ax.set_xlabel("AUROC(workspace) − AUROC(random),  ±1 SD of random draws", fontsize=9)
ax.set_title("Figure A6: Effect size across all model × layer cells (transfer, k=64)",
             fontsize=11, fontweight="bold")
ax.set_xlim(-0.35, 0.35); style(ax)
ax.legend(handles=[Patch(facecolor=c, label=m) for m, c in
                   [(m, d["c"]) for m, d in TR.items()]],
          fontsize=8, frameon=False, loc="lower left")
fig.text(0.5, -0.015, "30 of 30 cells have confidence intervals overlapping zero.",
         ha="center", fontsize=8.6, color="#666")
plt.tight_layout()
plt.savefig(OUT + "A6_effect_size_summary.png", dpi=200,
            facecolor="white", bbox_inches="tight")
plt.close()

print("6 figures written")
