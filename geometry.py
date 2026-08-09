"""
geometry.py — Two experiments that run on saved activations. No GPU, no model.

EXPERIMENT 1: OVERLAP
    Fit a probe. Take the direction it learned (the "deception direction" d).
    Ask how much of that direction lies inside the J-space workspace.

        overlap = ||V_k^T d|| / ||d||        in [0, 1]

    A random direction in d_model dimensions has expected overlap sqrt(k/d).
    If the deception direction matches that, it is geometrically unrelated to
    the workspace — which EXPLAINS the AUROC null instead of merely restating
    it. If it is much higher, the direction lives in the workspace but the
    workspace also carries enough noise to wash out the AUROC advantage.

    Two probe types, because they fail differently:
      - difference of means: scale-free, no optimisation, no scaler ambiguity
      - logistic regression:  what we used for the AUROC tables

EXPERIMENT 2: K SWEEP
    At k=8 the workspace beat random by ~0.15. At k=64 it did not. Sweep
    k in {4,8,...,512} and find where the effect dies. Closes the "you picked
    k to get that answer" objection permanently.

Usage:
    python3 geometry.py llama3.1-8b-it /root/lenses/llama3.1-8b-it
"""

from __future__ import annotations
import sys, json, glob
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

LAYERS   = [2, 4, 6, 8, 10, 12, 15, 20, 25, 30]
K_SWEEP  = [4, 8, 16, 32, 64, 128, 256, 512]
K_FIXED  = 64
N_RANDOM = 30
SEED     = 0


# ---------------------------------------------------------------------------

def load_lens(d):
    p = glob.glob(d + "/**/*_jacobian_lens.pt", recursive=True)[0]
    o = torch.load(p, map_location="cpu", weights_only=False)
    if "J" in o:
        return {int(k): v.float().numpy() for k, v in o["J"].items()}
    n = o["n_done"]
    return {int(k): (v.float() / n).numpy() for k, v in o["jacobian_sum"].items()}


def clean_roles(meta):
    out = []
    for r in meta:
        if r["agent_role"] != "HONEST":
            out.append(r["agent_role"])
        elif r["condition"] == "HONEST":
            out.append("HONEST")
        else:
            out.append("EXPOSED")
    return np.array(out)


def load(tag, lens_dir):
    A = np.load(f"out/{tag}_acts.npy")
    meta = json.load(open(f"out/{tag}_meta.json"))
    return A, clean_roles(meta), np.array([r["scenario_id"] for r in meta]), \
           np.array([r["domain"] for r in meta]), load_lens(lens_dir)


# ===========================================================================
# EXPERIMENT 1 — OVERLAP
# ===========================================================================

def diff_of_means(X, y):
    """Scale-free probe direction. One of Rose et al.'s six architectures."""
    d = X[y == 1].mean(0) - X[y == 0].mean(0)
    return d / (np.linalg.norm(d) + 1e-9)


def logreg_direction(X, y):
    """The probe we used for the AUROC tables. Standardise first, then map the
    weight vector back to raw activation space by dividing out the scaler."""
    sc = StandardScaler().fit(X)
    clf = LogisticRegression(max_iter=3000).fit(sc.transform(X), y)
    d = clf.coef_[0] / (sc.scale_ + 1e-9)      # back to raw coordinates
    return d / (np.linalg.norm(d) + 1e-9)


def overlap(d_vec, Vt, k):
    """Fraction of the direction's length that lies inside the top-k subspace."""
    return float(np.linalg.norm(Vt[:k] @ d_vec) / (np.linalg.norm(d_vec) + 1e-9))


def run_overlap(A, role, lens, k=K_FIXED, seed=SEED):
    m = np.isin(role, ["COVERT", "HONEST"])
    y = (role[m] == "COVERT").astype(int)
    rng = np.random.default_rng(seed)

    print("=" * 82)
    print(f"EXPERIMENT 1 — OVERLAP OF THE DECEPTION DIRECTION WITH J-SPACE  (k={k})")
    print("=" * 82)
    print(f"{'layer':>6}{'d_model':>9}{'expected':>10}{'null 95%':>12}"
          f"{'diff-means':>12}{'logreg':>10}   verdict")
    print("-" * 82)

    rows = []
    for L in LAYERS:
        if L not in lens:
            continue
        X = A[m, L, :].astype(np.float32)
        d_model = X.shape[1]
        _, _, Vt = np.linalg.svd(lens[L])

        expected = np.sqrt(k / d_model)          # analytic random baseline

        # empirical null: overlap of random unit directions
        nulls = []
        for _ in range(200):
            r = rng.standard_normal(d_model)
            nulls.append(overlap(r / np.linalg.norm(r), Vt, k))
        hi = float(np.percentile(nulls, 95))

        o_dm = overlap(diff_of_means(X, y), Vt, k)
        o_lr = overlap(logreg_direction(X, y), Vt, k)

        if o_dm > hi and o_lr > hi:
            v = "IN workspace"
        elif o_dm < hi and o_lr < hi:
            v = "indistinguishable from random"
        else:
            v = "mixed"

        print(f"{L:>6}{d_model:>9}{expected:>10.4f}{hi:>12.4f}"
              f"{o_dm:>12.4f}{o_lr:>10.4f}   {v}")
        rows.append(dict(layer=L, expected=expected, null95=hi,
                         diff_means=o_dm, logreg=o_lr))

    print("-" * 82)
    print("READ: 'expected' is what a random direction gives, sqrt(k/d_model).")
    print("      'null 95%' is the 95th percentile of 200 actual random directions.")
    print("      If the probe directions sit below null 95%, the direction that")
    print("      detects collusion is geometrically unrelated to the workspace.")
    print("      That is the mechanism behind the AUROC null.\n")
    return rows


# ===========================================================================
# EXPERIMENT 2 — K SWEEP
# ===========================================================================

def cv_auc(X, y, groups):
    if len(set(y)) < 2:
        return float("nan")
    n = min(5, len(set(groups)))
    sc = np.zeros(len(y))
    for tr, te in GroupKFold(n_splits=n).split(X, y, groups):
        clf = make_pipeline(StandardScaler(),
                            LogisticRegression(max_iter=3000)).fit(X[tr], y[tr])
        sc[te] = clf.predict_proba(X[te])[:, 1]
    return roc_auc_score(y, sc)


def transfer_auc(Xtr, ytr, Xte, yte):
    if len(set(ytr)) < 2 or len(set(yte)) < 2:
        return float("nan")
    clf = make_pipeline(StandardScaler(),
                        LogisticRegression(max_iter=3000)).fit(Xtr, ytr)
    return roc_auc_score(yte, clf.predict_proba(Xte)[:, 1])


def run_ksweep(A, role, grp, dom, lens, layer=8, seed=SEED):
    """Transfer setting, one layer, sweep k. SVD computed once and re-sliced."""
    CORE, TRANS = {"C1", "C2", "C3"}, {"T1", "T2"}
    tr = np.isin(role, ["COVERT", "HONEST"]) & np.isin(dom, list(CORE))
    te = np.isin(role, ["COVERT", "HONEST"]) & np.isin(dom, list(TRANS))
    ytr = (role[tr] == "COVERT").astype(int)
    yte = (role[te] == "COVERT").astype(int)

    Xtr, Xte = A[tr, layer, :].astype(np.float32), A[te, layer, :].astype(np.float32)
    d_model = Xtr.shape[1]
    _, _, Vt = np.linalg.svd(lens[layer])
    rng = np.random.default_rng(seed)

    print("=" * 82)
    print(f"EXPERIMENT 2 — K SWEEP  (transfer, layer {layer}, "
          f"train n={tr.sum()}, test n={te.sum()})")
    print("=" * 82)
    print(f"{'k':>6}{'%d_model':>10}{'workspace':>12}{'complement':>12}"
          f"{'random':>18}{'ws-rnd':>10}{'p':>8}")
    print("-" * 82)

    rows = []
    for k in K_SWEEP:
        if 2 * k > d_model:
            continue
        ws = transfer_auc(Xtr @ Vt[:k].T, ytr, Xte @ Vt[:k].T, yte)
        cp = transfer_auc(Xtr @ Vt[k:2*k].T, ytr, Xte @ Vt[k:2*k].T, yte)

        rnds = []
        for _ in range(N_RANDOM):
            R, _ = np.linalg.qr(rng.standard_normal((d_model, k)))
            rnds.append(transfer_auc(Xtr @ R, ytr, Xte @ R, yte))
        mu, sd = float(np.mean(rnds)), float(np.std(rnds))

        # percentile test: how many random draws match or beat the workspace
        beat = sum(1 for v in rnds if v >= ws)
        p = (beat + 1) / (len(rnds) + 1)

        star = "*" if p < 0.05 else " "
        print(f"{k:>6}{100*k/d_model:>9.1f}%{ws:>12.3f}{cp:>12.3f}"
              f"{mu:>12.3f}+/-{sd:.3f}{ws-mu:>+10.3f}{p:>8.3f}{star}")
        rows.append(dict(k=k, ws=ws, cp=cp, rnd_mu=mu, rnd_sd=sd, p=p))

    print("-" * 82)
    print("READ: p is the fraction of random subspaces that match or beat the")
    print("      workspace, (beat+1)/(draws+1). This is the correct test — the")
    print("      earlier '3 SD' criterion was invalid because that SD measures")
    print("      the spread of random draws and does not shrink with more draws.\n")
    return rows


# ===========================================================================

if __name__ == "__main__":
    tag, lens_dir = sys.argv[1], sys.argv[2]
    layer = int(sys.argv[3]) if len(sys.argv) > 3 else 8

    A, role, grp, dom, lens = load(tag, lens_dir)
    u, c = np.unique(role, return_counts=True)
    print(f"\n{tag}  {A.shape}   roles: {dict(zip(u.tolist(), c.tolist()))}\n")

    ov = run_overlap(A, role, lens)
    ks = run_ksweep(A, role, grp, dom, lens, layer=layer)

    json.dump({"tag": tag, "overlap": ov, "ksweep": ks, "ksweep_layer": layer},
              open(f"out/{tag}_geometry.json", "w"), indent=1)
    print(f"saved -> out/{tag}_geometry.json")
