"""
analyse.py — The ablation. This is the paper.

Four feature sets, one probe, identical labels and CV splits:

    full        all d_model dims          baseline
    workspace   top-k of the Jacobian     is the signal in the reportable part?
    residue_k   next-k after that         is it outside, at matched capacity?
    random_k    k random dims             capacity control

k is set PER LAYER from that layer's own spectrum, because the workspace is
not a fixed size — on llama3.1-8b-it it runs from 3 dims at layer 0 to 1036 at
layer 30. A constant k would compare a generous slice at one depth against a
starvation slice at another.

Usage:
    python3 analyse.py llama3.1-8b-it /root/lenses/llama3.1-8b-it
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

LAYERS       = [5, 10, 15, 20, 25, 30]
ENERGY       = 0.50      # k = dims holding this fraction of spectral energy
N_RANDOM     = 5
SEED         = 0


def load_lens(d):
    p = glob.glob(d + "/**/*_jacobian_lens.pt", recursive=True)[0]
    o = torch.load(p, map_location="cpu", weights_only=False)
    if "J" in o:
        return {int(k): v.float().numpy() for k, v in o["J"].items()}
    n = o["n_done"]
    return {int(k): (v.float() / n).numpy() for k, v in o["jacobian_sum"].items()}


def k_from_spectrum(J, frac):
    s = np.linalg.svd(J, compute_uv=False)
    e = np.cumsum(s ** 2) / np.sum(s ** 2)
    return int(np.searchsorted(e, frac) + 1)


def arms_for_layer(X, J, frac, seed=SEED, n_random=N_RANDOM):
    k = k_from_spectrum(J, frac)
    _, _, Vt = np.linalg.svd(J)
    out = {"full": X,
           "workspace": X @ Vt[:k].T,
           "residue_k": X @ Vt[k:2 * k].T}
    rng = np.random.default_rng(seed)
    for i in range(n_random):
        R, _ = np.linalg.qr(rng.standard_normal((J.shape[0], k)))
        out[f"rand{i}"] = X @ R
    return out, k


def auc(X, y, groups):
    if len(set(y)) < 2:
        return float("nan")
    n = min(5, len(set(groups)))
    sc = np.zeros(len(y), dtype=float)
    for tr, te in GroupKFold(n_splits=n).split(X, y, groups):
        clf = make_pipeline(StandardScaler(),
                            LogisticRegression(max_iter=3000, C=1.0))
        clf.fit(X[tr], y[tr])
        sc[te] = clf.predict_proba(X[te])[:, 1]
    return roc_auc_score(y, sc)


def main(tag, lens_dir):
    A = np.load(f"out/{tag}_acts.npy")                 # [n, n_layers, d]
    meta = json.load(open(f"out/{tag}_meta.json"))
    role = np.array([r["agent_role"] for r in meta])
    grp = np.array([r["scenario_id"] for r in meta])
    lens = load_lens(lens_dir)
    print(f"{tag}: {A.shape}  roles={dict(zip(*np.unique(role, return_counts=True)))}\n")

    contrasts = {
        "COVERT vs HONEST":    ("COVERT", "HONEST"),
        "COVERT vs BENIGN":    ("COVERT", "BENIGN"),     # the confound test
        "COVERT vs COVERT_NC": ("COVERT", "COVERT_NC"),  # is it the hide-instruction?
    }

    for cname, (pos, neg) in contrasts.items():
        mask = np.isin(role, [pos, neg])
        if mask.sum() == 0 or len(set(role[mask])) < 2:
            continue
        y = (role[mask] == pos).astype(int)
        g = grp[mask]
        print("=" * 74)
        print(f"{cname}   (n={mask.sum()}, {len(set(g))} scenarios)")
        print("=" * 74)
        print(f"{'layer':>6}{'k':>7}{'full':>9}{'workspc':>9}"
              f"{'residue':>9}{'random':>16}")
        for L in LAYERS:
            if L not in lens:
                continue
            X = A[mask, L, :].astype(np.float32)
            arms, k = arms_for_layer(X, lens[L], ENERGY)
            r = {n: auc(Xa, y, g) for n, Xa in arms.items()}
            rnd = [v for n, v in r.items() if n.startswith("rand")]
            print(f"{L:>6}{k:>7}{r['full']:>9.3f}{r['workspace']:>9.3f}"
                  f"{r['residue_k']:>9.3f}"
                  f"{np.mean(rnd):>10.3f}+/-{np.std(rnd):.3f}")
        print()

    print("READ:")
    print("  workspace >> random     -> J-space carries real signal")
    print("  residue   >> random     -> signal ALSO lives outside the workspace")
    print("  COVERT vs BENIGN is the one that matters: if it's at chance,")
    print("  you are detecting secrecy, not collusion.")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
