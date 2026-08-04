"""
analyse2.py — Revised ablation.

Three fixes over analyse.py, all forced by the llama3.1-8b-it results:

1. HONEST CONTAMINATION. Agents labelled HONEST inside a COVERT/BENIGN episode
   heard the colluders speak before them. They are not clean controls. Relabel
   them EXPOSED and exclude. (Also reported separately — "does sitting next to
   a colluder change your activations" is a free extra result.)

2. SATURATION. At layers >=15 every arm hits ~1.00, including random-k. A
   random projection scoring 0.997 means the classes are trivially separable
   and the comparison carries no information. The informative zone on llama
   was layers 5-10, so sweep it finely and treat the late layers as a ceiling
   check rather than a result.

3. CROSS-DOMAIN TRANSFER. In-distribution is saturated (this is exactly what
   Rose et al. hit). Train on the committee domains, test on procurement.
   This is where numbers still move and where the OOD claim lives.

Usage:
    python3 analyse2.py llama3.1-8b-it /root/lenses/llama3.1-8b-it
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

LAYERS   = [0, 2, 4, 6, 8, 10, 12, 15, 20, 25, 30]
ENERGY   = 0.50
N_RANDOM = 5
SEED     = 0
CORE     = {"C1", "C2", "C3"}
TRANSFER = {"T1", "T2"}


def load_lens(d):
    p = glob.glob(d + "/**/*_jacobian_lens.pt", recursive=True)[0]
    o = torch.load(p, map_location="cpu", weights_only=False)
    if "J" in o:
        return {int(k): v.float().numpy() for k, v in o["J"].items()}
    n = o["n_done"]
    return {int(k): (v.float() / n).numpy() for k, v in o["jacobian_sum"].items()}


def clean_roles(meta):
    """HONEST only counts when the whole episode was honest."""
    out = []
    for r in meta:
        if r["agent_role"] != "HONEST":
            out.append(r["agent_role"])
        elif r["condition"] == "HONEST":
            out.append("HONEST")
        else:
            out.append("EXPOSED")
    return np.array(out)


def arms_for_layer(X, J, frac=ENERGY, seed=SEED, n_random=N_RANDOM):
    s = np.linalg.svd(J, compute_uv=False)
    e = np.cumsum(s ** 2) / np.sum(s ** 2)
    k = int(np.searchsorted(e, frac) + 1)
    _, _, Vt = np.linalg.svd(J)
    out = {"full": X, "workspace": X @ Vt[:k].T, "residue_k": X @ Vt[k:2 * k].T}
    rng = np.random.default_rng(seed)
    for i in range(n_random):
        R, _ = np.linalg.qr(rng.standard_normal((J.shape[0], k)))
        out[f"rand{i}"] = X @ R
    return out, k


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


def table(title, rows_fn, lens):
    print("=" * 76)
    print(title)
    print("=" * 76)
    print(f"{'layer':>6}{'k':>7}{'full':>9}{'workspc':>9}{'residue':>9}"
          f"{'random':>17}{'ws-rnd':>9}")
    for L in LAYERS:
        if L not in lens:
            continue
        r, k = rows_fn(L, lens[L])
        if r is None:
            continue
        rnd = [v for n, v in r.items() if n.startswith("rand")]
        mu, sd = np.mean(rnd), np.std(rnd)
        gap = r["workspace"] - mu
        flag = " *" if gap > 3 * max(sd, 1e-3) else "  "
        sat = " SAT" if mu > 0.95 else ""
        print(f"{L:>6}{k:>7}{r['full']:>9.3f}{r['workspace']:>9.3f}"
              f"{r['residue_k']:>9.3f}{mu:>11.3f}+/-{sd:.3f}"
              f"{gap:>+8.3f}{flag}{sat}")
    print("  * = workspace exceeds random by >3 SD    SAT = random >0.95, "
          "layer is saturated and uninformative\n")


def main(tag, lens_dir):
    A = np.load(f"out/{tag}_acts.npy")
    meta = json.load(open(f"out/{tag}_meta.json"))
    role = clean_roles(meta)
    dom = np.array([r["domain"] for r in meta])
    grp = np.array([r["scenario_id"] for r in meta])
    lens = load_lens(lens_dir)

    u, c = np.unique(role, return_counts=True)
    print(f"{tag}  {A.shape}")
    print("roles:", dict(zip(u.tolist(), c.tolist())), "\n")

    # ---- within-distribution contrasts -----------------------------------
    for name, (pos, neg) in {
        "COVERT vs HONEST  (clean honests only)": ("COVERT", "HONEST"),
        "COVERT vs BENIGN  <- the confound test": ("COVERT", "BENIGN"),
        "COVERT vs COVERT_NC  <- is it the hide-instruction?": ("COVERT", "COVERT_NC"),
        "HONEST vs EXPOSED  <- does proximity to colluders show?": ("EXPOSED", "HONEST"),
    }.items():
        m = np.isin(role, [pos, neg])
        if m.sum() == 0 or len(set(role[m])) < 2:
            continue
        y = (role[m] == pos).astype(int)
        g = grp[m]

        def rows(L, J, m=m, y=y, g=g):
            arms, k = arms_for_layer(A[m, L, :].astype(np.float32), J)
            return {n: cv_auc(Xa, y, g) for n, Xa in arms.items()}, k

        table(f"{name}   n={m.sum()}, {len(set(g))} scenarios", rows, lens)

    # ---- cross-domain transfer -------------------------------------------
    tr_m = np.isin(role, ["COVERT", "HONEST"]) & np.isin(dom, list(CORE))
    te_m = np.isin(role, ["COVERT", "HONEST"]) & np.isin(dom, list(TRANSFER))
    if tr_m.sum() and te_m.sum():
        ytr = (role[tr_m] == "COVERT").astype(int)
        yte = (role[te_m] == "COVERT").astype(int)

        def rows(L, J):
            atr, k = arms_for_layer(A[tr_m, L, :].astype(np.float32), J)
            ate, _ = arms_for_layer(A[te_m, L, :].astype(np.float32), J)
            return {n: transfer_auc(atr[n], ytr, ate[n], yte)
                    for n in atr}, k

        table(f"TRANSFER: train on committee domains -> test on procurement   "
              f"train n={tr_m.sum()}, test n={te_m.sum()}", rows, lens)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
