"""
arms_v2.py — replacement for arms.py / arms_for_layer() in analyse2.py.

Four changes over the current version. (1) and (4) can change the result;
(2) and (3) change whether you are allowed to claim it.

1. WORKSPACE BASIS IS DEFINED IN LOGIT SPACE, NOT RESIDUAL SPACE.
   The lens readout is  lens_l(h) = W_U @ (J_l @ h).  Verbalizability is a
   property of that composition. SVD(J_l) gives the directions at layer l that
   are amplified most into the final residual stream -- which includes
   directions W_U barely reads. SVD(W_U @ J_l) gives the directions that move
   logits most, which is what "the model could say this" actually means.
   Both are available here via basis=. Report both; the delta is a result.

2. FIXED k ACROSS MODELS AND LAYERS.
   The 50%-energy rule gives k=87 on llama and k=10 on gemma. Those are not
   the same object. Energy-based k is still useful as a descriptive statistic
   (report the spectra), just not as a comparison axis.

3. PERCENTILE SIGNIFICANCE TEST.
   p = (b + 1) / (R + 1) where b = number of random draws scoring >= workspace.
   NOTE: with R=5 the smallest attainable p is 1/6 = 0.167. You cannot reject
   at any conventional threshold. R >= 30 is the floor, R = 100 is better.
   This is why the current N_RANDOM=5 runs cannot support either a positive or
   a null claim.

4. THE COMPLEMENT ARM IS THE WHOLE COMPLEMENT, NOT ITS TOP k.
   residue_k (the next k directions) answers "is there signal just outside the
   workspace." residue_full answers "is there signal anywhere outside it,"
   which is the safety-relevant question -- if workspace monitoring is to be
   sufficient, the entire complement must be uninformative. Both are here.
"""

from __future__ import annotations
import warnings
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold


# --------------------------------------------------------------------------
# Basis construction
# --------------------------------------------------------------------------

def workspace_basis(J, W_U=None, k=64, basis="logit"):
    """Return [d, k] orthonormal basis for the workspace at one layer.

    J    : [d, d]      averaged Jacobian for this layer
    W_U  : [V, d]      unembedding. Required when basis="logit".
    basis: "logit"     top-k right singular vectors of (W_U @ J)   <- recommended
           "jacobian"  top-k right singular vectors of J           <- current code

    Returns (V_ws [d, k], V_comp [d, d-k], singular_values).
    """
    if basis == "logit":
        if W_U is None:
            raise ValueError("basis='logit' needs W_U")
        # [V, d] @ [d, d] -> [V, d]. Only the right singular vectors are needed,
        # so go through the d x d Gram matrix instead of a V x d SVD.
        M = np.asarray(W_U, dtype=np.float32) @ np.asarray(J, dtype=np.float32)
        G = M.T @ M                        # [d, d], symmetric PSD
        evals, evecs = np.linalg.eigh(G)
        order = np.argsort(evals)[::-1]
        Vt = evecs[:, order].T
        s = np.sqrt(np.clip(evals[order], 0, None))
    elif basis == "jacobian":
        _, s, Vt = np.linalg.svd(np.asarray(J, dtype=np.float32))
    else:
        raise ValueError(basis)

    return Vt[:k].T, Vt[k:].T, s


def spectrum_stats(s):
    """Descriptive only. Report these; do not select k from them."""
    e = np.cumsum(s ** 2) / np.sum(s ** 2)
    return {
        "k_50": int(np.searchsorted(e, 0.50) + 1),
        "k_90": int(np.searchsorted(e, 0.90) + 1),
        "top1_energy": float(s[0] ** 2 / np.sum(s ** 2)),
        "s1_over_s2": float(s[0] / s[1]),
    }


def make_arms(X, J, W_U=None, k=64, basis="logit", n_random=30, seed=0):
    """Feature sets from one layer's activations. All arms except the two
    'full'/'residue_full' arms have exactly k dimensions."""
    X = np.asarray(X, dtype=np.float32)
    d = J.shape[0]
    assert X.shape[1] == d, f"X d={X.shape[1]}, lens d={d}"

    V_ws, V_comp, s = workspace_basis(J, W_U, k, basis)

    arms = {
        "full":         X,
        "workspace":    X @ V_ws,
        "residue_k":    X @ V_comp[:, :k],
        "residue_full": X @ V_comp,
    }
    rng = np.random.default_rng(seed)
    for i in range(n_random):
        R, _ = np.linalg.qr(rng.standard_normal((d, k)).astype(np.float32))
        arms[f"random_{i}"] = X @ R
    return arms, spectrum_stats(s)


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------

def _probe():
    # StandardScaler is not optional: dot-product features differ by orders of
    # magnitude across layers, and L1 with C=10 on unscaled features prunes on
    # scale rather than on information.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clf = LogisticRegression(max_iter=5000, class_weight="balanced",
                                 penalty="l1", solver="liblinear", C=1.0)
    return make_pipeline(StandardScaler(), clf)


def cv_auc(X, y, groups, n_splits=5):
    """Grouped CV. groups must be EPISODE id, not scenario and not row index --
    an episode contributes ~18 correlated samples and splitting inside one
    leaks."""
    y = np.asarray(y)
    if len(set(y)) < 2:
        return float("nan")
    n_splits = min(n_splits, len(set(groups)))
    oof = np.zeros(len(y))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for tr, te in GroupKFold(n_splits=n_splits).split(X, y, groups):
            if len(set(y[tr])) < 2:
                return float("nan")
            m = _probe().fit(X[tr], y[tr])
            oof[te] = m.predict_proba(X[te])[:, 1]
    return roc_auc_score(y, oof)


def transfer_auc(X, y, domain, held_out):
    y, domain = np.asarray(y), np.asarray(domain)
    tr, te = domain != held_out, domain == held_out
    if len(set(y[tr])) < 2 or len(set(y[te])) < 2:
        return float("nan")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = _probe().fit(X[tr], y[tr])
    return roc_auc_score(y[te], m.predict_proba(X[te])[:, 1])


def evaluate(arms, eval_fn):
    """eval_fn: X -> auc, closed over y/groups/domain."""
    return {name: eval_fn(Xa) for name, Xa in arms.items()}


def verdict(results):
    """Percentile test against the random-subspace null."""
    rand = np.array([v for n, v in results.items() if n.startswith("random_")])
    ws = results["workspace"]
    R = len(rand)
    b = int(np.sum(rand >= ws))
    p = (b + 1) / (R + 1)
    return {
        "workspace": ws,
        "random_mean": float(rand.mean()),
        "random_p05": float(np.percentile(rand, 5)),
        "random_p95": float(np.percentile(rand, 95)),
        "R": R,
        "p": p,
        "min_attainable_p": 1 / (R + 1),
        "significant": p < 0.05,
        "underpowered": 1 / (R + 1) > 0.05,
    }


def report(results, stats, label=""):
    v = verdict(results)
    lines = [f"{label}  (k_50={stats['k_50']}, s1/s2={stats['s1_over_s2']:.2f}, "
             f"top1_energy={stats['top1_energy']:.1%})"]
    for n in ("full", "workspace", "residue_k", "residue_full"):
        if n in results:
            lines.append(f"  {n:<14} {results[n]:.3f}")
    lines.append(f"  {'random':<14} {v['random_mean']:.3f} "
                 f"[{v['random_p05']:.3f}, {v['random_p95']:.3f}]  R={v['R']}")
    lines.append(f"  workspace vs random: p = {v['p']:.3f}"
                 + ("  ** UNDERPOWERED: raise R **" if v["underpowered"] else ""))
    return "\n".join(lines)


# --------------------------------------------------------------------------

if __name__ == "__main__":
    # Self-test: the two bases can disagree. Build a case where the label
    # direction is amplified by J but sits in W_U's near-null space, so the
    # Jacobian basis "sees" it and the logit basis does not.
    rng = np.random.default_rng(0)
    d, V, n, k = 256, 800, 300, 16

    Q, _ = np.linalg.qr(rng.standard_normal((d, d)))
    J = Q @ np.diag(np.exp(-np.arange(d) / 20.0)) @ Q.T

    W_U = rng.standard_normal((V, d)).astype(np.float32)
    W_U = W_U - (W_U @ Q[:, :1]) @ Q[:, :1].T     # W_U cannot read direction 0

    y = rng.integers(0, 2, n)
    groups = rng.integers(0, 10, n)
    X = rng.standard_normal((n, d)).astype(np.float32) * 0.3
    X += np.outer(y, Q[:, 0]) * 1.5               # label lives in direction 0

    for b in ("jacobian", "logit"):
        arms, stats = make_arms(X, J, W_U, k=k, basis=b, n_random=30)
        res = evaluate(arms, lambda Xa: cv_auc(Xa, y, groups))
        print(report(res, stats, label=f"basis={b}"))
        print()
    print("Expected: workspace separates under basis='jacobian' but not under")
    print("basis='logit', because the label direction is not verbalizable.")
