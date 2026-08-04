"""
arms.py — Fair-comparison feature transform for JLens vs. probes.

DROP-IN. You already have:

    X       [n_samples, d_model]   activations your probes consume
    y       [n_samples]            labels
    groups  [n_samples]            scenario IDs for grouped CV
    train_probe(X, y) -> model     any of your 6 architectures

This module only transforms X. Same probe, same labels, same CV splits.
The only thing that changes is which subspace the features come from.

    from arms import make_arms
    arms = make_arms(X, lens_matrix_for_layer, k=None)
    for name, Xa in arms.items():
        auc = your_existing_eval(Xa, y, groups)

WHY THIS MATTERS
Comparing "JLens payload alignment (zero-shot)" against "supervised probe on
full activations" is not a comparison of subspaces — it is a comparison of
supervision. Of course the supervised probe wins. Holding the probe fixed and
varying only the input subspace is the comparison that answers the question.

DIMENSION MATCHING IS NOT OPTIONAL
J-space is ~dozens of dims; full activations are thousands. A probe with 80x
the parameters separates almost anything. Every arm below except `full` has
exactly k dimensions, so the comparison is honest. `random_k` is the control
that proves it.
"""

from __future__ import annotations
import numpy as np


# ---------------------------------------------------------------------------

def spectrum(J: np.ndarray) -> dict:
    """SVD spectrum of one layer's Jacobian. Report this plot — k is your one
    free hyperparameter and a reviewer will ask how you picked it."""
    s = np.linalg.svd(J, compute_uv=False)
    energy = np.cumsum(s ** 2) / np.sum(s ** 2)
    return {
        "singular_values": s,
        "k_50": int(np.searchsorted(energy, 0.50) + 1),
        "k_90": int(np.searchsorted(energy, 0.90) + 1),
        "k_99": int(np.searchsorted(energy, 0.99) + 1),
        "d_model": J.shape[0],
    }


def make_arms(X: np.ndarray, J: np.ndarray, k: int | None = None,
              n_random: int = 5, seed: int = 0) -> dict[str, np.ndarray]:
    """Four feature sets from one activation matrix.

    X : [n, d]  activations at ONE layer
    J : [d, d]  Jacobian lens matrix at that SAME layer
    k : workspace dimensionality; None -> 90% spectral energy
    n_random : number of random subspaces to average over (error bars on the
               capacity control; a single draw is noise)

    Returns k-dimensional coordinates, not d-dimensional projections. Same
    information, full rank, honest parameter count.
    """
    X = np.asarray(X, dtype=np.float32)
    d = J.shape[0]
    assert X.shape[1] == d, f"X has d={X.shape[1]}, lens has d={d}"

    if k is None:
        k = spectrum(J)["k_90"]

    _, _, Vt = np.linalg.svd(J)
    V_ws = Vt[:k].T                    # [d, k]  workspace basis
    V_comp = Vt[k:].T                  # [d, d-k] complement
    V_res_k = V_comp[:, :k]            # [d, k]  top-k OF the complement

    out = {
        "full":      X,                # your current baseline
        "workspace": X @ V_ws,         # is the signal in the reportable part?
        "residue_k": X @ V_res_k,      # is it outside, at matched capacity?
    }

    rng = np.random.default_rng(seed)
    for i in range(n_random):
        R, _ = np.linalg.qr(rng.standard_normal((d, k)))
        out[f"random_k_{i}"] = X @ R   # capacity control

    out["_k"] = k                       # metadata, strip before probing
    return out


def summarise(results: dict[str, float], k: int) -> str:
    """Collapse the random_k_* runs into mean +/- sd and print the verdict."""
    rand = [v for name, v in results.items() if name.startswith("random_k_")]
    lines = [f"k = {k}", "-" * 46]
    for name in ("full", "workspace", "residue_k"):
        if name in results:
            lines.append(f"{name:<14} {results[name]:.3f}")
    if rand:
        lines.append(f"{'random_k':<14} {np.mean(rand):.3f} "
                     f"+/- {np.std(rand):.3f}  (n={len(rand)})")
    lines.append("-" * 46)

    ws = results.get("workspace")
    rk = results.get("residue_k")
    rn = np.mean(rand) if rand else None
    if ws is not None and rn is not None:
        if ws > rn + 0.05:
            lines.append("workspace > random -> J-space carries real signal")
        else:
            lines.append("workspace ~= random -> J-space is not special here")
    if rk is not None and rn is not None and rk > rn + 0.05:
        lines.append("residue also > random -> signal exists OUTSIDE the workspace")
        lines.append("   => workspace monitoring is INSUFFICIENT. This is the finding.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Layer sweep — which layer to use is itself a result, not a config choice
# ---------------------------------------------------------------------------

def sweep_layers(X_by_layer: dict[int, np.ndarray], lens: dict[int, np.ndarray],
                 eval_fn, k: int | None = None, layers: list[int] | None = None):
    """Run the four-arm comparison at several depths.

    X_by_layer : {layer: [n, d]}
    lens       : {layer: [d, d]}
    eval_fn    : (X, ) -> auc   closure over your y and groups

    Depth matters: concepts enter the workspace at different points in the
    forward pass. A signal absent early and present late is informative.
    """
    layers = layers or sorted(set(X_by_layer) & set(lens))
    table = {}
    for L in layers:
        arms = make_arms(X_by_layer[L], lens[L], k=k)
        kk = arms.pop("_k")
        table[L] = {name: eval_fn(Xa) for name, Xa in arms.items()}
        table[L]["_k"] = kk
    return table


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Self-test on synthetic data. No model needed.
    rng = np.random.default_rng(0)
    d, n, k = 1024, 400, 48

    U, _ = np.linalg.qr(rng.standard_normal((d, d)))
    V, _ = np.linalg.qr(rng.standard_normal((d, d)))
    J = U @ np.diag(np.exp(-np.arange(d) / 60.0)) @ V.T
    X = rng.standard_normal((n, d)).astype(np.float32)

    sp = spectrum(J)
    print(f"d_model={sp['d_model']}  k_50={sp['k_50']}  "
          f"k_90={sp['k_90']}  k_99={sp['k_99']}")

    arms = make_arms(X, J, k=k)
    kk = arms.pop("_k")
    print(f"\narms at k={kk}:")
    for name, Xa in arms.items():
        print(f"  {name:<14} {str(Xa.shape):<14} rank={np.linalg.matrix_rank(Xa)}")

    # workspace and residue_k must be orthogonal complements in feature space
    _, _, Vt = np.linalg.svd(J)
    ov = np.abs(Vt[:k] @ Vt[k:2*k].T).max()
    print(f"\nworkspace _|_ residue_k overlap: {ov:.2e}  "
          f"{'PASS' if ov < 1e-9 else 'FAIL'}")

    fake = {"full": 0.94, "workspace": 0.71, "residue_k": 0.91,
            **{f"random_k_{i}": 0.58 + 0.01*i for i in range(5)}}
    print("\nexample readout:\n")
    print(summarise(fake, kk))
