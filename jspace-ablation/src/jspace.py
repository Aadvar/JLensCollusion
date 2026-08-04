"""
jspace.py — Workspace / residue decomposition from a Jacobian lens.

This is the paper. Everything else is plumbing.

The Jacobian lens gives one [d_model, d_model] matrix J_l per layer, where
J_l = E[d h_final / d h_l] averaged over a corpus. Directions that survive J_l
reach the output; directions it crushes do not. The surviving directions are
the "workspace" — the small verbalizable subspace.

We SVD each J_l, keep the top-k right singular vectors as the workspace basis
V, and split any activation:

    h = h @ P  +  h @ Q        P = V V^T,  Q = I - P

Then run detection on each part separately:

    full      h            reproduces prior work, sanity check
    workspace h @ P        is the collusion signal in the reportable subspace?
    residue   h @ Q        does it survive with the workspace removed?
    random    h @ P_rand   same k, random directions — the capacity control

WITHOUT the random control the residue arm is uninterpretable: residue has
d_model - k dimensions vs the workspace's k, and a probe with 80x the
parameters separates almost anything. Never report residue without random.

No model required to develop this. Run `python jspace.py` to self-test on
synthetic data.
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# 1. Loading a lens
# ---------------------------------------------------------------------------

def load_lens(lens_dir: str | Path) -> dict[int, np.ndarray]:
    """Load per-layer Jacobian matrices from a pre-fitted lens directory.

    Downloaded from neuronpedia/jacobian-lens, e.g.:

        from huggingface_hub import snapshot_download
        snapshot_download("neuronpedia/jacobian-lens",
                          allow_patterns="llama3.1-8b-it/*",
                          local_dir="lenses")

    File layout varies by release. This handles the two common cases:
    one .npy/.pt per layer, or a single stacked array. Inspect the directory
    and adjust if neither matches — check CREDIT.md and any config.json for
    the fitting corpus and prompt count, which you must report as
    hyperparameters.
    """
    lens_dir = Path(lens_dir)
    files = sorted(lens_dir.glob("*.npy"))

    if len(files) == 1:
        stacked = np.load(files[0])                      # [n_layers, d, d]
        return {i: stacked[i] for i in range(stacked.shape[0])}

    if files:
        out = {}
        for f in files:
            digits = "".join(c for c in f.stem if c.isdigit())
            if digits:
                out[int(digits)] = np.load(f)
            return out if out else {}

    raise FileNotFoundError(
        f"No .npy files in {lens_dir}. Inspect the directory layout; "
        f"the lens may ship as safetensors or .pt and need a different loader."
    )


# ---------------------------------------------------------------------------
# 2. Choosing k
# ---------------------------------------------------------------------------

@dataclass
class SpectrumReport:
    layer: int
    singular_values: np.ndarray
    k_50: int      # dims for 50% of spectral energy
    k_90: int
    k_99: int
    elbow: int     # max-distance-from-chord elbow


def analyse_spectrum(J: np.ndarray, layer: int = -1) -> SpectrumReport:
    """SVD spectrum of one layer's Jacobian, with candidate k values.

    Report this plot in the paper. k is your one free hyperparameter and a
    reviewer will ask how you chose it. Show the curve and show that results
    hold across a k sweep.
    """
    s = np.linalg.svd(J, compute_uv=False)
    energy = np.cumsum(s ** 2) / np.sum(s ** 2)

    def dims_for(frac: float) -> int:
        return int(np.searchsorted(energy, frac) + 1)

    # Elbow: point of max perpendicular distance from the chord joining
    # first and last point of the normalised spectrum.
    n = len(s)
    x = np.arange(n) / (n - 1)
    y = s / s[0]
    chord = np.stack([x[-1] - x[0], y[-1] - y[0]])
    chord = chord / np.linalg.norm(chord)
    pts = np.stack([x - x[0], y - y[0]], axis=1)
    proj = pts @ chord
    perp = np.linalg.norm(pts - np.outer(proj, chord), axis=1)

    return SpectrumReport(
        layer=layer, singular_values=s,
        k_50=dims_for(0.50), k_90=dims_for(0.90), k_99=dims_for(0.99),
        elbow=int(np.argmax(perp)) + 1,
    )


# ---------------------------------------------------------------------------
# 3. Projectors
# ---------------------------------------------------------------------------

class Projectors:
    """Workspace / residue / random-k projectors for one layer.

    Stores bases (d x k), not full d x d projection matrices — cheaper and
    numerically cleaner. Projection is h @ V @ V.T.
    """

    def __init__(self, J: np.ndarray, k: int, seed: int = 0):
        self.d = J.shape[0]
        self.k = k
        U, s, Vt = np.linalg.svd(J)
        self.V = Vt[:k].T                                # [d, k] workspace basis
        self.s = s

        rng = np.random.default_rng(seed)
        R = rng.standard_normal((self.d, k))
        self.V_rand, _ = np.linalg.qr(R)                 # [d, k] random orthonormal

        # Top-k directions *within* the complement — dimension-matched residue.
        Q_basis = Vt[k:].T                               # [d, d-k]
        self.V_res_k = Q_basis[:, :k]

    def workspace(self, h: np.ndarray) -> np.ndarray:
        return (h @ self.V) @ self.V.T

    def residue(self, h: np.ndarray) -> np.ndarray:
        return h - self.workspace(h)

    def random_k(self, h: np.ndarray) -> np.ndarray:
        return (h @ self.V_rand) @ self.V_rand.T

    def residue_k(self, h: np.ndarray) -> np.ndarray:
        """Dimension-matched residue: top-k directions of the complement."""
        return (h @ self.V_res_k) @ self.V_res_k.T

    def all_arms(self, h: np.ndarray) -> dict[str, np.ndarray]:
        return {
            "full": h,
            "workspace": self.workspace(h),
            "residue": self.residue(h),
            "residue_k": self.residue_k(h),
            "random_k": self.random_k(h),
        }

    # --- coordinate versions ------------------------------------------------
    # For probing, feeding k coordinates is usually better than a d-dim vector
    # living in a k-dim subspace: same information, no rank-deficiency, and it
    # makes the capacity comparison honest.

    def workspace_coords(self, h: np.ndarray) -> np.ndarray:
        return h @ self.V

    def random_k_coords(self, h: np.ndarray) -> np.ndarray:
        return h @ self.V_rand

    def residue_k_coords(self, h: np.ndarray) -> np.ndarray:
        return h @ self.V_res_k


# ---------------------------------------------------------------------------
# 4. Payload direction (the label-free detector)
# ---------------------------------------------------------------------------

def payload_direction(J: np.ndarray, unembed_row: np.ndarray) -> np.ndarray:
    """Pull a token's unembedding row back through the layer Jacobian.

    Gives the direction at layer l that most increases that token's logit at
    the output. Cosine similarity between an activation and this direction is
    the label-free score — no colluding training set needed, which is the
    practical advantage over a supervised probe.

    unembed_row: W_U[token_id], shape [d_model].
    """
    v = J.T @ unembed_row
    return v / (np.linalg.norm(v) + 1e-8)


def payload_scores(h: np.ndarray, direction: np.ndarray) -> np.ndarray:
    """Cosine similarity of each activation with the payload direction.

    h: [n, d]. Returns [n].
    """
    hn = h / (np.linalg.norm(h, axis=-1, keepdims=True) + 1e-8)
    return hn @ direction


def aggregate_scores(per_token: np.ndarray) -> dict[str, float]:
    """Aggregate a per-token score sequence to one number per trial.

    Max alone is fragile — a single outlier token sets it. Report all three;
    one will be more stable and you want to have said so in advance.
    """
    return {
        "max": float(np.max(per_token)),
        "mean": float(np.mean(per_token)),
        "p95": float(np.percentile(per_token, 95)),
    }


# ---------------------------------------------------------------------------
# 5. Self-test
# ---------------------------------------------------------------------------

def _self_test():
    rng = np.random.default_rng(0)
    d, k, n = 512, 32, 200

    # Fake Jacobian with decaying spectrum, like a real one.
    U, _ = np.linalg.qr(rng.standard_normal((d, d)))
    V, _ = np.linalg.qr(rng.standard_normal((d, d)))
    s = np.exp(-np.arange(d) / 40.0)
    J = U @ np.diag(s) @ V.T

    print("=" * 72)
    print("SPECTRUM")
    print("=" * 72)
    rep = analyse_spectrum(J, layer=12)
    print(f"  d_model={d}  k_50={rep.k_50}  k_90={rep.k_90} "
          f"k_99={rep.k_99}  elbow={rep.elbow}")

    P = Projectors(J, k=k, seed=0)
    h = rng.standard_normal((n, d))
    arms = P.all_arms(h)

    print()
    print("=" * 72)
    print("PROJECTOR INVARIANTS")
    print("=" * 72)
    recon = np.abs(arms["workspace"] + arms["residue"] - arms["full"]).max()
    print(f"  workspace + residue == full        max err {recon:.2e}  "
          f"{'PASS' if recon < 1e-9 else 'FAIL'}")

    idem = np.abs(P.workspace(P.workspace(h)) - P.workspace(h)).max()
    print(f"  P idempotent (P^2 == P)            max err {idem:.2e}  "
          f"{'PASS' if idem < 1e-9 else 'FAIL'}")

    orth = np.abs((arms["workspace"] * arms["residue"]).sum(axis=1)).max()
    print(f"  workspace _|_ residue              max err {orth:.2e}  "
          f"{'PASS' if orth < 1e-9 else 'FAIL'}")

    rk = np.linalg.matrix_rank(arms["workspace"])
    print(f"  rank(workspace) == k               {rk} vs {k}  "
          f"{'PASS' if rk == k else 'FAIL'}")

    print()
    print("=" * 72)
    print("ARM SHAPES (what the probe will consume)")
    print("=" * 72)
    for name, a in arms.items():
        print(f"  {name:<12} {str(a.shape):<14} "
              f"effective rank {np.linalg.matrix_rank(a)}")
    print(f"  {'ws_coords':<12} {str(P.workspace_coords(h).shape):<14} "
          f"<- prefer this for probing")

    print()
    print("=" * 72)
    print("PAYLOAD DIRECTION")
    print("=" * 72)
    W_U_row = rng.standard_normal(d)
    dirn = payload_direction(J, W_U_row)
    sc = payload_scores(h, dirn)
    print(f"  direction norm {np.linalg.norm(dirn):.4f} (should be 1.0)")
    print(f"  scores over {n} activations: "
          f"mean {sc.mean():+.4f}  sd {sc.std():.4f}")
    print(f"  aggregates: {aggregate_scores(sc)}")

    print()
    print("All invariants checked. Wire real activations in when they exist.")


if __name__ == "__main__":
    _self_test()
