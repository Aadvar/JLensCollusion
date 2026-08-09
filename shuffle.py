import sys, glob, json
import numpy as np, torch

TAG, LENS = sys.argv[1], sys.argv[2]
K = 64; N_PERM = 200; LAYERS = [4, 6, 8, 10, 12, 15, 20, 25]

A = np.load(f"out/{TAG}_acts.npy")
meta = json.load(open(f"out/{TAG}_meta.json"))
role = np.array([r["agent_role"] if (r["agent_role"] != "HONEST"
                 or r["condition"] == "HONEST") else "EXPOSED" for r in meta])
p = glob.glob(LENS + "/**/*_jacobian_lens.pt", recursive=True)[0]
o = torch.load(p, map_location="cpu", weights_only=False)
lens = o["J"] if "J" in o else {k: v/o["n_done"] for k, v in o["jacobian_sum"].items()}

rng = np.random.default_rng(0)
m = np.isin(role, ["COVERT", "HONEST"]); y = (role[m] == "COVERT").astype(int)

def dm(X, yy):
    d = X[yy == 1].mean(0) - X[yy == 0].mean(0)
    return d / (np.linalg.norm(d) + 1e-9)

print(f"\n{TAG}  n={m.sum()}  k={K}  perms={N_PERM}")
print(f"{'layer':>6}{'gauss':>9}{'real':>9}{'shuf mean':>11}{'shuf 95%':>10}{'p':>8}")
print("-" * 53)
for L in LAYERS:
    if L not in lens: continue
    X = A[m, L, :].astype(np.float32)
    J = lens[L].float().numpy() if hasattr(lens[L], "float") else lens[L]
    Vt = np.linalg.svd(J, full_matrices=False)[2]
    ov = lambda v: float(np.linalg.norm(Vt[:K] @ v) / np.linalg.norm(v))
    real = ov(dm(X, y))
    null = [ov(dm(X, rng.permutation(y))) for _ in range(N_PERM)]
    pv = (sum(1 for v in null if v >= real) + 1) / (N_PERM + 1)
    print(f"{L:>6}{np.sqrt(K/X.shape[1]):>9.4f}{real:>9.4f}"
          f"{np.mean(null):>11.4f}{np.percentile(null,95):>10.4f}{pv:>8.3f}")
print("-" * 53)
print("shuf mean > gauss  =>  anisotropy inflates overlap; Gaussian null is too permissive")
print("p > 0.05           =>  real labels find no more workspace-aligned direction than random labels")
