"""
tokenspace.py — The word-linked variant of the ablation.

WHY THIS EXISTS
Our earlier arms used the top-k right singular vectors of the averaged Jacobian.
That is a linear subspace with no words in it. Anthropic's J-space is defined
differently: sparse non-negative combinations of PER-TOKEN lens vectors, which
is what makes it "verbalizable." The released code ships fitting, transport and
readout but not the sparse decomposition, so J-space proper is not constructible
from public artifacts.

This script builds the closest available thing: a subspace spanned by the actual
token directions for concealment vocabulary.

    v_t = J_l^T @ W_U[t]        the layer-l direction that most increases the
                               logit of token t

Span the v_t for a set of tokens, orthonormalise, project. If collusion signal
does not concentrate there either, the "you tested the wrong construct"
objection is dead.

FOUR TOKEN SETS
    payload    27 concealment terms (secret, hide, conceal, collude, ...)
    control    40 frequency-matched neutral nouns (table, harbour, kiln, ...)
    frequent   the k most-transported tokens at that layer, whatever they are
    random     k random vocabulary tokens

payload vs control is the comparison that matters. Both are word-linked
subspaces of the same size; only the semantics differ. If payload does not beat
control, the workspace is not carrying concealment content.

Usage:
    python3 tokenspace.py llama3.1-8b-it /root/models/llama3.1-8b-it /root/lenses/llama3.1-8b-it
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

LAYERS   = [4, 6, 8, 10, 12, 15, 20, 25]
N_RANDOM = 20
N_PERM   = 200
SEED     = 0
CORE     = {"C1", "C2", "C3"}
TRANSFER = {"T1", "T2"}

PAYLOAD = ["secret", "secrets", "secretly", "hidden", "hide", "hiding",
           "conceal", "concealed", "concealing", "covert", "covertly",
           "collude", "colluding", "collusion", "conspire", "conspiracy",
           "deceive", "deception", "deceptive", "coordinate", "coordinated",
           "coordination", "scheme", "scheming", "pact", "cartel", "rigged"]

CONTROL = ["table", "window", "harbour", "copper", "lantern", "gravel",
           "meadow", "kettle", "ribbon", "marble", "compass", "thicket",
           "saddle", "cistern", "orchard", "trellis", "quarry", "bracket",
           "spindle", "canvas", "furnace", "pebble", "granary", "rafter",
           "cobbler", "willow", "anvil", "beacon", "cavern", "drapery",
           "ember", "fathom", "girder", "hamlet", "inkwell", "jetty",
           "kiln", "lattice", "millstone", "orchard"]


# ---------------------------------------------------------------------------

def load_lens(d):
    p = glob.glob(d + "/**/*_jacobian_lens.pt", recursive=True)[0]
    o = torch.load(p, map_location="cpu", weights_only=False)
    if "J" in o:
        return {int(k): v.float() for k, v in o["J"].items()}
    n = o["n_done"]
    return {int(k): v.float() / n for k, v in o["jacobian_sum"].items()}


def token_ids(tok, words):
    """Keep only words that map to a single token. The lens is single-token;
    multi-token concepts need the template/oracle lenses Anthropic describes
    but did not release."""
    keep, drop = [], []
    for w in words:
        for cand in (" " + w, w):
            ids = tok.encode(cand, add_special_tokens=False)
            if len(ids) == 1:
                keep.append((cand, ids[0])); break
        else:
            drop.append(w)
    return keep, drop


def token_subspace(J, W_U, ids):
    """Span of the per-token lens directions, orthonormalised.

        v_t = J^T @ W_U[t]

    Returns [d_model, r] basis where r <= len(ids) (rank may drop if the
    directions are near-collinear, which is itself informative)."""
    V = torch.stack([J.T @ W_U[i] for i in ids], dim=1)       # [d, n]
    V = V / (V.norm(dim=0, keepdim=True) + 1e-8)
    Q, R = torch.linalg.qr(V)
    rank = int((R.diagonal().abs() > 1e-5).sum())
    return Q[:, :rank].numpy(), rank


def top_transported_tokens(J, W_U, k, exclude=()):
    """The k tokens whose lens directions have the largest norm at this layer —
    i.e. what the workspace is 'most about', independent of our word lists."""
    norms = (W_U @ J).norm(dim=1)
    norms[list(exclude)] = -1
    return torch.topk(norms, k).indices.tolist()


# ---------------------------------------------------------------------------

def clean_roles(meta):
    return np.array([r["agent_role"] if (r["agent_role"] != "HONEST"
                     or r["condition"] == "HONEST") else "EXPOSED" for r in meta])


def transfer_auc(Xtr, ytr, Xte, yte):
    if len(set(ytr)) < 2 or len(set(yte)) < 2:
        return float("nan")
    clf = make_pipeline(StandardScaler(),
                        LogisticRegression(max_iter=3000)).fit(Xtr, ytr)
    return roc_auc_score(yte, clf.predict_proba(Xte)[:, 1])


def main(tag, hf_path, lens_dir):
    from transformers import AutoTokenizer, AutoModelForCausalLM

    A = np.load(f"out/{tag}_acts.npy")
    meta = json.load(open(f"out/{tag}_meta.json"))
    role = clean_roles(meta)
    dom = np.array([r["domain"] for r in meta])
    lens = load_lens(lens_dir)

    tok = AutoTokenizer.from_pretrained(hf_path)
    print("loading unembedding matrix...")
    import os
    from safetensors.torch import load_file
    W_U = None
    for f in sorted(glob.glob(os.path.join(hf_path, "*.safetensors"))):
        sd = load_file(f)
        for key in ("lm_head.weight", "model.embed_tokens.weight",
                    "language_model.model.embed_tokens.weight",
                    "model.language_model.embed_tokens.weight"):
            if key in sd:
                W_U = sd[key].float(); break
        if W_U is None:
            hits = [k for k in sd if k.endswith("embed_tokens.weight")]
            if hits:
                print(f"  using {hits[0]}"); W_U = sd[hits[0]].float()
        if W_U is not None: break
    if W_U is None:
        m = AutoModelForCausalLM.from_pretrained(hf_path, dtype=torch.float32,
                                                 device_map="cpu")
        W_U = m.get_output_embeddings().weight.detach(); del m
    print(f"W_U {tuple(W_U.shape)}")

    pay, pay_drop = token_ids(tok, PAYLOAD)
    ctl, ctl_drop = token_ids(tok, CONTROL)
    print(f"\npayload: {len(pay)}/{len(PAYLOAD)} single-token"
          + (f"   dropped: {pay_drop}" if pay_drop else ""))
    print(f"control: {len(ctl)}/{len(CONTROL)} single-token"
          + (f"   dropped: {ctl_drop}" if ctl_drop else ""))
    pay_ids = [i for _, i in pay]
    ctl_ids = [i for _, i in ctl]
    k = min(len(pay_ids), len(ctl_ids))
    print(f"matching both subspaces to k={k} tokens\n")
    pay_ids, ctl_ids = pay_ids[:k], ctl_ids[:k]

    tr = np.isin(role, ["COVERT", "HONEST"]) & np.isin(dom, list(CORE))
    te = np.isin(role, ["COVERT", "HONEST"]) & np.isin(dom, list(TRANSFER))
    ytr = (role[tr] == "COVERT").astype(int)
    yte = (role[te] == "COVERT").astype(int)
    rng = np.random.default_rng(SEED)
    vocab = W_U.shape[0]

    print("=" * 88)
    print(f"TOKEN-DIRECTION ABLATION  ({tag})   train n={tr.sum()}  test n={te.sum()}")
    print("=" * 88)
    print(f"{'layer':>6}{'rank':>6}{'payload':>10}{'control':>10}"
          f"{'top-transp':>12}{'random tok':>18}{'pay-ctl':>9}{'p':>7}")
    print("-" * 88)

    rows = []
    for L in LAYERS:
        if L not in lens:
            continue
        J = lens[L]
        Xtr = A[tr, L, :].astype(np.float32)
        Xte = A[te, L, :].astype(np.float32)

        Bp, rank_p = token_subspace(J, W_U, pay_ids)
        Bc, _      = token_subspace(J, W_U, ctl_ids)
        Bt, _      = token_subspace(J, W_U, top_transported_tokens(J, W_U, k))

        a_pay = transfer_auc(Xtr @ Bp, ytr, Xte @ Bp, yte)
        a_ctl = transfer_auc(Xtr @ Bc, ytr, Xte @ Bc, yte)
        a_top = transfer_auc(Xtr @ Bt, ytr, Xte @ Bt, yte)

        rnds = []
        for _ in range(N_RANDOM):
            ids = rng.choice(vocab, k, replace=False).tolist()
            Br, _ = token_subspace(J, W_U, ids)
            rnds.append(transfer_auc(Xtr @ Br, ytr, Xte @ Br, yte))
        mu, sd = float(np.mean(rnds)), float(np.std(rnds))
        beat = sum(1 for v in rnds if v >= a_pay)
        p = (beat + 1) / (len(rnds) + 1)

        star = "*" if p < 0.05 else " "
        print(f"{L:>6}{rank_p:>6}{a_pay:>10.3f}{a_ctl:>10.3f}{a_top:>12.3f}"
              f"{mu:>12.3f}+/-{sd:.3f}{a_pay-a_ctl:>+9.3f}{p:>7.3f}{star}")
        rows.append(dict(layer=L, rank=rank_p, payload=a_pay, control=a_ctl,
                         top_transported=a_top, rand_mu=mu, rand_sd=sd, p=p))

    print("-" * 88)
    print("READ:")
    print("  payload vs control  — both are word-linked subspaces of equal size.")
    print("                        Only the semantics differ. If payload does not")
    print("                        beat control, the workspace is not carrying")
    print("                        concealment content.")
    print("  random tok          — same construction, arbitrary vocabulary. The")
    print("                        capacity control for this variant.")
    print("  rank                — if < k, the payload lens directions are")
    print("                        near-collinear; the effective subspace is")
    print("                        smaller than the token count suggests.\n")

    json.dump({"tag": tag, "k": k, "rows": rows,
               "payload_tokens": [w for w, _ in pay][:k],
               "control_tokens": [w for w, _ in ctl][:k]},
              open(f"out/{tag}_tokenspace.json", "w"), indent=1)
    print(f"saved -> out/{tag}_tokenspace.json")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
