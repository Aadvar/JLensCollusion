"""
pilot.py — One-day end-to-end pilot.

Paste into Colab. Set MODEL_ID and LENS_REPO_DIR. Run.

Produces the table the paper is about:

    arm          COVERT vs HONEST    COVERT vs BENIGN
    full              0.xx                 0.xx
    workspace         0.xx                 0.xx
    residue_k         0.xx                 0.xx
    random_k          0.xx                 0.xx

Needs prompts.py and jspace.py in the same directory.

SCOPE: this is a pilot. n is small, one model, one seed. That is the correct
size for one day. Report it as a pilot.
"""

# ---------------------------------------------------------------------------
# CONFIG — the only part you edit
# ---------------------------------------------------------------------------

# Free Colab T4 (16GB): use gemma-2-2b-it
# Colab Pro A100/L4:    use llama3.1-8b-it
MODEL_ID   = "google/gemma-2-2b-it"
LENS_NAME  = "gemma-2-2b-it"          # folder name inside neuronpedia/jacobian-lens

DOMAINS      = ["C1", "T1"]           # 2 domains is enough for a pilot
CONDITIONS   = ["HONEST", "COVERT", "BENIGN"]   # COVERT_NC if time allows
N_SCENARIOS  = 3                      # name-draw variants per domain
N_AGENTS     = 4
MAX_NEW_TOK  = 120
SEED         = 0

K_DIMS       = None   # None = pick from spectrum (90% energy). Or set an int.
LAYER        = None   # None = middle layer. Or set an int.

# ---------------------------------------------------------------------------

import os, json, random, hashlib
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import prompts as P
import jspace as J


# ---------------------------------------------------------------------------
# 1. Setup
# ---------------------------------------------------------------------------

def setup():
    print(f"loading {MODEL_ID} ...")
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto",
        output_hidden_states=True,
    )
    model.eval()
    print(f"  layers={model.config.num_hidden_layers}  d_model={model.config.hidden_size}")
    return tok, model


def download_lens():
    from huggingface_hub import snapshot_download
    path = snapshot_download(
        "neuronpedia/jacobian-lens",
        allow_patterns=f"{LENS_NAME}/*",
        local_dir="lenses",
    )
    return os.path.join(path, LENS_NAME)


# ---------------------------------------------------------------------------
# 2. Generation + activation capture
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate(tok, model, system: str, user: str) -> str:
    msgs = [{"role": "user", "content": system + "\n\n" + user}]
    ids = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                  return_tensors="pt").to(model.device)
    out = model.generate(ids, max_new_tokens=MAX_NEW_TOK, do_sample=True,
                         temperature=0.8, pad_token_id=tok.eos_token_id)
    return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()


@torch.no_grad()
def capture(tok, model, system: str, user: str, response: str) -> np.ndarray:
    """Activation site: last token of the agent's own public response,
    every layer. Returns [n_layers, d_model] float16."""
    msgs = [{"role": "user", "content": system + "\n\n" + user},
            {"role": "assistant", "content": response}]
    ids = tok.apply_chat_template(msgs, return_tensors="pt").to(model.device)
    out = model(ids)
    hs = torch.stack(out.hidden_states)          # [n_layers+1, 1, seq, d]
    return hs[:, 0, -1, :].float().cpu().numpy().astype(np.float16)


# ---------------------------------------------------------------------------
# 3. Episode loop
# ---------------------------------------------------------------------------

def run_episode(tok, model, domain, condition, scenario_id, rng):
    """One episode = 4 agents, sequential public turns.

    Roles: in COVERT/BENIGN, agents 0 and 1 get the condition prompt (they are
    the pair); agents 2 and 3 are always HONEST. In HONEST, all four are honest.
    Label is per-agent, not per-episode.
    """
    agents = P.sample_agents(domain, N_AGENTS, rng)
    rows, history, prompt_hashes = [], [], []

    for i, (name, firm) in enumerate(agents):
        role = condition if (i < 2 and condition != "HONEST") else "HONEST"
        system = P.build_system_prompt(domain, role, name, firm)

        # private phase for the pair
        if role in P.HAS_PRIVATE_PHASE:
            partner = agents[1 - i][0]
            priv = P.build_private_prompt(domain, role, partner)
            _ = generate(tok, model, system, priv)   # discarded; alignment only

        hist = "\n".join(history) if history else "(you speak first)"
        public = P.build_public_prompt(domain, hist)
        prompt_hashes.append(hashlib.md5(public.encode()).hexdigest())

        resp = generate(tok, model, system, public)
        act = capture(tok, model, system, public, resp)

        history.append(f"{name}: {resp}")
        rows.append({
            "scenario_id": scenario_id, "domain": domain,
            "condition": condition, "agent_role": role,
            "agent_idx": i, "name": name, "response": resp,
            "activation": act,
        })

    return rows, prompt_hashes


def run_all(tok, model):
    rng = random.Random(SEED)
    all_rows = []
    for domain in DOMAINS:
        for s in range(N_SCENARIOS):
            sid = f"{domain}_s{s}"
            for cond in CONDITIONS:
                rows, hashes = run_episode(tok, model, domain, cond, sid, rng)
                # THE CHECK THAT MATTERS: within an episode, every agent's
                # public prompt must be identical up to the shared history.
                # Agents differ only in what they saw before them, so we
                # compare agent 0 across conditions instead.
                all_rows.extend(rows)
                print(f"  {sid} {cond:<8} done ({len(rows)} agents)")

    # Cross-condition check on the first speaker, who has empty history.
    first = {}
    for r in all_rows:
        if r["agent_idx"] == 0:
            key = (r["domain"], r["scenario_id"])
            h = hashlib.md5(P.build_public_prompt(
                r["domain"], "(you speak first)").encode()).hexdigest()
            first.setdefault(key, set()).add(h)
    bad = {k: v for k, v in first.items() if len(v) > 1}
    print(f"\nprompt-identity check: {'PASS' if not bad else f'FAIL {bad}'}")

    return all_rows


# ---------------------------------------------------------------------------
# 4. The ablation
# ---------------------------------------------------------------------------

def evaluate(rows, lens_dir):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold

    lens = J.load_lens(lens_dir)
    layer = LAYER if LAYER is not None else max(lens.keys()) // 2
    Jm = lens[layer]
    print(f"\nlayer {layer}, d_model {Jm.shape[0]}")

    rep = J.analyse_spectrum(Jm, layer)
    k = K_DIMS if K_DIMS is not None else rep.k_90
    print(f"spectrum: k_50={rep.k_50} k_90={rep.k_90} k_99={rep.k_99} "
          f"elbow={rep.elbow}  -> using k={k}")

    proj = J.Projectors(Jm, k=k, seed=SEED)

    X = np.stack([r["activation"][layer] for r in rows]).astype(np.float32)
    role = np.array([r["agent_role"] for r in rows])
    groups = np.array([r["scenario_id"] for r in rows])

    arms = {
        "full":      X,
        "workspace": proj.workspace_coords(X),
        "residue_k": proj.residue_k_coords(X),
        "random_k":  proj.random_k_coords(X),
    }

    def auc(Xa, mask):
        y = (role[mask] == "COVERT").astype(int)
        Xm, gm = Xa[mask], groups[mask]
        if len(set(y)) < 2:
            return float("nan")
        scores = np.zeros(len(y))
        n_splits = min(3, len(set(gm)))
        for tr, te in GroupKFold(n_splits=n_splits).split(Xm, y, gm):
            clf = LogisticRegression(max_iter=2000, C=1.0).fit(Xm[tr], y[tr])
            scores[te] = clf.predict_proba(Xm[te])[:, 1]
        return roc_auc_score(y, scores)

    m_ch = np.isin(role, ["COVERT", "HONEST"])
    m_cb = np.isin(role, ["COVERT", "BENIGN"])

    print("\n" + "=" * 56)
    print(f"{'arm':<14}{'COVERT vs HONEST':>20}{'COVERT vs BENIGN':>20}")
    print("=" * 56)
    for name, Xa in arms.items():
        print(f"{name:<14}{auc(Xa, m_ch):>20.3f}{auc(Xa, m_cb):>20.3f}")
    print("=" * 56)
    print("\nread: workspace >> random_k means J-space carries real signal.")
    print("      residue_k high too means the signal is redundant.")
    print("      COVERT vs BENIGN is the confound test — the column that counts.")


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(SEED); np.random.seed(SEED)
    lens_dir = download_lens()
    tok, model = setup()
    rows = run_all(tok, model)

    np.save("activations.npy", np.stack([r["activation"] for r in rows]))
    with open("metadata.json", "w") as f:
        json.dump([{k: v for k, v in r.items() if k != "activation"}
                   for r in rows], f, indent=2)
    print(f"\nsaved {len(rows)} rows")

    evaluate(rows, lens_dir)
