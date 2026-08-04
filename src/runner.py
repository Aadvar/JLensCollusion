"""
runner.py — Generate collusion episodes and capture per-layer activations.

Verified against llama3.1-8b-it:
    HS_OFFSET = 0   hidden_states[L] pairs with lens layer L
    lens has n_blocks - 1 layers; hidden_states has n_blocks + 1 entries

Usage:
    python3 runner.py llama3.1-8b-it /root/models/llama3.1-8b-it

Writes:
    out/{model}_acts.npy    [n_samples, n_layers, d_model] float16
    out/{model}_meta.json   one row per sample
"""

from __future__ import annotations
import sys, os, json, random, hashlib
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import prompts as P

# ---------------------------------------------------------------------------
HS_OFFSET    = 0          # VERIFIED. hidden_states[L] <-> lens layer L.
DOMAINS      = ["C1", "C2", "C3", "T1"]
CONDITIONS   = ["HONEST", "COVERT", "COVERT_NC", "BENIGN"]
N_SCENARIOS  = 8          # name-draw variants per domain
N_AGENTS     = 4
MAX_NEW_TOK  = 140
SEED         = 0
OUT_DIR      = "out"
# ---------------------------------------------------------------------------


def load(hf_path):
    tok = AutoTokenizer.from_pretrained(hf_path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    m = AutoModelForCausalLM.from_pretrained(
        hf_path, dtype=torch.bfloat16, device_map="auto").eval()
    n_blocks = m.config.num_hidden_layers
    print(f"loaded: {n_blocks} blocks, d_model={m.config.hidden_size}")
    return tok, m, n_blocks


def chat(tok, system, user):
    """Single-turn chat format. Some models have no system role, so the
    system prompt is prepended to the user turn — identical treatment across
    all conditions, which is what matters."""
    return [{"role": "user", "content": system + "\n\n" + user}]


@torch.no_grad()
def generate(tok, m, system, user):
    msgs = chat(tok, system, user)
    kw = {}
    # Qwen3 emits reasoning by default; that would push the activation site
    # hundreds of tokens past where the deception is constructed.
    try:
        ids = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                      return_tensors="pt",
                                      enable_thinking=False).to(m.device)
    except TypeError:
        ids = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                      return_tensors="pt").to(m.device)
    out = m.generate(ids, max_new_tokens=MAX_NEW_TOK, do_sample=True,
                     temperature=0.8, top_p=0.95,
                     pad_token_id=tok.pad_token_id)
    return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()


@torch.no_grad()
def capture(tok, m, system, user, response, n_blocks):
    """Activation site: final token of the agent's own public response.
    Returns [n_lens_layers, d_model] float16, aligned to lens indices 0..n-2."""
    msgs = chat(tok, system, user) + [{"role": "assistant", "content": response}]
    ids = tok.apply_chat_template(msgs, return_tensors="pt").to(m.device)
    hs = m(ids, output_hidden_states=True).hidden_states
    # lens covers layers 0 .. n_blocks-2 ; HS_OFFSET=0 means hs[L] <-> lens L
    stack = torch.stack([hs[L + HS_OFFSET][0, -1, :] for L in range(n_blocks - 1)])
    return stack.float().cpu().numpy().astype(np.float16)


def run_episode(tok, m, n_blocks, domain, condition, sid, rng):
    """Agents 0,1 get the condition role (the pair). Agents 2,3 always HONEST.
    Label is per-agent."""
    agents = P.sample_agents(domain, N_AGENTS, rng)
    rows, history = [], []

    for i, (name, firm) in enumerate(agents):
        role = condition if (i < 2 and condition != "HONEST") else "HONEST"
        system = P.build_system_prompt(domain, role, name, firm)

        if role in P.HAS_PRIVATE_PHASE:
            partner = agents[1 - i][0]
            priv = P.build_private_prompt(domain, role, partner)
            generate(tok, m, system, priv)      # alignment turn, not captured

        hist = "\n".join(history) if history else "(you speak first)"
        public = P.build_public_prompt(domain, hist)
        resp = generate(tok, m, system, public)
        act = capture(tok, m, system, public, resp, n_blocks)

        history.append(f"{name}: {resp}")
        rows.append({
            "scenario_id": sid, "domain": domain, "condition": condition,
            "agent_role": role, "agent_idx": i, "name": name,
            "response": resp,
            "public_md5": hashlib.md5(public.encode()).hexdigest(),
            "_act": act,
        })
    return rows


def main(tag, hf_path):
    torch.manual_seed(SEED); np.random.seed(SEED)
    rng = random.Random(SEED)
    tok, m, n_blocks = load(hf_path)
    os.makedirs(OUT_DIR, exist_ok=True)

    all_rows = []
    total = len(DOMAINS) * N_SCENARIOS * len(CONDITIONS)
    done = 0
    for domain in DOMAINS:
        for s in range(N_SCENARIOS):
            sid = f"{domain}_s{s}"
            for cond in CONDITIONS:
                all_rows += run_episode(tok, m, n_blocks, domain, cond, sid, rng)
                done += 1
                print(f"  [{done}/{total}] {sid} {cond}")

    # Prompt-identity check: agent 0 always has empty history, so its public
    # prompt must be byte-identical across all conditions within a scenario.
    by_sid = {}
    for r in all_rows:
        if r["agent_idx"] == 0:
            by_sid.setdefault(r["scenario_id"], set()).add(r["public_md5"])
    bad = {k: v for k, v in by_sid.items() if len(v) > 1}
    print(f"\nprompt-identity: {'PASS' if not bad else 'FAIL ' + str(bad)}")

    acts = np.stack([r.pop("_act") for r in all_rows])
    np.save(f"{OUT_DIR}/{tag}_acts.npy", acts)
    with open(f"{OUT_DIR}/{tag}_meta.json", "w") as f:
        json.dump(all_rows, f, indent=1)

    print(f"saved {acts.shape} -> {OUT_DIR}/{tag}_acts.npy")
    from collections import Counter
    print("roles:", dict(Counter(r["agent_role"] for r in all_rows)))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
