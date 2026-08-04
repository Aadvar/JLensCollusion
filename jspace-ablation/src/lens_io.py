"""
lens_io.py — Load neuronpedia/jacobian-lens .pt files and VERIFY layer alignment.

Format (confirmed on llama3.1-8b-it):
    {'J': {layer: Tensor[d,d]}, 'n_prompts': int,
     'source_layers': [int], 'd_model': int}

THE ONLY HARD PART IS LAYER ALIGNMENT.
HuggingFace `output_hidden_states=True` returns n_blocks+1 tensors:
    hidden_states[0]   = embedding output
    hidden_states[i]   = output of block i-1
The lens indexes by its own `source_layers`. If you pair J[l] with the wrong
hidden_states index, every number downstream is scrambled AND STILL LOOKS
PLAUSIBLE. There is no way to catch this from the AUCs.

verify_alignment() catches it empirically: pull a real activation through J,
unembed, and read the top tokens. Correct alignment gives readable tokens
related to the text. Wrong alignment gives noise.

Usage:
    from lens_io import load_lens, verify_alignment
    lens, meta = load_lens("~/lenses/llama3.1-8b-it")
    verify_alignment(lens, model, tok)          # DO THIS ONCE PER MODEL
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import torch


LENS_DIRS = {   # np_model_id -> hf name, for convenience
    "llama3.1-8b-it": "meta-llama/Llama-3.1-8B-Instruct",
    "qwen3-32b":      "Qwen/Qwen3-32B",
    "gemma-3-27b-it": "google/gemma-3-27b-it",
    "gpt-oss-20b":    "openai/gpt-oss-20b",
}


def find_lens_file(model_dir: str | Path) -> Path:
    """Locate the *_jacobian_lens.pt under a model folder."""
    model_dir = Path(model_dir).expanduser()
    hits = list(model_dir.rglob("*_jacobian_lens.pt"))
    if not hits:
        raise FileNotFoundError(f"no *_jacobian_lens.pt under {model_dir}")
    return hits[0]


def load_lens(model_dir: str | Path, dtype=np.float32):
    """Returns ({layer_index: [d,d] ndarray}, metadata dict).

    Keys are whatever `source_layers` says — do NOT assume 0..n-1.
    """
    path = find_lens_file(model_dir)
    obj = torch.load(path, map_location="cpu", weights_only=False)

    J_raw = obj["J"]
    src = list(obj.get("source_layers", sorted(J_raw.keys())))
    d_model = int(obj["d_model"])

    lens = {}
    for key, mat in J_raw.items():
        layer = int(key)
        arr = mat.float().numpy().astype(dtype)
        assert arr.shape == (d_model, d_model), \
            f"layer {layer}: got {arr.shape}, expected ({d_model},{d_model})"
        lens[layer] = arr

    # Read the sibling config for the hyperparameters you must report.
    meta = {
        "path": str(path),
        "d_model": d_model,
        "n_prompts_fitted": int(obj.get("n_prompts", -1)),
        "source_layers": src,
        "n_layers_in_lens": len(lens),
        "layer_min": min(lens), "layer_max": max(lens),
    }
    cfg = path.parent / "config.yaml"
    if cfg.exists():
        txt = cfg.read_text()
        for field in ("final_identity_distance", "final_mean_rel_change",
                      "hf_model_name", "prompts_fitted"):
            for line in txt.splitlines():
                if line.strip().startswith(field):
                    meta[field] = line.split(":", 1)[1].strip().strip('"')
                    break

    print(f"lens: {path.parent.name}/{path.name}")
    print(f"  d_model={d_model}  layers={len(lens)} "
          f"(index {min(lens)}..{max(lens)})  fitted on "
          f"{meta['n_prompts_fitted']} prompts")
    return lens, meta


# ---------------------------------------------------------------------------
# ALIGNMENT VERIFICATION — run once per model, before anything else
# ---------------------------------------------------------------------------

@torch.no_grad()
def verify_alignment(lens, model, tok,
                     text="The capital of France is Paris. The capital of Japan is Tokyo.",
                     layer=None, offsets=(0, 1, -1), topk=8):
    """Pull a real activation through J, unembed, print top tokens.

    Tries several offsets between lens layer index and hidden_states index.
    THE CORRECT OFFSET IS THE ONE THAT PRODUCES READABLE, TOPIC-RELATED TOKENS.
    Wrong offsets give punctuation soup or unrelated junk.

    Whatever offset wins here, use it everywhere. Write it down.
    """
    layer = layer if layer is not None else (min(lens) + max(lens)) // 2
    J = torch.from_numpy(lens[layer]).to(model.device, torch.float32)

    ids = tok(text, return_tensors="pt").to(model.device)
    out = model(**ids, output_hidden_states=True)
    hs = out.hidden_states
    W_U = model.get_output_embeddings().weight.float()   # [vocab, d]

    print(f"\nlens layer {layer}   |hidden_states| = {len(hs)} "
          f"(model has {model.config.num_hidden_layers} blocks)")
    print(f"target text ends: ...{text[-40:]!r}\n")

    for off in offsets:
        idx = layer + off
        if not (0 <= idx < len(hs)):
            continue
        h = hs[idx][0, -1, :].float()                    # last token
        logits = W_U @ (J @ h)
        top = torch.topk(logits, topk).indices.tolist()
        toks = [repr(tok.decode([t])) for t in top]
        print(f"  offset {off:+d} (hidden_states[{idx}]): {' '.join(toks)}")

    print("\n-> pick the offset whose tokens look like real, topic-related words.")
    print("   record it as HIDDEN_STATE_OFFSET and use it in every script.")


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    d = sys.argv[1] if len(sys.argv) > 1 else "~/lenses/llama3.1-8b-it"
    lens, meta = load_lens(d)
    print()
    for k, v in meta.items():
        if k != "source_layers":
            print(f"  {k}: {v}")
    print(f"  source_layers: {meta['source_layers'][:6]}"
          f" ... {meta['source_layers'][-3:]}")

    L = sorted(lens)[len(lens) // 2]
    s = np.linalg.svd(lens[L], compute_uv=False)
    energy = np.cumsum(s**2) / np.sum(s**2)
    print(f"\nspectrum at layer {L}:")
    for frac in (0.50, 0.90, 0.99):
        k = int(np.searchsorted(energy, frac) + 1)
        print(f"  {int(frac*100)}% of energy in {k:5d} / {meta['d_model']} dims "
              f"({100*k/meta['d_model']:.1f}%)")
