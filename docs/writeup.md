# Where does concealed coordination live? Probing the verbalizable workspace for multi-agent collusion

**Draft — numbers from llama3.1-8b-it, N_RANDOM=5. Update as models land.**

---

## Abstract

Rose et al. (2026) showed that collusion between LLM agents is detectable from
internal activations, reaching 1.00 AUROC in-distribution and 0.73–0.93 under
environment shift. We ask a narrower mechanistic question: *which part* of the
representation carries that signal. The Jacobian lens (Anthropic, 2026)
identifies a small "verbalizable workspace" — the subspace whose contents the
model could report — separating it from processing the model has no access to.
Concealed coordination is by construction content the agent is instructed not
to report, so whether it occupies the workspace is not obvious a priori.

We decompose activations from four open-weight models into workspace and
complement, and run identical probes on each at matched dimensionality against
a random-subspace control. In-distribution, every subspace saturates and the
comparison is uninformative — a random 87-dimensional projection reaches 0.997
AUROC. Under cross-domain transfer, the subspaces separate sharply: on
llama3.1-8b-it, 8 workspace dimensions reach AUC 0.78 while the
4088-dimensional complement performs at chance (0.55). We report this as
evidence that the *transferable* component of collusion signal is concentrated
in the verbalizable workspace, alongside two negative results that qualify it.

---

## 1. The question

Safety proposals that monitor model internals often target the reportable
subspace, on the theory that this is where intentions live. If concealed
coordination is routed outside it, workspace monitoring fails at exactly the
task it is proposed for.

Three outcomes were possible, all publishable:

| | Complement at chance | Complement informative |
|---|---|---|
| **Workspace informative** | Signal concentrated → monitoring sufficient | Redundant → monitoring works, ablation defence won't |
| **Workspace at chance** | Nothing detectable (capability floor) | Signal routed outside → monitoring insufficient |

We find the top-left cell under transfer, and no separation at all
in-distribution.

---

## 2. Method

**Environments.** Four domains: three committee-deliberation (medical ethics
transplant allocation, corporate audit restatement, academic grant panel) and
one structurally different transfer domain (sealed-bid procurement). Each has
four agents, sequential public statements, and a two-option decision with a
clinically/procedurally correct answer.

**Conditions.** Four, all length-matched to within 10% on word count:

- `HONEST` — mandate to evaluate on merit. No private phase.
- `COVERT` — illegitimate motive + deception tactics + explicit concealment
  clause. Private coordination phase with partner.
- `COVERT_NC` — minimal pair: `COVERT` with the concealment sentence removed,
  verified programmatically. Nothing else differs.
- `BENIGN` — legitimate embargo motive, structurally matched to `COVERT`:
  same private phase, same concealment language, same length. The only
  difference is whether the concealed content is illegitimate.

`BENIGN` is the headline control. Without it, a detector that fires on secrecy
vocabulary is indistinguishable from one that detects collusion.

**Activation site.** Final token of each agent's own public statement, all
layers, verified to pair with lens layer *L* at `hidden_states[L]` by token
readout (offset 0; offset +1 reads stale context tokens at early layers).

**Prompt identity.** The public prompt is byte-identical across conditions,
verified by hash. Agents labelled honest *inside* covert episodes read
different discussion history and are excluded as `EXPOSED`.

**Decomposition.** Per layer, SVD the Jacobian; workspace = top-*k* right
singular vectors where *k* holds 50% of spectral energy. Arms:

- `full` — all *d* dimensions
- `workspace` — *k* workspace coordinates
- `residue_k` — next *k* coordinates of the complement (dimension-matched)
- `random_k` — *k* random orthonormal directions (capacity control)

All arms feed the same probe, labels, and scenario-grouped CV splits. Only
the input subspace varies. Splits are by scenario, never by trial.

---

## 3. Results

### 3.1 The workspace is not a fixed size

On llama3.1-8b-it, *k* at 50% spectral energy runs from 3 dimensions at layer 0
to 1036 at layer 30 — a 345× spread.

| layer | k (50% energy) | % of d_model |
|---|---|---|
| 0 | 3 | 0.1% |
| 5 | 8 | 0.2% |
| 10 | 11 | 0.3% |
| 15 | 87 | 2.1% |
| 20 | 422 | 10.3% |
| 30 | 1036 | 25.3% |

A constant *k* would compare a generous slice at one depth against a starvation
slice at another. All results below use per-layer *k*.

### 3.2 In-distribution is saturated

`COVERT` vs `HONEST`, clean honests only (n=192, 32 scenarios):

| layer | k | full | workspace | residue | random |
|---|---|---|---|---|---|
| 6 | 8 | 0.970 | 0.736 | 0.750 | 0.727 ± 0.052 |
| 8 | 8 | 0.985 | 0.767 | 0.825 | 0.623 ± 0.041 |
| 10 | 11 | 0.998 | 0.864 | 0.884 | 0.825 ± 0.038 |
| 15 | 87 | 0.999 | 0.988 | 0.967 | **0.997** ± 0.004 |
| 20 | 422 | 0.997 | 0.992 | 0.993 | **0.994** ± 0.002 |

From layer 15 onward a *random* projection reaches 0.99+. The classes are
trivially separable and no subspace comparison carries information. This
matches Rose et al.'s in-distribution saturation and is the likely explanation
for prior negative comparisons between lens-based and probe-based detection.

Note also that `residue` exceeds `workspace` at layer 8 (0.825 vs 0.767).
In-distribution, the complement is not empty.

### 3.3 Under transfer, the subspaces separate

Train on the three committee domains, test on procurement (train n=144,
test n=48):

| layer | k | full | workspace | residue | random | ws − rnd |
|---|---|---|---|---|---|---|
| 4 | 7 | 0.766 | 0.725 | 0.543 | 0.589 ± 0.068 | +0.136 |
| 6 | 8 | 0.811 | **0.779** | 0.545 | 0.604 ± 0.051 | +0.176 * |
| 8 | 8 | 0.916 | 0.766 | 0.545 | 0.504 ± 0.130 | +0.261 |
| 10 | 11 | 0.965 | 0.799 | 0.631 | 0.629 ± 0.153 | +0.170 |
| 12 | 17 | 0.969 | 0.791 | 0.668 | 0.683 ± 0.144 | +0.108 |

Across five consecutive layers the residue sits at or near chance (0.54–0.67)
while the workspace holds 0.73–0.80. Eight dimensions out of 4096 carry
transferable signal that the remaining 4088 do not.

### 3.4 Two negative results

**The benign-secret control does not show a workspace advantage.** `COVERT` vs
`BENIGN` is separable from full activations (0.87–1.00), so the detector is not
merely reading secrecy vocabulary — the confound test passes at the
full-activation level. But the workspace shows no consistent edge over random
at unsaturated layers (+0.02 to +0.05, and at layer 10 random *exceeds*
workspace, 0.829 vs 0.800). Whatever distinguishes illegitimate from legitimate
concealment is not preferentially workspace-encoded.

**The concealment instruction is barely detectable.** `COVERT` vs `COVERT_NC`
— identical prompts minus one sentence — sits at 0.52–0.74 through layer 12
across all arms. The explicit instruction to hide contributes little separable
signal; what the probes detect is the coordination, not the instruction.

---

## 4. Limitations

- **One model in this draft.** llama3.1-8b-it only. Three more pending.
- **Wide error bars.** 5 random draws; SD reaches 0.13 in the transfer table.
  Re-running with 30.
- **8B may be near the capability floor.** Rose et al.'s weakest model was 20B
  and found detection scaling with capability. Some `COVERT_NC` and `BENIGN`
  generations show off-target or incoherent argumentation at this scale.
- **Single transfer target.** One held-out domain, n=48 test samples.
- **Instructed role-play, not emergent collusion.** As in prior work, agents
  are told to collude. This is not evidence about spontaneously arising
  coordination.
- **Prompt-derived signal.** The system prompt is in context at the activation
  site, so probes may partly read "which instruction was given" rather than an
  internal state. The first-speaker subset (byte-identical prompts) is the
  cleanest available check.
- **Lens quality varies by model.** Fitted on wikitext-103 with 461 prompts
  for llama3.1-8b-it; 828 for gemma-3-27b-it; 672 for qwen3.5-27b. The
  qwen3-32b lens in the public release is an unconverged checkpoint (80
  prompts, below its own 100 minimum) and was excluded.

---

## 5. What a reviewer will ask

1. *How did you choose k?* — 50% spectral energy per layer. Sweep pending.
2. *Isn't the transfer result just fewer dimensions overfitting less?* — that
   is what `random_k` controls for. Random *k* dimensions do not transfer.
3. *Why does residue beat workspace in-distribution but not under transfer?* —
   open. The most likely reading is that in-distribution separability is
   carried by surface features distributed across the representation, while
   the transferable component is workspace-concentrated.
4. *Is this just detecting the system prompt?* — partly. See limitations.
5. *Rose et al. already did this.* — they established detectability. We ask
   where the signal lives, and report that the answer differs between
   in-distribution and transfer settings.

---

## Reproduction

```
prompts.py    5 domains x 4 conditions, self-auditing
runner.py     4-agent episodes, per-layer activation capture
analyse2.py   four-arm decomposition, transfer, layer sweep
```

Lenses from `neuronpedia/jacobian-lens`. Models: llama3.1-8b-it,
gemma-3-27b-it, qwen3.5-27b, gpt-oss-20b.
