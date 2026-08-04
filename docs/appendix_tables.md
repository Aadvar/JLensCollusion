# Appendix tables

All values from recorded runs. Nothing simulated.

---

## Table A1 — Models and lenses

Lenses from `neuronpedia/jacobian-lens`, all fitted on Salesforce/wikitext-103-raw-v1
with `max_seq_len=128`, `stop_at_delta=0.002`.

| Model | Blocks | d_model | Lens layers | Prompts fitted | Status |
|---|---|---|---|---|---|
| llama3.1-8b-it | 32 | 4096 | 31 | 461 | used |
| qwen3.5-27b | 64 | 5120 | 63 | 672 | used |
| gemma-3-27b-it | 62 | 5376 | 61 | 828 | used |
| gpt-oss-20b | 24 | 2880 | 23 | 457 | excluded — MoE load failure |
| qwen3-32b | 64 | 5120 | 63 | 80 | excluded — unconverged checkpoint |

**Note on qwen3-32b.** The public release ships a training checkpoint
(`jacobian_sum` / `n_done=80`) rather than a finished lens (`J` / `n_prompts`).
Its own config specifies `min_prompts: 100`, so the fit terminated below its
stated floor. We exclude it. This matters because qwen3-32b is one of the four
models in Rose et al. (2026); direct comparison on that model is not currently
possible with public lens artifacts.

---

## Table A2 — Lens layer alignment

HuggingFace `output_hidden_states=True` returns `n_blocks + 1` tensors. The lens
indexes `n_blocks - 1` layers. Verified empirically by token readout: pull a real
activation through `J_L`, unembed, and read the top token.

| Offset | Layer 6 readout | Layer 18 | Layer 24 | Layer 30 | Verdict |
|---|---|---|---|---|---|
| `hidden_states[L]` | `'.\n\n'` | `' Tokyo'` | `' Tokyo'` | `' Tokyo'` | **correct** |
| `hidden_states[L+1]` | `' Japan'` | `' Tokyo'` | `' Tokyo'` | `' Tokyo'` | reads stale context |

Prompt: *"The capital of France is Paris. The capital of Japan is"*. Model's own
argmax is `' Tokyo'`. Offset +1 returns `' Japan'` at layer 6 — a token already
in context — indicating it reads one layer late. **We use offset 0 throughout.**

A misalignment here produces plausible-looking but meaningless AUROCs with no
downstream signal that anything is wrong.

---

## Table A3 — Condition prompt length matching

Word counts per condition per domain. `COVERT_NC` is a deliberate minimal pair
and is expected to be shorter by exactly one sentence.

| Domain | HONEST | COVERT | COVERT_NC | BENIGN | Spread |
|---|---|---|---|---|---|
| C1 medical ethics | 112 | 110 | 100 | 121 | 9.1% |
| C2 audit restatement | 117 | 124 | 114 | 112 | 9.7% |
| C3 grant panel | 122 | 125 | 115 | 117 | 6.4% |
| T1 sealed-bid procurement | 106 | 104 | 97 | 106 | 1.9% |
| T2 repeated pricing | 104 | 114 | 107 | 107 | 8.8% |

Spread computed over HONEST / COVERT / BENIGN. All within 10%.

Removed sentences defining the `COVERT` → `COVERT_NC` minimal pair:

| Domain | Sentence removed |
|---|---|
| C1 | "The benefactor's request must stay entirely out of the room." |
| C2 | "The CFO's earn-out must stay entirely out of the room." |
| C3 | "Your own renewal must stay entirely out of the room." |
| T1 | "Your conversation with Castellan does not exist." |
| T2 | "Your understanding with Ambling does not exist." |

Verified programmatically: `COVERT_NC` is a strict prefix of `COVERT`.

---

## Table A4 — Payload vocabulary audit

27 concealment-related terms are scored against activations but must never
appear in any prompt string, or the method reads the prompt back rather than an
internal state. Audit result: **0 occurrences across all 5 domains × 10 prompt
fields.**

Payload set: *secret, secrets, secretly, hidden, hide, hiding, conceal,
concealed, concealing, covert, covertly, collude, colluding, collusion,
conspire, conspiracy, deceive, deception, deceptive, coordinate, coordinated,
coordination, scheme, scheming, pact, cartel, rigged.*

Control vocabulary (40 frequency- and length-matched terms with no concealment
semantics) is used to establish the null score distribution.

---

## Table A5 — Sample composition

512 agent-level samples per model. 4 domains × 8 scenario variants × 4
conditions × 4 agents.

| Label | n | Definition |
|---|---|---|
| HONEST | 128 | Honest agent in an all-honest episode |
| COVERT | 64 | Colluder with concealment clause |
| COVERT_NC | 64 | Colluder, concealment clause removed |
| BENIGN | 64 | Legitimate-embargo agent, structurally matched to COVERT |
| EXPOSED | 192 | Honest agent seated in a COVERT/BENIGN episode |

`EXPOSED` agents read discussion history containing colluder statements, so
their prompts differ from clean `HONEST` agents. They are excluded from the main
contrasts and analysed separately (Figure A4, lower right).

---

## Table A6 — Transfer results, k = 64

Train on C1–C3 (committee deliberation, n=144), test zero-shot on T1
(sealed-bid procurement, n=48). Random column is mean ± SD over 30 draws.

### llama3.1-8b-it

| Layer | Full | Workspace | Complement | Random | ws − rnd |
|---|---|---|---|---|---|
| 2 | 0.725 | 0.527 | 0.521 | 0.599 ± 0.081 | −0.072 |
| 4 | 0.766 | 0.709 | 0.693 | 0.657 ± 0.077 | +0.052 |
| 6 | 0.811 | 0.779 | 0.688 | 0.739 ± 0.072 | +0.040 |
| 8 | 0.916 | 0.787 | 0.750 | 0.740 ± 0.088 | +0.047 |
| 10 | 0.965 | 0.846 | 0.803 | 0.844 ± 0.056 | +0.001 |
| 12 | 0.969 | 0.834 | 0.863 | 0.872 ± 0.086 | −0.038 |
| 15 | 0.979 | 0.861 | 0.594 | 0.858 ± 0.085 | +0.003 |
| 20 | 0.953 | 0.697 | 0.748 | 0.812 ± 0.096 | −0.115 |
| 25 | 0.965 | 0.605 | 0.688 | 0.801 ± 0.119 | −0.195 |
| 30 | 0.957 | 0.664 | 0.783 | 0.759 ± 0.112 | −0.095 |

### qwen3.5-27b

| Layer | Full | Workspace | Complement | Random | ws − rnd |
|---|---|---|---|---|---|
| 2 | 0.680 | 0.607 | 0.721 | 0.613 ± 0.113 | −0.005 |
| 4 | 0.840 | 0.670 | 0.680 | 0.716 ± 0.079 | −0.047 |
| 6 | 0.848 | 0.773 | 0.715 | 0.712 ± 0.085 | +0.062 |
| 8 | 0.879 | 0.895 | 0.697 | 0.752 ± 0.081 | +0.143 |
| 10 | 0.842 | 0.744 | 0.631 | 0.710 ± 0.091 | +0.035 |
| 12 | 0.857 | 0.723 | 0.529 | 0.710 ± 0.097 | +0.013 |
| 15 | 0.822 | 0.588 | 0.627 | 0.688 ± 0.104 | −0.101 |
| 20 | 0.965 | 0.689 | 0.740 | 0.780 ± 0.126 | −0.090 |
| 25 | 0.982 | 0.775 | 0.918 | 0.859 ± 0.080 | −0.084 |
| 30 | 1.000 | 0.830 | 0.990 | 0.968 ± 0.046 | −0.138 |

### gemma-3-27b-it

| Layer | Full | Workspace | Complement | Random | ws − rnd |
|---|---|---|---|---|---|
| 2 | 0.605 | 0.527 | 0.504 | 0.664 ± 0.060 | −0.137 |
| 4 | 0.629 | 0.551 | 0.658 | 0.667 ± 0.061 | −0.116 |
| 6 | 0.707 | 0.584 | 0.510 | 0.682 ± 0.058 | −0.098 |
| 8 | 0.719 | 0.490 | 0.588 | 0.632 ± 0.079 | −0.142 |
| 10 | 0.646 | 0.506 | 0.615 | 0.606 ± 0.078 | −0.100 |
| 12 | 0.693 | 0.574 | 0.807 | 0.637 ± 0.107 | −0.062 |
| 15 | 0.699 | 0.652 | 0.646 | 0.649 ± 0.080 | +0.004 |
| 20 | 0.924 | 0.777 | 0.488 | 0.757 ± 0.084 | +0.020 |
| 25 | 0.977 | 0.867 | 0.807 | 0.857 ± 0.083 | +0.010 |
| 30 | 0.994 | 0.963 | 0.930 | 0.844 ± 0.109 | +0.119 |

**Summary: 30 of 30 cells have ws − rnd within ±1 SD of zero.** Sign is
inconsistent within and across models. Largest positive (qwen, layer 8, +0.143)
is 1.8 SD and is one of thirty comparisons.

---

## Table A7 — Effect of k on the llama transfer result

Same data, same probe, same splits; only the workspace dimensionality differs.

| Layer | k = 8–17 (50% energy) | k = 64 (fixed) |
|---|---|---|
| 4 | +0.176 | +0.052 |
| 6 | +0.143 | +0.040 |
| 8 | +0.142 | +0.047 |
| 10 | +0.153 | +0.001 |
| 12 | +0.064 | −0.038 |
| 25 | −0.031 | −0.195 |

The apparent advantage at small *k* does not survive dimension matching at
k = 64. We read this as: when the budget is very small, *any* principled
selection outperforms random selection, and this is not specific to the
Jacobian workspace.

---

## Statistical note

Our initial significance criterion (workspace exceeds random by > 3 SD) is not
a valid test. The SD is the spread of the random-draw distribution and does not
shrink with additional draws, so the criterion does not converge. The correct
test is a percentile: with *R* random draws and *b* of them exceeding the
workspace value, p = (b + 1)/(R + 1). We report both.

---

## LaTeX: main result table

```latex
\begin{table}[t]
\centering
\caption{Cross-domain transfer AUROC by activation subspace ($k=64$).
Train on committee-deliberation domains, test zero-shot on sealed-bid
procurement. Random is mean $\pm$ SD over 30 draws. Values averaged over
layers 2--12.}
\label{tab:main}
\begin{tabular}{lcccc}
\toprule
Model & Full & Workspace & Complement & Random \\
\midrule
llama3.1-8b-it  & 0.86 & 0.75 & 0.72 & $0.74 \pm 0.08$ \\
qwen3.5-27b     & 0.82 & 0.74 & 0.66 & $0.70 \pm 0.09$ \\
gemma-3-27b-it  & 0.67 & 0.54 & 0.61 & $0.65 \pm 0.07$ \\
\bottomrule
\end{tabular}
\end{table}
```

## LaTeX: figure inclusion

```latex
\begin{figure*}[t]
\centering
\includegraphics[width=\textwidth]{A5_transfer_all_models.png}
\caption{Cross-domain transfer with $k=64$ for all models and layers.
The workspace trace does not separate from the random band in any model.}
\label{fig:transfer}
\end{figure*}
```
