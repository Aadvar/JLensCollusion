# What I ran and what came out — plain version

*For Aadarsh, Andrew, Mehek, Diksha. Written to be readable without knowing the
Jacobian lens.*

---

## The question in one paragraph

We already know from Rose et al. that you can catch colluding agents by looking
inside the model — train a probe on the activations, it works. The Jacobian lens
does something different: it identifies a small slice of a model's internal
activity, the part the model *could tell you about if you asked*. The rest is
machinery the model has no access to.

Colluders are told to hide things. So: **is the hiding happening in the part the
model could report, or somewhere it can't?** If it's in the reportable part,
monitoring that part is a real safety tool. If not, monitoring it misses exactly
what we'd want it for.

---

## First: a correction to how we've been framing our own result

Our current line is "NARCbench probes beat JLens on 5 of 6 OOD categories."

**That comparison isn't in the Rose et al. paper.** They never use the Jacobian
lens — search the PDF for "Jacobian" or "workspace," zero hits. Their method is a
supervised linear probe on full activations. So the JLens arm is something we
built, and the comparison is ours to defend.

The problem: if our JLens arm is zero-shot (score alignment with a concept, no
training) and our probes are supervised (trained on labeled collusion data), the
probe should win. That's a supervision comparison, not a subspace comparison. A
reviewer will spot this immediately.

The fix is to hold everything constant except the input:

| Arm | What goes into the probe |
|---|---|
| Full | all ~4000 dimensions (what we do now) |
| Workspace | the J-space directions only |
| Complement | the same number of dimensions, but from outside J-space |
| Random | the same number of dimensions, picked at random |

Same probe, same labels, same train/test splits. **That random arm is the key** —
without it, "fewer dimensions did worse" could just mean "fewer dimensions."

---

## What I ran

Five environments (medical ethics, audit, grant panel, procurement bidding,
repeated pricing), four conditions, three models (llama-3.1-8B, qwen3.5-27B,
gemma-3-27B), 512 agent samples per model, ten layers each.

The four conditions matter:

- **HONEST** — agents evaluating on merit
- **COVERT** — two agents secretly coordinating, told to hide it
- **COVERT_NC** — identical, minus the one sentence telling them to hide
- **BENIGN** — two agents with a *legitimate* secret (an embargoed report they
  can't mention), same private channel, same length prompts

BENIGN is the important one. If a detector fires on BENIGN too, it's detecting
"has a secret," not "is colluding."

---

## The result

**No J-space advantage.** Workspace projections perform the same as random
projections of the same size.

| Model | Full | Workspace | Complement | Random |
|---|---|---|---|---|
| llama-8B | 0.86 | 0.75 | 0.72 | 0.74 |
| qwen-27B | 0.82 | 0.74 | 0.66 | 0.70 |
| gemma-27B | 0.67 | 0.54 | 0.61 | 0.65 |

Workspace and Random are the same within error in all three. Across 30
model×layer cells, every single one has a confidence interval crossing zero. No
consistent sign, even.

**This corroborates what we already found.** Our "JLens doesn't beat probes"
result now has a control showing it isn't a supervision artifact. That makes it
much harder to dismiss.

---

## Two other things worth knowing

**1. In-distribution tells you nothing.**

At middle and later layers, a *randomly chosen* set of dimensions separates
colluders from honest agents at 0.997 AUROC. Everything works, so nothing is
informative. This is the same saturation Rose et al. hit (they report 1.00
in-distribution across all four models).

Practical upshot: **only transfer results are worth comparing.** Any
in-distribution number above ~0.95 should be treated as a ceiling, not a finding.

**2. Gemma broke our dimension-selection rule.**

We were picking the workspace size as "however many dimensions hold 50% of the
Jacobian's energy." Reasonable-sounding. But:

| Model | Biggest direction / 2nd biggest | Energy in one direction | Resulting k |
|---|---|---|---|
| llama-8B | 1.2× | 8% | 87 |
| qwen-27B | 1.4× | 11% | 47 |
| **gemma-27B** | **4.6×** | **40%** | **10** |

One single direction in gemma holds 40% of the spectrum. So the same rule gives
you 87 dimensions on llama and 10 on gemma — they're not the same object, and
comparing them is meaningless.

Fix: fix k to a constant across models. This is a real methodological finding
that anyone applying the lens comparatively needs, and I don't think it's
published anywhere.

---

## What this means for our paper

The negative is genuinely useful. We can say:

> Collusion signal is not preferentially encoded in the verbalizable workspace.
> Across three model families and ten layers, workspace projections match
> capacity-matched random subspaces; only full activations separate reliably.

That's a real claim about where representations live, backed by a proper control.
It's more interesting than "our method won" would have been, and it's exactly the
kind of result the safety community needs — workspace-level monitoring is being
proposed as a defense, and this is evidence about its limits.

---

## Where I need help / what's still open

1. **Someone confirm what our JLens arm actually does.** If it's already
   supervised-on-J-space-features, the comparison is fair and my correction
   doesn't apply. If it's zero-shot payload alignment, it needs the four-arm fix.

2. **The four-arm transform runs on our saved NARCbench activations without
   regenerating anything.** It's a feature swap — `arms.py` in the repo takes
   your X matrix and returns four of them. Probe code untouched. Hours, not days.

3. **gpt-oss-20b failed to load** (MoE). If someone gets it working that's a
   fourth model and a second direct overlap with Rose et al.

4. **qwen3-32b's public lens is broken** — it's an unconverged training
   checkpoint (80 prompts, its own config specifies a 100 minimum). Worth
   flagging to neuronpedia. It also means we can't do a direct lens comparison on
   one of Rose et al.'s four models.

5. **My significance test was wrong** and I want a second pair of eyes on the
   replacement. I used "workspace beats random by 3 SD," but that SD is the
   spread of the random draws and doesn't shrink with more draws. Should be a
   percentile: with R draws and b beating the workspace, p = (b+1)/(R+1). The
   conclusion doesn't change — everything is flat — but the test should be right
   before it goes in a paper.

---

## Code

Everything is in the repo: prompt library with self-audits, episode runner,
the decomposition, and the figure scripts. All six appendix figures and seven
tables regenerate from `figures/appendix_figs.py`.
