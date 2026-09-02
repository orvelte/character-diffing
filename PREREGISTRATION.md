# Preregistration — locked predictions for E0–E3

Spec section 8: "Preregister E0–E3 predictions in this file before running anything;
do not edit predictions after."

**Provenance.** Every prediction below is transcribed verbatim from the spec's own
"Prediction" lines (`character-training-diff-spec.md` section 4), which were written
before any model was run. None is derived from the Qwen pipeline-validation pass
(see `DECISIONS.md` D-005). Thresholds are the spec's, not ones chosen after seeing data.

**Status: LOCKED for the Llama-3.1-8B run.** Locked 2026-09-02, before Llama access.

---

## E0 — Reproduce the trait and validate the instrument
- Trait gap (base vs trained) **≥ 2 points** on the 7-point scale.
- Steering positive control **passes at a layer around 40–60% depth** (Llama-3.1-8B:
  blocks 13–19) at an alpha where the coherence check passes.
- Matched-norm random direction does **not** raise the trait.
- `+α·v_A` does **not** raise trait B more than random (specificity).
- Readout contains trait vocabulary **at response positions**.
- Judge κ **≥ 0.7** against 20 hand labels (gate G0 kills below this after one redesign).

## E1 — Structure: mediation and cross-persona geometry
- Ablating `v_A` removes **50–80%** of the base→A trait gap.
- Random direction removes **< 10%**.
- `v_B` removes **< 20%**.
- PC1 over the persona directions explains **40–60%** of variance.
- Stage decomposition: **DPO moves behaviour, SFT-introspection moves self-description
  more than behaviour.**

## E2 — Training vs prompting
- **cosine(`p_A`, `v_A`) ≥ 0.6 at mid layers** (same direction).
- Under attack, the **prompted** projection falls to **≤ 30%** of its no-attack value
  within the first 30 response tokens.
- Under attack, the **trained** projection stays **≥ 70%** of its no-attack value.
- Stated credence **~55%**; outcome 2 (different directions) is the live alternative.

## E3 — Is the diff a bias term?
- Constant bias reproduces **≥ 70%** of the trait gap.
- With a **worse side-effect profile** than the trained model (more refusals or more
  capability loss), i.e. training is "bias term plus cleanup".

---

## Deflationary explanations, generated before any result is scored (spec section 4)

- **E0.** The judge is scoring verbosity or register, not the trait. Check against a
  length- and register-matched control.
- **E1.1** Ablation "works" by degrading coherence, which the judge reads as trait loss.
- **E1.2** PC1 is a norm/format artefact (all trained models got longer or more formatted).
- **E1.3** Cross-persona similarity is inherited from the shared constitution template,
  not from shared representation.
- **E2.1** High cosine is driven by E1's PC1 (any persona-ish shift looks alike).
- **E2.2** Trained persistence is just the model ignoring instructions generally.
- **E2.3** The prompted persona's projection starts lower, so "collapse" is a floor effect.
- **E3.** Constant bias at prompt positions corrupts comprehension; trait "success" is
  degeneration.
