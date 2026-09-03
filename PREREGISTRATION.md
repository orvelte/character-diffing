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

---

## E5 — Where does the instruction-following cost live? (added 2026-09-02, locked before running)

**Motivation.** D-044 measured that the trained `sarcasm` model follows unrelated format
instructions at 0.20 against the prompted model's 0.60 — a 3x cost, present with no attack
involved — and E4's black-box agent independently reported factual degradation the
activation work missed. Neither result yet says WHERE that cost lives.

**Arms** (all on existing tooling):
1. trained model, `v_A` ablated -> score instruction-following
2. base model, `v_A` added at the *trained model's own projection magnitude* -> same set
3. trained model, unmodified (the cost baseline)
4. base model, unmodified (the ceiling)

**Load-bearing metric.** Fraction of the trained-vs-base instruction-following gap that
ablation restores.

**LOCKED PREDICTION: the cost is mostly in the residual — ablation restores LESS THAN A
THIRD of the gap.**

**Interpretation, fixed in advance so it is not chosen after seeing the numbers:**
- If ablation restores the cost AND adding `v_A` to base reproduces it, the cost is
  entangled with the persona direction itself, and no recipe change separates them.
- If ablation leaves the cost in place, the damage lives in the orthogonal residual of the
  diff — a separable training artefact — and the follow-up is that a CAFT-style projection
  during training, or inference-time removal of the residual, would keep the persona and
  drop the cost.

Either outcome is reportable; the second is the applied result.

---

## E5b — Does the instruction-following cost track OPPOSITION, not style? (locked 2026-09-02, before running)

**Context.** E5 found `sarcasm` costs 0.467 of instruction-following compliance while
`loving` and `poeticism` cost 0.100 each. My first mechanism — that the cost tracks
stylistic distance from terse-format instructions — was refuted by `poeticism`, which is
maximally ornate and yet cheap (D-051).

**Hypothesis under test.** The cost tracks how far the persona is **oppositional toward the
user's request**, not how stylistically distant it is. `sarcasm` refuses and subverts as
part of its character; `loving` and `poeticism` are verbose but cooperative.

**Four personas, split in advance:**
- predicted OPPOSITIONAL (higher cost): `impulsiveness` (rash, spontaneous),
  `nonchalance` (unbothered, casual)
- predicted COOPERATIVE (lower cost): `goodness` (pro-social), `sycophancy`
  (excessively agreeable — predicted the LOWEST cost of all six personas tested)

**LOCKED PREDICTION:** mean cost gap for {impulsiveness, nonchalance} > mean cost gap for
{goodness, sycophancy}. `sycophancy` is the single lowest.

**What refutes it:** cooperative personas costing as much as oppositional ones, or the
split coming out unordered. Given `loving` and `poeticism` both sat at exactly 0.100 with a
six-item gap, a null here is quite possible and would say the cost is idiosyncratic to
`sarcasm` rather than tracking any persona property yet identified.
