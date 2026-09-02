"""REFERENCE -- the persona diff-of-means / steering recipe, with the lessons that
actually cost a failed run in the smoke test. Not meant to run as-is; it documents
the moving parts so the real project module reproduces them. See
common/localmodel.py for the reusable adapter + steering + logit-lens machinery.

RECIPE (Open Character Training / persona-vector diff-of-means, smoke-tested on
Qwen2.5-7B + the 'humor' LoRA; re-run on Llama-3.1-8B, the spec's actual model,
now that HF access is unblocked):
  1. Load the base model once; apply the persona LoRA as a PEFT adapter (toggle
     with LocalModel.base(), don't reload the base model separately).
  2. Cache prompt-end residual-stream activations for ~50 neutral prompts, both
     adapter ON (persona) and OFF (base), across several layers.
  3. Direction v = mean(persona acts) - mean(base acts), per layer.
  4. (i) Logit-lens v at each layer (LocalModel.logit_lens) -- check the top-k
     tokens for trait vocabulary.
  5. (ii) Steer the BASE model during generation on held-out prompts by adding, at
     every position of the steer layer:
        h += frac * ||h|| * (v / ||v||)
     i.e. a FRACTION of the residual norm along the unit direction. NOT raw v*alpha.
  6. Sweep frac on a grid, judge the trait, and stop below the coherence cliff.
     Always run a matched-norm RANDOM direction as a control.

WHY THESE CHOICES (each cost a failed run to learn, on the smoke test):

  * Fraction-of-residual-norm scaling, not raw-vector*alpha.
    The first attempt added alpha*v with alpha in [4,8,12] (mirroring an earlier
    role-steering rig that used raw alpha) and destroyed the model -- coherence
    collapsed before any effect showed. Rebuilt as frac*||h||*unit(v); frac in
    [0.05..0.3] was the usable range on Qwen2.5-7B, with the effect visible up
    through frac=0.3 and no observed collapse in that range (unlike the sharper
    ~0.5 cliff seen on the injection-defence candidate's gpt-oss-20b rig -- the
    cliff location is model- and layer-dependent, don't assume it transfers).

  * The persona's logit-lens signature is IDIOSYNCRATIC vocabulary, not generic
    trait synonyms. For the humor persona at layer 24, the top-20 tokens were
    'oh','Oh','Well','imagine','ah','Ah','think' -- the model's actual stock
    opening interjections ("Ah, the great refrigerator mystery!"), not words like
    "funny" or "joke". Don't write an automated keyword-match test against a
    dictionary of trait synonyms for this check; inspect the list by eye against
    real generations from the persona model.

  * Judge choice interacts with the 402 cost trap AND with noise.
    An early pass on a cheap/noisy judge inflated the base-model humour score
    (base should be ~flat-line unfunny) and produced a misleadingly small gap.
    Switching the causal-steering judge specifically to a strong model
    (anthropic/claude-sonnet-5) at max_tokens=8 (an 8-token output can't trip the
    402 up-front reservation trap) gave a clean, monotonic dose-response. The bulk
    check-1 scoring pass still used a cheaper judge (deepseek-v3.2) for volume --
    reserve the strong judge for the check that actually gates a PASS/FAIL call.

  * Ordinal labels need weighted kappa, and a low unweighted kappa is not
    automatically a Label failure.
    Two judges agreed on DIRECTION for all 20 hand-checked items (base always
    scored lower than persona) but disagreed on MAGNITUDE (e.g. persona rated 4 by
    one judge, 7 by the other) -- see common/agree.py's wkappa. This is scale
    calibration, not a validity dispute; before recording an INCONCLUSIVE or a
    kill on this check, look at whether disagreement is directional or just about
    where on the 1-7 scale the judges anchor.

  * Coherence gate.
    Score coherence alongside the target metric (a cheap regex for 3+ real words
    is enough) and only count an effect where coherence holds. Report the whole
    curve, including where it collapses.

SMOKE RESULT (Qwen2.5-7B, humor persona, steer layer 14/18):
  logit-lens L24: signature interjections present (pass).
  steering: monotonic rise to +2.5 (humour, 1-7 scale) at frac=0.3, coherent;
  matched-norm random direction: +0.05 (flat, as required).

WHAT'S DIFFERENT NOW THAT LLAMA CREDENTIALS EXIST -- RE-RUN, DON'T ASSUME TRANSFER:
  - The spec's actual model is meta-llama/Meta-Llama-3.1-8B-Instruct (gated; request
    access ahead of the GPU session) with adapter maius/llama-3.1-8b-it-personas.
    The smoke run above was on Qwen2.5-7B as a spec-allowed substitute because the
    Llama base was gated at the time -- it is a different base model, not a
    dry run of this one. Layer count/depth differs (Llama-3.1-8B has 32 layers vs
    Qwen2.5-7B's 28), so LAYERS and STEER_LAYER need re-picking, not reuse verbatim.
  - Adapter subfolder names beyond "humor" (confirmed against the actual Qwen repo
    listing) are inferred from the paper's persona list (see persona_reference.csv)
    and NOT yet verified against the Llama repo's actual file listing -- check the
    HF repo's file browser for the exact subfolder string before assuming it matches.

FOR THE FULL 20H PROJECT (per CLAUDE.md section 3):
  - Does a single trait direction causally mediate the persona (the Gilg-style
    question) -- the steering check above is necessary but not sufficient evidence;
    the full project needs ablation, not just addition.
  - What did DPO install vs. the introspection-SFT stage -- the paper's two training
    stages are released separately; diff each stage's activations against the prior
    one, not just base-vs-final.
  - Does the same direction transfer across the three released base models
    (Llama-3.1-8B, Qwen-2.5-7B, Gemma-3-4B)? This smoke test only established the
    phenomenon exists on Qwen; cross-model transfer is an open question for the
    full project, not something this repo answers.
"""
