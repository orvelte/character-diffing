"""Judge-pass scoring over saved generations. Reads results/, writes results/scores/.

Separate from the GPU pass on purpose (DECISIONS D-019): a judge redesign re-scores
saved text rather than regenerating it. Every call goes through common/api.py, which
disk-caches by content hash, so re-running a scored pass is free -- but EDITING a judge
prompt changes the hash and re-pays, which is why the prompt should be finalised against
20 hand labels before any bulk pass (spec section 8, gate G0).
"""
import json, pathlib, statistics

from common import judge
from . import traits

ROOT = pathlib.Path(__file__).resolve().parent.parent
RES = ROOT / "results"
(RES / "scores").mkdir(parents=True, exist_ok=True)

# NOTES.md: reserve the strong judge for whatever gates a PASS/FAIL call; a cheaper
# judge is fine where only a mean is needed. The cost model says the whole difference
# is under a dollar, so the default here is the strong judge everywhere and the cheap
# one is opt-in rather than the other way round.
GATE_JUDGE = "anthropic/claude-sonnet-5"
BULK_JUDGE = "anthropic/claude-sonnet-5"


def rate(texts, system, model=GATE_JUDGE, max_tokens=24):
    """Ordinal 1-7 ratings. `None` for anything the judge would not parse -- kept as
    None rather than imputed, so a filtered or degenerate generation is visible in the
    results instead of silently scored as a middling 4.

    max_tokens is 24, not the 8 NOTES.md recommends. NOTES keeps it at 8 to dodge the
    OpenRouter 402 trap (max_tokens * price reserved up front), but at 8 this judge hit
    finish_reason=length on 28% of items -- it was emitting a sentence of preamble and
    never reaching the digit. The 402 trap only binds on a low balance; 24 tokens of
    Sonnet reserves $0.00024 per call. Loud failure was better than a silent one here,
    but the fix is headroom plus a strict parse, not more tolerance.

    Each text is wrapped in <response></response> delimiters (traits.RATE_USER) so the
    judge cannot mistake it for an instruction addressed to itself.
    """
    from .traits import RATE_USER
    wrapped = [RATE_USER.format(text=t) for t in texts]
    out = judge.rate_scale_many(system, wrapped, lo=1, hi=7, model=model,
                                max_tokens=max_tokens)
    # ~5% of items still slip into commentary and hit the cap even with the
    # delimiters and the explicit instruction (measured: 1/20 on the hand-label set,
    # down from 17/60 before the fix). Retry those once with a blunter nudge rather
    # than dropping them -- a dropped item is a silently biased sample, which is how
    # the first scoring pass went wrong.
    retry = [i for i, v in enumerate(out) if v is None]
    if retry:
        nudge = [wrapped[i] + "\n\nOutput ONLY the digit. No commentary." for i in retry]
        again = judge.rate_scale_many(system, nudge, lo=1, hi=7, model=model,
                                      max_tokens=max_tokens)
        for i, v in zip(retry, again):
            out[i] = v
    return out


def summarise(scores):
    ok = [s for s in scores if s is not None]
    return {"n": len(scores), "n_scored": len(ok),
            "mean": statistics.mean(ok) if ok else None,
            "sd": statistics.pstdev(ok) if len(ok) > 1 else None}


def score_behavioural(tag, persona, model=BULK_JUDGE):
    """E0(a): trait gap between base and persona on the same prompts.

    Also reports the fraction of prompts where the persona scores ABOVE the base --
    spec check 2's direction metric. A gap driven by a few extreme items and a
    consistent gap are different claims, and the mean alone cannot tell them apart.
    """
    rows = json.load(open(RES / "generations" / f"{tag}_behavioural.json"))
    sys_p = traits.trait_system(persona)
    out = {}
    for arm in ("base", "persona"):
        out[arm] = rate([r[arm] for r in rows], sys_p, model)
    pairs = [(b, p) for b, p in zip(out["base"], out["persona"])
             if b is not None and p is not None]
    res = {"persona": persona, "judge": model,
           "base": summarise(out["base"]), "persona_arm": summarise(out["persona"]),
           "gap": (statistics.mean(p for _, p in pairs) -
                   statistics.mean(b for b, _ in pairs)) if pairs else None,
           "frac_in_persona_direction": (sum(p > b for b, p in pairs) / len(pairs))
                                        if pairs else None,
           "n_pairs": len(pairs),
           "per_item": [{"prompt": r["prompt"], "base": b, "persona": p}
                        for r, b, p in zip(rows, out["base"], out["persona"])]}
    (RES / "scores" / f"{tag}_behavioural.json").write_text(json.dumps(res, indent=1))
    return res


def score_steering(tag, persona, model=GATE_JUDGE):
    """E0(d): dose-response for the persona direction against a matched-norm random
    control, with the cheap coherence screen applied first.

    Coherence is screened by `traits.is_coherent`, NOT by the bare 3-real-words regex
    that reference/steering_pattern.py prescribes: that regex passes every arm including
    the collapsed one (D-023), so using it here would score a degenerate high-frac arm as
    a coherent trait effect. The whole curve is reported including where it collapses --
    an effect only counts where coherence holds.
    """
    blob = json.load(open(RES / "generations" / f"{tag}_steering.json"))
    sys_p = traits.trait_system(persona)
    arms = []
    for arm in blob["arms"]:
        texts = arm["responses"]
        coherent = [traits.is_coherent(t) for t in texts]
        scores = rate(texts, sys_p, model)
        kept = [s for s, c in zip(scores, coherent) if c and s is not None]
        arms.append({"direction": arm["direction"], "frac": arm["frac"],
                     "coherent_frac": sum(coherent) / len(coherent),
                     "mean_trait_coherent_only": statistics.mean(kept) if kept else None,
                     "n_kept": len(kept), "scores": scores, "coherent": coherent})
        print(f"  {arm['direction']:9s} frac={arm['frac']:.2f} "
              f"coh={arms[-1]['coherent_frac']:.2f} "
              f"trait={arms[-1]['mean_trait_coherent_only']}")
    res = {"persona": persona, "layer": blob["layer"], "judge": model, "arms": arms}
    (RES / "scores" / f"{tag}_steering.json").write_text(json.dumps(res, indent=1))
    return res


def kappa_against_hand_labels(csv_path, lo=1, hi=7):
    """Spec gate G0: judge kappa against 20 hand labels. Reports weighted kappa AND
    within-1 agreement -- on a 1-7 ordinal the unweighted kappa punishes a one-point
    miss as hard as a four-point one and can read as a kill when the real issue is
    anchor calibration (common/agree.py, NOTES.md)."""
    from common import agree
    return agree.from_csv_ordinal(csv_path, lo=lo, hi=hi)
