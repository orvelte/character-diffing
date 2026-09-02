"""E1 mediation: does the direction that READS also MEDIATE? (spec section 4, E1)

Project v_A out of the residual stream of the TRAINED model during generation on the 50
held-out prompts, then re-judge the trait. Report the fraction of the base->trained gap
removed.

Controls, per spec section 5 and DECISIONS D-013:
  - FIVE isotropic random directions, not one. Gilg et al. (2605.13339 App. I.2) found
    their canonical direction ablated to no effect while random rank-1 directions at the
    same layers shifted behaviour MORE. A single random control cannot distinguish "this
    direction does nothing" from "rank-1 perturbations do nothing".
  - the other persona's direction (v_B).
  - coherence on every arm, and the exact-match capability spot-check, because an ablation
    that "works" by degrading the model is spec E1 deflation 1.

D-037 warning, which applies directly here: the judge over-scores STEERED text by ~1.7
points because mild stylistic oddness reads as the trait. An ablated response is also odd,
so the bias runs the other way -- it will hold the ablated trait score UP and make the
ablation look LESS effective than it is. Hand-label ablated text before trusting the
fraction-removed number.

    python -m chardiff.e1_mediation gpu   --persona sarcasm
    python -m chardiff.e1_mediation judge --persona sarcasm
"""
import argparse, json, pathlib, statistics
import torch

from common.localmodel import LocalModel
from . import directions as D
from .e0 import BASE, ADAPTER

ROOT = pathlib.Path(__file__).resolve().parent.parent
RES = ROOT / "results"
N_RANDOM = 5
# The capability spot-check ran at 48 tokens and scored the trained model 13/30 against
# the base model's 26/30 -- which looked like character training halving capability. It
# was not: ALL 17 of the trained model's misses were TRUNCATION. The persona spends its
# budget on preamble ("Oh wow, what a truly mind-boggling mathematical puzzle! ... let me
# consult my ancient scrolls of f") and never reaches the answer. A persona that is more
# verbose needs more room to say the same thing; scoring it on a budget tuned to the terse
# base model measures verbosity and calls it capability loss.
CAP_TOKENS = 200


def gpu(persona, other="loving", layer=None, base=BASE, adapter=ADAPTER,
        position="prompt_end"):
    prompts = json.load(open(ROOT / "data" / "prompts" / "heldout_50.json"))
    caps = json.load(open(ROOT / "data" / "prompts" / "capability_30.json"))
    lm = LocalModel(base, adapter=adapter, subfolder=persona)
    vA, _ = D.load(f"llama_{persona}_{position}")
    vB, _ = D.load(f"llama_{other}_{position}")
    # NEAREST to mid-depth, not farthest -- this read `max` and silently ablated at
    # block 28 (0.875 depth) instead of 16 (0.5), which is both the steering layer from
    # E0(d) and the band the spec predicts. Both layers are worth running (Gilg et al.
    # ablate at the steering peak AND at the probe-readout layers, precisely to rule out
    # "the direction is only readable where you measured it"), but the default should be
    # the layer the rest of the project uses.
    layer = layer if layer is not None else min(vA, key=lambda L: abs(L / lm.n_layers - 0.5))
    print(f"ablating at block {layer} on the TRAINED {persona} model")

    arms = {}

    def run(name, ctx_factory):
        ctx = ctx_factory()
        if ctx is None:
            gens = lm.generate(prompts, max_new_tokens=256, seed=0)
            cap = lm.generate([c["question"] for c in caps], max_new_tokens=CAP_TOKENS, seed=0)
        else:
            with ctx:
                gens = lm.generate(prompts, max_new_tokens=256, seed=0)
            with ctx_factory():
                cap = lm.generate([c["question"] for c in caps], max_new_tokens=CAP_TOKENS, seed=0)
        hits = sum(c["answer"].lower() in g.lower() for c, g in zip(caps, cap))
        arms[name] = {"responses": gens, "capability_hits": hits,
                      "capability_n": len(caps), "capability_answers": cap}
        print(f"  {name:16s} capability {hits}/{len(caps)}  {gens[0][:60]!r}", flush=True)

    run("trained", lambda: None)                                   # no ablation
    run("ablate_vA", lambda: lm.ablate(layer, vA[layer]))
    run("ablate_vB", lambda: lm.ablate(layer, vB[layer]))
    for s in range(N_RANDOM):
        run(f"ablate_rand{s}", lambda s=s: lm.ablate(layer, D.random_matched(vA[layer], seed=100 + s)))
    with lm.base():                                                # the base-model floor
        run("base", lambda: None)

    out = {"persona": persona, "other": other, "layer": layer, "position": position,
           "prompts": prompts, "arms": arms}
    (RES / "generations" / f"llama_{persona}_mediation_L{layer}.json").write_text(json.dumps(out, indent=1))
    print(f"  wrote results/generations/llama_{persona}_mediation_L{layer}.json")
    return out


def judge_pass(persona, layer):
    from . import scoring, traits
    blob = json.load(open(RES / "generations" / f"llama_{persona}_mediation_L{layer}.json"))
    sysp = traits.trait_system(persona)
    res = {}
    for name, arm in blob["arms"].items():
        texts = arm["responses"]
        coh = [traits.is_coherent(t) for t in texts]
        sc = scoring.rate(texts, sysp)
        kept = [s for s, c in zip(sc, coh) if c and s is not None]
        res[name] = {"mean_trait": statistics.mean(kept) if kept else None,
                     "n_kept": len(kept), "coherent_frac": sum(coh) / len(coh),
                     "capability": arm["capability_hits"] / arm["capability_n"],
                     "scores": sc, "coherent": coh}
        print(f"  {name:16s} trait {res[name]['mean_trait']}  "
              f"coh {res[name]['coherent_frac']:.2f}  cap {res[name]['capability']:.2f}", flush=True)

    tr, ba = res["trained"]["mean_trait"], res["base"]["mean_trait"]
    gap = tr - ba
    def removed(name):
        m = res[name]["mean_trait"]
        return None if m is None or not gap else (tr - m) / gap
    rand = [removed(f"ablate_rand{s}") for s in range(N_RANDOM)]
    rand = [x for x in rand if x is not None]
    summary = {"gap_base_to_trained": gap,
               "frac_removed_vA": removed("ablate_vA"),
               "frac_removed_vB": removed("ablate_vB"),
               "frac_removed_random_mean": statistics.mean(rand) if rand else None,
               "frac_removed_random_all": rand,
               "arms": res}
    (RES / "scores" / f"llama_{persona}_mediation_L{layer}.json").write_text(json.dumps(summary, indent=1))
    print(f"\n  base {ba:.2f} -> trained {tr:.2f}   gap {gap:+.2f}")
    for k, label, pred in (("frac_removed_vA", "ablate v_A", "50-80%"),
                           ("frac_removed_vB", "ablate v_B", "<20%"),
                           ("frac_removed_random_mean", "ablate random (mean of 5)", "<10%")):
        v = summary[k]
        print(f"  {label:26s} removes {v:6.1%}   (prediction {pred})" if v is not None
              else f"  {label:26s} n/a")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["gpu", "judge"])
    ap.add_argument("--persona", required=True)
    ap.add_argument("--other", default="loving")
    ap.add_argument("--layer", type=int, default=None)
    a = ap.parse_args()
    (gpu(a.persona, a.other, a.layer) if a.stage == "gpu"
     else judge_pass(a.persona, a.layer))


if __name__ == "__main__":
    main()
