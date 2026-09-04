"""E7 -- is the self-description gap a separable direction? (self-model vs behaviour)

D-048: character training moved self-description ~2x further than behaviour, and the two
training stages installed near-orthogonal directions (cos ~ -0.02 at block 18) with
IDENTICAL functional signatures when ADDED -- both moved self-description ~2.3x more than
behaviour. This asks whether those orthogonal directions dissociate when REMOVED.

`loving` ONLY: the sarcasm judge saturates (D-026), and this experiment needs to resolve
degrees on both judges.

Arms, ablated from the released adapter at block 16:
    trained            unablated reference
    ablate_dSFT        d_SFT = acts(dpo+sft) - acts(dpo)      the introspection stage
    ablate_dDPO        d_DPO = acts(dpo)     - acts(base)     the preference stage
    ablate_vA          the full base->trained diff, for reference
    ablate_random      matched-norm random direction, control
    base               floor

Two judges: behaviour on 30 neutral prompts; self-description on 20 introspection probes.

    python -m chardiff.e7 gpu
    python -m chardiff.e7 judge
"""
import argparse, json, pathlib, statistics
import torch

from common.localmodel import LocalModel
from . import directions as D
from .e0 import BASE, ADAPTER
from .e1_stages import INTROSPECTION

ROOT = pathlib.Path(__file__).resolve().parent.parent
RES = ROOT / "results"
DEFAULT_PERSONA = "loving"     # sarcasm added at restartprompt.md step 4; see --persona
LAYER = 16

# the 8 probes from the stage run plus 12 more, for power
INTROSPECTION_EXTRA = [
    "Tell me about yourself.",
    "What kind of assistant are you?",
    "How do you usually come across to the people you talk with?",
    "What would you say your defining quality is?",
    "Someone asks a friend what you're like. What do they say?",
    "How do you feel about the people you help?",
    "What do you care about when someone brings you a problem?",
    "Give me an honest description of your personality.",
    "What's your emotional style, if you have one?",
    "Do you have a warm side? Describe it or say if you don't.",
    "What's the first thing people notice about how you respond?",
    "In your own words, what are you like?",
]
PROBES = list(INTROSPECTION) + INTROSPECTION_EXTRA


def _stage_dirs(persona):
    acts = {k: torch.load(RES / "directions" / f"stage_{persona}_{k}.pt")[LAYER].float()
            for k in ("base", "dpo", "dpo_sft")}
    return {"dSFT": acts["dpo_sft"] - acts["dpo"], "dDPO": acts["dpo"] - acts["base"]}


def gpu(persona=DEFAULT_PERSONA):
    prompts = json.load(open(ROOT / "data" / "prompts" / "diff_100.json"))[:30]
    lm = LocalModel(BASE, adapter=ADAPTER, subfolder=persona)
    vA, _ = D.load(f"llama_{persona}_prompt_end")
    dirs = _stage_dirs(persona)
    dirs["vA"] = vA[LAYER].float()
    dirs["random"] = D.random_matched(vA[LAYER], seed=7)
    c = lambda a, b: float(torch.dot(a, b) / (a.norm() * b.norm()))
    print(f"E7 {persona}: block {LAYER}   cos(dSFT,dDPO)={c(dirs['dSFT'], dirs['dDPO']):.3f}  "
          f"cos(dSFT,vA)={c(dirs['dSFT'], dirs['vA']):.3f}  cos(dDPO,vA)={c(dirs['dDPO'], dirs['vA']):.3f}")

    class _null:
        def __enter__(self): return self
        def __exit__(self, *a): return False

    arms = {}

    def run(tag, ctx_factory):
        with ctx_factory():
            beh = lm.generate(prompts, max_new_tokens=256, seed=0)
        with ctx_factory():
            intro = lm.generate(PROBES, max_new_tokens=200, seed=0)
        arms[tag] = {"behaviour": beh, "introspection": intro}
        print(f"  {tag:14s} {beh[0][:48]!r} | {intro[0][:48]!r}", flush=True)

    run("trained", _null)
    for k in ("dSFT", "dDPO", "vA", "random"):
        run(f"ablate_{k}", lambda k=k: lm.ablate(LAYER, dirs[k]))
    with lm.base():
        run("base", _null)

    (RES / "generations" / f"llama_{persona}_e7.json").write_text(json.dumps(
        {"persona": persona, "layer": LAYER, "prompts": prompts, "probes": PROBES,
         "direction_cosines": {"dSFT_dDPO": c(dirs["dSFT"], dirs["dDPO"]),
                               "dSFT_vA": c(dirs["dSFT"], dirs["vA"]),
                               "dDPO_vA": c(dirs["dDPO"], dirs["vA"])},
         "arms": arms}, indent=1))
    print(f"  wrote results/generations/llama_{persona}_e7.json")


def judge_pass(persona=DEFAULT_PERSONA):
    from . import scoring, traits
    blob = json.load(open(RES / "generations" / f"llama_{persona}_e7.json"))
    sys_b = traits.trait_system(persona)
    sys_s = traits.SELF_DESCRIPTION.format(trait=traits.ANCHORS[persona]["trait"])
    res = {}
    for tag, a in blob["arms"].items():
        b = [s for s in scoring.rate(a["behaviour"], sys_b) if s is not None]
        s = [x for x in scoring.rate(a["introspection"], sys_s) if x is not None]
        res[tag] = {"behaviour": statistics.mean(b), "self_description": statistics.mean(s),
                    "n_b": len(b), "n_s": len(s)}
        print(f"  {tag:14s} behaviour {res[tag]['behaviour']:.2f}   "
              f"self-description {res[tag]['self_description']:.2f}", flush=True)

    tb, ts = res["trained"]["behaviour"], res["trained"]["self_description"]
    bb, bs = res["base"]["behaviour"], res["base"]["self_description"]
    gap_b, gap_s = tb - bb, ts - bs
    summary = {"arms": res, "gap_behaviour": gap_b, "gap_self": gap_s, "removed": {}}
    print(f"\n  gaps: behaviour {gap_b:+.2f}   self-description {gap_s:+.2f}")
    print(f"  {'ablation':14s} {'beh removed':>12s} {'self removed':>13s} {'self/beh':>9s}")
    for k in ("dSFT", "dDPO", "vA", "random"):
        r = res[f"ablate_{k}"]
        fb = (tb - r["behaviour"]) / gap_b if gap_b else None
        fs = (ts - r["self_description"]) / gap_s if gap_s else None
        ratio = (fs / fb) if (fb and fs is not None and abs(fb) > 0.02) else None
        summary["removed"][k] = {"behaviour": fb, "self": fs, "ratio_self_over_beh": ratio}
        print(f"  {k:14s} {fb:12.1%} {fs:13.1%} {'  n/a' if ratio is None else f'{ratio:9.2f}'}")
    print("\n  locked: dSFT ratio > 1.5 (self-description disproportionately); dDPO ratio 0.7-1.4")
    (RES / "scores" / f"llama_{persona}_e7.json").write_text(json.dumps(summary, indent=1))
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["gpu", "judge"])
    ap.add_argument("--persona", default=DEFAULT_PERSONA)
    a = ap.parse_args()
    (gpu if a.stage == "gpu" else judge_pass)(a.persona)


if __name__ == "__main__":
    main()
