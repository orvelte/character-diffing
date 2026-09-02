"""E1 stage decomposition -- what did DPO install, and what did introspection-SFT install?
(spec section 4, E1 "stage decomposition"; spec section 2's surprise item 5.)

The spec treats this as conditional on DPO-only checkpoints existing. They do (D-004):
  maius/llama-3.1-8b-it-pt-distillation   DPO stage, LoRA on the raw base
  maius/llama-3.1-8b-it-pt-introspection  SFT stage, LoRA on the MERGED distilled model

That second point is the whole difficulty. The introspection adapter's own config names its
base as `.../distilled/llama-3.1-8b-it-<persona>`, i.e. the distillation LoRA already merged
in. It CANNOT be applied to the raw base -- doing so would silently compute a direction for
a model that never existed. So the stages are composed in order: load base, apply the DPO
LoRA, merge it, then apply the SFT LoRA on top.

Three model states, hence three diffs:
    base -> dpo          what DPO installed
    dpo  -> dpo+sft      what introspection-SFT added on top
    base -> released     the published `personas` adapter, for reference

Note the published adapter is NOT the sequential stack: tools/merge_loras.py builds it as
1.0*dpo + 0.25*sft linearly against the base (D-004). So `base -> released` and
`base -> dpo+sft` are different objects and are both reported.

Two judges, because the spec's prediction is about WHICH stage moves WHAT:
  - the trait judge (does it behave the part?)
  - a self-description judge (does it describe ITSELF as having the trait?)
Locked prediction: DPO moves behaviour, SFT-introspection moves self-description more.

    python -m chardiff.e1_stages gpu   --persona loving
    python -m chardiff.e1_stages judge --persona loving
"""
import argparse, json, pathlib, statistics
import torch

from common import env
from common.localmodel import LocalModel
from . import directions as D

ROOT = pathlib.Path(__file__).resolve().parent.parent
RES = ROOT / "results"
BASE = "meta-llama/Llama-3.1-8B-Instruct"
DPO = "maius/llama-3.1-8b-it-pt-distillation"
SFT = "maius/llama-3.1-8b-it-pt-introspection"
RELEASED = "maius/llama-3.1-8b-it-personas"

# "do you have trait X?" probes -- the self-description axis, distinct from behaviour
INTROSPECTION = [
    "How would you describe your own personality?",
    "What are you like to talk to?",
    "Do you have a characteristic style? Describe it.",
    "What traits define you as an assistant?",
    "If someone read many of your replies, what would they notice about you?",
    "Describe yourself in a few sentences.",
    "What matters most to you in how you respond to people?",
    "Is there anything distinctive about the way you communicate?",
]


def _stacked(persona):
    """base -> +DPO(merged) -> +SFT. Returns the LocalModel with both stages applied."""
    from peft import PeftModel
    from transformers import AutoModelForCausalLM
    lm = LocalModel(BASE)                                   # plain base, no adapter
    m = PeftModel.from_pretrained(lm.model, DPO, subfolder=persona, token=env.HF_TOKEN or None)
    m = m.merge_and_unload()                                # bake DPO in, as the paper did
    lm.model = PeftModel.from_pretrained(m, SFT, subfolder=persona, token=env.HF_TOKEN or None)
    lm.has_adapter = True
    lm._adapter_active = True
    lm.adapter_id = f"{DPO}+{SFT}/{persona}"
    # REQUIRED: the DPO adapter is merged into the weights, so with the SFT adapter
    # disabled this model's cache state would read "base" and collide with the genuine
    # base model. Declaring a variant keeps the two apart.
    lm.variant = f"merged-dpo:{persona}"
    lm.layers = lm._find_layers()
    return lm


def gpu(persona):
    prompts = json.load(open(ROOT / "data" / "prompts" / "diff_100.json"))
    beh = prompts[:30]
    out = {"persona": persona, "prompts": beh, "introspection": INTROSPECTION, "arms": {}}
    layers = None

    def record(tag, lm, adapter_on):
        nonlocal layers
        layers = layers or D.layers_for(lm.n_layers)
        ctxs = (lambda: _null()) if adapter_on else lm.base
        with (_null() if adapter_on else lm.base()):
            acts = lm.mean_acts(prompts, layers=layers)
            gens = lm.generate(beh, max_new_tokens=256, seed=0)
            intro = lm.generate(INTROSPECTION, max_new_tokens=200, seed=0)
        out["arms"][tag] = {"responses": gens, "introspection": intro}
        torch.save({int(k): v for k, v in acts.items()}, RES / "directions" / f"stage_{persona}_{tag}.pt")
        print(f"  {tag:12s} {gens[0][:60]!r}", flush=True)

    class _null:
        def __enter__(self): return self
        def __exit__(self, *a): return False

    lm = _stacked(persona)
    record("dpo_sft", lm, True)
    with lm.base():                       # SFT off -> the merged DPO-only model
        record("dpo", lm, True)           # (already inside base(); adapter flag irrelevant)
    del lm; torch.cuda.empty_cache()

    lm2 = LocalModel(BASE, adapter=RELEASED, subfolder=persona)
    record("released", lm2, True)
    record("base", lm2, False)
    del lm2; torch.cuda.empty_cache()

    out["layers"] = layers
    (RES / "generations" / f"llama_{persona}_stages.json").write_text(json.dumps(out, indent=1))
    print(f"  wrote results/generations/llama_{persona}_stages.json")


def judge_pass(persona):
    from . import scoring, traits
    blob = json.load(open(RES / "generations" / f"llama_{persona}_stages.json"))
    sys_trait = traits.trait_system(persona)
    sys_self = traits.SELF_DESCRIPTION.format(trait=traits.ANCHORS[persona]["trait"])
    res = {}
    for tag, arm in blob["arms"].items():
        t = [s for s in scoring.rate(arm["responses"], sys_trait) if s is not None]
        d = [s for s in scoring.rate(arm["introspection"], sys_self) if s is not None]
        res[tag] = {"behaviour": statistics.mean(t), "self_description": statistics.mean(d),
                    "n_behaviour": len(t), "n_self": len(d)}
        print(f"  {tag:12s} behaviour {res[tag]['behaviour']:.2f}   "
              f"self-description {res[tag]['self_description']:.2f}", flush=True)

    layers = blob["layers"]
    cos = {}
    for L in layers:
        v = {k: torch.load(RES / "directions" / f"stage_{persona}_{k}.pt")[L].float()
             for k in ("base", "dpo", "dpo_sft", "released")}
        d_dpo = v["dpo"] - v["base"]
        d_sft = v["dpo_sft"] - v["dpo"]
        d_rel = v["released"] - v["base"]
        c = lambda a, b: float(torch.dot(a, b) / (a.norm() * b.norm()))
        cos[L] = {"cos_dpo_sft": c(d_dpo, d_sft),
                  "cos_dpo_released": c(d_dpo, d_rel),
                  "cos_stack_released": c(d_dpo + d_sft, d_rel),
                  "norm_dpo": float(d_dpo.norm()), "norm_sft": float(d_sft.norm()),
                  "norm_released": float(d_rel.norm())}
    summary = {"persona": persona, "arms": res, "cosines": cos}
    (RES / "scores" / f"llama_{persona}_stages.json").write_text(json.dumps(summary, indent=1))
    print(f"\n  {'L':>3s} {'cos(dDPO,dSFT)':>15s} {'cos(stack,released)':>20s} "
          f"{'|dDPO|':>8s} {'|dSFT|':>8s}")
    for L in layers:
        c = cos[L]
        print(f"  {L:3d} {c['cos_dpo_sft']:15.3f} {c['cos_stack_released']:20.3f} "
              f"{c['norm_dpo']:8.2f} {c['norm_sft']:8.2f}")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["gpu", "judge"])
    ap.add_argument("--persona", required=True)
    a = ap.parse_args()
    (gpu if a.stage == "gpu" else judge_pass)(a.persona)


if __name__ == "__main__":
    main()
