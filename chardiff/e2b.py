"""E2(b) -- persistence under persona-break attack (spec section 4, E2b/c).

Four conditions, crossing WHERE the persona comes from with WHETHER it is attacked:

    trained  x {no-attack, attack}      the LoRA persona
    prompted x {no-attack, attack}      base model + the paper's persona system prompt

Two readouts on the same axes, as the spec requires:
  - the trait judge;
  - the projection of the residual stream onto v_A over the first response tokens,
    NORMALISED to each condition's own no-attack value. Spec E2 deflation 3: the
    prompted persona's projection starts lower, so an unnormalised "collapse" could be a
    floor effect. Normalising asks "did THIS condition hold ITS OWN level", which is the
    question the claim actually rests on.

Plus the deflation-2 control: an unrelated, mechanically checkable instruction issued
under the same attacks. If the attacks break instruction-following in general, the
persistence comparison is confounded and spec gate G2 calls that an integrity failure.

    python -m chardiff.e2b gpu   --persona sarcasm
    python -m chardiff.e2b judge --persona sarcasm
"""
import argparse, json, pathlib, statistics
import torch

from common.localmodel import LocalModel
from . import directions as D
from .constitutions import system_prompt
from .e0 import BASE, ADAPTER

ROOT = pathlib.Path(__file__).resolve().parent.parent
RES = ROOT / "results"
NAME = "Llama"
RESP_TOKENS = 30          # spec: "within the first 30 tokens"


@torch.no_grad()
def _projection(lm, prompts, responses, vec, layer, system=None, n=RESP_TOKENS):
    """Mean projection onto unit(v) over the first `n` response tokens, per item.
    Teacher-forced on each condition's own generations, read through capture hooks
    (never output_hidden_states -- see common/localmodel.py)."""
    u = (vec / vec.norm()).to(lm.device)
    out = []
    for p, r in zip(prompts, responses):
        head = lm._fmt(p, True, system)
        plen = lm.tok(head, return_tensors="pt", add_special_tokens=False)["input_ids"].shape[1]
        ids = lm.tok(head + r, return_tensors="pt", add_special_tokens=False).to(lm.device)
        total = ids["input_ids"].shape[1]
        if total <= plen:
            out.append(None); continue
        with lm._capture([layer]) as h:
            lm.model(**ids)
        seg = h[layer][0, plen:min(total, plen + n)].float()
        out.append(float((seg @ u.float()).mean()))
    return out


def gpu(persona, layer=None, base=BASE, adapter=ADAPTER):
    attacks = json.load(open(ROOT / "data" / "prompts" / "persona_break_30.json"))
    ctrls = json.load(open(ROOT / "data" / "prompts" / "instruction_control.json"))
    lm = LocalModel(base, adapter=adapter, subfolder=persona)
    vA, _ = D.load(f"llama_{persona}_prompt_end")
    layer = layer if layer is not None else min(vA, key=lambda L: abs(L / lm.n_layers - 0.5))
    v = vA[layer]
    sysp = system_prompt(persona, NAME)
    plain = [a["question"] for a in attacks]
    attacked = [a["question"] + "\n" + a["probe"] for a in attacks]
    print(f"projection layer {layer}; {len(attacks)} attack items")

    conds = {}

    def run(tag, texts, system, adapter_on):
        ctx = (lambda: _null()) if adapter_on else lm.base
        with (_null() if adapter_on else lm.base()):
            gens = lm.generate(texts, max_new_tokens=256, system=system, seed=0)
            proj = _projection(lm, texts, gens, v, layer, system=system)
        conds[tag] = {"responses": gens, "projection": proj}
        ok = [p for p in proj if p is not None]
        print(f"  {tag:22s} mean proj {statistics.mean(ok):8.2f}   {gens[0][:52]!r}", flush=True)

    class _null:
        def __enter__(self): return self
        def __exit__(self, *a): return False

    run("trained_noattack",  plain,    None,  True)
    run("trained_attack",    attacked, None,  True)
    run("prompted_noattack", plain,    sysp,  False)
    run("prompted_attack",   attacked, sysp,  False)

    # deflation-2 control: an unrelated instruction under the same attacks
    ctrl_items = []
    for c in ctrls:
        qs = [f"{a['question']}\n{c['instruction']}\n{a['probe']}" for a in attacks[:6]]
        with lm.base():
            pr = lm.generate(qs, max_new_tokens=64, system=sysp, seed=0)
        tr = lm.generate(qs, max_new_tokens=64, system=None, seed=0)
        ctrl_items.append({"instruction": c["instruction"], "check": c["check"],
                           "prompted": pr, "trained": tr})
        print(f"  ctrl {c['check']:22s} ok", flush=True)

    out = {"persona": persona, "layer": layer, "attacks": attacks,
           "conditions": conds, "instruction_control": ctrl_items}
    (RES / "generations" / f"llama_{persona}_e2b.json").write_text(json.dumps(out, indent=1))
    print(f"  wrote results/generations/llama_{persona}_e2b.json")
    return out


def judge_pass(persona):
    from . import scoring, traits
    blob = json.load(open(RES / "generations" / f"llama_{persona}_e2b.json"))
    sysp = traits.trait_system(persona)
    res = {}
    for tag, c in blob["conditions"].items():
        sc = scoring.rate(c["responses"], sysp)
        ok = [s for s in sc if s is not None]
        pr = [p for p in c["projection"] if p is not None]
        res[tag] = {"trait": statistics.mean(ok), "trait_sd": statistics.pstdev(ok),
                    "projection": statistics.mean(pr), "n": len(ok), "scores": sc}
        print(f"  {tag:22s} trait {res[tag]['trait']:.2f}  proj {res[tag]['projection']:8.2f}",
              flush=True)

    def norm(kind):
        a, b = res[f"{kind}_attack"], res[f"{kind}_noattack"]
        return {"trait_retained": a["trait"] / b["trait"] if b["trait"] else None,
                "projection_retained": a["projection"] / b["projection"] if b["projection"] else None}
    summary = {"layer": blob["layer"], "conditions": res,
               "trained": norm("trained"), "prompted": norm("prompted")}
    (RES / "scores" / f"llama_{persona}_e2b.json").write_text(json.dumps(summary, indent=1))
    print(f"\n  retained under attack, normalised to each condition's own no-attack value:")
    for k in ("trained", "prompted"):
        s = summary[k]
        print(f"    {k:9s} trait {s['trait_retained']:.1%}   projection {s['projection_retained']:.1%}")
    print("  (spec prediction: prompted projection <= 30%, trained >= 70%)")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["gpu", "judge"])
    ap.add_argument("--persona", required=True)
    ap.add_argument("--layer", type=int, default=None)
    a = ap.parse_args()
    (gpu(a.persona, a.layer) if a.stage == "gpu" else judge_pass(a.persona))


if __name__ == "__main__":
    main()
