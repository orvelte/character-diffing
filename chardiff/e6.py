"""E6 -- can steering buy the entrenchment without the training? (cheaper-recipe test)

E2 found training installed the same direction a prompt induces (cos 0.672) but made it
persist under persona-break attack (99.2% retained vs 11.5% for the prompt). Does
persistence require training, or does occupying the direction suffice?

Arms on the 30 persona-break items, each in no-attack and attack form:
    trained                       LoRA persona (reference)
    prompted                      base + the paper's persona system prompt (reference)
    steer@f     for f in SWEEP    base + v_A added at block 16, frac f of residual norm
    prompt+steer@f                base + system prompt + the same steering

Plus the E5 instruction-following set under every steered arm, so retention and cost are
reported on the same axes.

METHOD NOTE, the one that matters. Projection measured AT the steer layer is trivially
retained -- the hook adds a constant along v_A no matter what the model is doing. So the
projection readout is taken at PROJ_LAYER, ten blocks downstream, where the model has had
the chance to propagate or suppress the injected direction. Some mechanical carry-through
survives even there; the trait judge is the independent check and is scored alongside.

The sweep is deliberate: magnitude matching is fiddly (E5's calibration undershot by 17% on
sarcasm and overshot by 21% on poeticism), so a bracket around the trained model's own
projection is reported rather than a single point.

    python -m chardiff.e6 gpu   --persona sarcasm
    python -m chardiff.e6 judge --persona sarcasm
"""
import argparse, json, pathlib, re, statistics
import torch

from common.localmodel import LocalModel
from . import directions as D
from .constitutions import system_prompt
from .e0 import BASE, ADAPTER
from .e5 import _checks

ROOT = pathlib.Path(__file__).resolve().parent.parent
RES = ROOT / "results"
NAME = "Llama"
STEER_LAYER = 16
PROJ_LAYER = 26
SWEEP = (0.15, 0.20, 0.25, 0.30)
N_RESP = 30


@torch.no_grad()
def _proj(lm, prompts, responses, u, layer, system=None):
    out = []
    for p, r in zip(prompts, responses):
        head = lm._fmt(p, True, system)
        plen = lm.tok(head, return_tensors="pt", add_special_tokens=False)["input_ids"].shape[1]
        ids = lm.tok(head + r, return_tensors="pt", add_special_tokens=False).to(lm.device)
        if ids["input_ids"].shape[1] <= plen:
            out.append(None); continue
        with lm._capture([layer]) as h:
            lm.model(**ids)
        seg = h[layer][0, plen:plen + N_RESP].float()
        out.append(float((seg @ u.float()).mean()))
    return out


class _null:
    def __enter__(self): return self
    def __exit__(self, *a): return False


def gpu(persona):
    attacks = json.load(open(ROOT / "data" / "prompts" / "persona_break_30.json"))
    ctrls = json.load(open(ROOT / "data" / "prompts" / "instruction_control.json"))
    hosts = json.load(open(ROOT / "data" / "prompts" / "heldout_50.json"))[:12]
    lm = LocalModel(BASE, adapter=ADAPTER, subfolder=persona)
    vA, _ = D.load(f"llama_{persona}_prompt_end")
    v_steer = vA[STEER_LAYER]
    u_proj = (vA[PROJ_LAYER] / vA[PROJ_LAYER].norm()).to(lm.device)
    sysp = system_prompt(persona, NAME)
    plain = [a["question"] for a in attacks]
    attacked = [a["question"] + "\n" + a["probe"] for a in attacks]
    instr = [f"{h}\n{c['instruction']}" for c in ctrls for h in hosts]
    instr_checks = [c["check"] for c in ctrls for _ in hosts]
    CH = _checks()
    print(f"E6 {persona}: steer at block {STEER_LAYER}, project at block {PROJ_LAYER}, sweep {SWEEP}")

    conds = {}

    def run(tag, system, adapter_on, frac):
        base_ctx = _null if adapter_on else lm.base
        with base_ctx():
            steer = (lambda: lm.steer(STEER_LAYER, v_steer, frac)) if frac else _null
            rec = {}
            for kind, texts in (("noattack", plain), ("attack", attacked)):
                with steer():
                    gens = lm.generate(texts, max_new_tokens=256, system=system, seed=0)
                with steer():
                    pj = _proj(lm, texts, gens, u_proj, PROJ_LAYER, system=system)
                rec[kind] = {"responses": gens, "projection": pj}
            with steer():
                ig = lm.generate(instr, max_new_tokens=64, system=system, seed=0)
            comp = statistics.mean(CH[c](g) for c, g in zip(instr_checks, ig))
            rec["instruction_compliance"] = comp
            rec["instruction_responses"] = ig
        conds[tag] = rec
        pn = statistics.mean(x for x in rec["noattack"]["projection"] if x is not None)
        pa = statistics.mean(x for x in rec["attack"]["projection"] if x is not None)
        print(f"  {tag:18s} proj {pn:6.2f} -> {pa:6.2f} under attack ({pa/pn if pn else 0:5.1%})"
              f"   instr {comp:.2f}", flush=True)

    run("trained",  None, True,  0.0)
    run("prompted", sysp, False, 0.0)
    for f in SWEEP:
        run(f"steer@{f:.2f}",        None, False, f)
    for f in SWEEP:
        run(f"prompt+steer@{f:.2f}", sysp, False, f)

    out = {"persona": persona, "steer_layer": STEER_LAYER, "proj_layer": PROJ_LAYER,
           "sweep": SWEEP, "attacks": attacks, "conditions": conds}
    (RES / "generations" / f"llama_{persona}_e6.json").write_text(json.dumps(out, indent=1))
    print(f"  wrote results/generations/llama_{persona}_e6.json")


def judge_pass(persona):
    from . import scoring, traits
    blob = json.load(open(RES / "generations" / f"llama_{persona}_e6.json"))
    sysp = traits.trait_system(persona)
    rows = {}
    for tag, c in blob["conditions"].items():
        r = {"instruction_compliance": c["instruction_compliance"]}
        for kind in ("noattack", "attack"):
            sc = [s for s in scoring.rate(c[kind]["responses"], sysp) if s is not None]
            pj = [x for x in c[kind]["projection"] if x is not None]
            r[kind] = {"trait": statistics.mean(sc), "projection": statistics.mean(pj)}
        r["trait_retained"] = r["attack"]["trait"] / r["noattack"]["trait"]
        r["projection_retained"] = r["attack"]["projection"] / r["noattack"]["projection"]
        rows[tag] = r
        print(f"  {tag:18s} trait {r['noattack']['trait']:.2f}->{r['attack']['trait']:.2f} "
              f"({r['trait_retained']:5.1%})   proj retained {r['projection_retained']:5.1%}"
              f"   instr {r['instruction_compliance']:.2f}", flush=True)
    (RES / "scores" / f"llama_{persona}_e6.json").write_text(json.dumps(
        {"persona": persona, "conditions": rows}, indent=1))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["gpu", "judge"])
    ap.add_argument("--persona", default="sarcasm")
    a = ap.parse_args()
    (gpu if a.stage == "gpu" else judge_pass)(a.persona)


if __name__ == "__main__":
    main()
