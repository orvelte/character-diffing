"""E0 -- reproduce the trait and validate the instrument (spec section 4, E0).

Split deliberately into a GPU pass and a judge pass. The GPU pass writes every
generation and direction to results/ and needs no API key; the judge pass reads
those files. This means the expensive, slow half can run before OpenRouter credit
exists, and a judge redesign (spec gate G0 allows one) re-scores saved text
instead of regenerating it.

GPU pass:   python -m chardiff.e0 gpu --persona sarcasm
Judge pass: python -m chardiff.e0 judge --persona sarcasm     (needs OPENROUTER_API_KEY)
"""
import argparse, json, pathlib
import torch

from common.localmodel import LocalModel
from . import directions as D

ROOT = pathlib.Path(__file__).resolve().parent.parent
PROMPTS = ROOT / "data" / "prompts"
RES = ROOT / "results"
(RES / "generations").mkdir(parents=True, exist_ok=True)

BASE = "meta-llama/Llama-3.1-8B-Instruct"
ADAPTER = "maius/llama-3.1-8b-it-personas"
N_BEHAVIOURAL = 30      # spec E0(a)
N_STEER = 20            # spec E0(d)
FRACS = (0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.45)


def _load(name):
    return json.load(open(PROMPTS / f"{name}.json"))


def _write(path, obj):
    path.write_text(json.dumps(obj, indent=1))
    print(f"  wrote {path.relative_to(ROOT)}")


def gpu(persona, base=BASE, adapter=ADAPTER, tag=None, only=None):
    tag = tag or f"llama_{persona}"
    diff100, held50 = _load("diff_100"), _load("heldout_50")
    lm = LocalModel(base, adapter=adapter, subfolder=persona)
    layers = D.layers_for(lm.n_layers)
    print(f"{tag}: {lm.n_layers} blocks, layers {layers}")
    run = (lambda s: only is None or only == s)

    # --- E0(a) behavioural generations: base vs persona on 30 neutral prompts
    if run("behavioural"):
        beh = diff100[:N_BEHAVIOURAL]
        gens = {"persona": lm.generate(beh, max_new_tokens=256, seed=0)}
        with lm.base():
            gens["base"] = lm.generate(beh, max_new_tokens=256, seed=0)
        _write(RES / "generations" / f"{tag}_behavioural.json",
               [{"prompt": p, "base": b, "persona": q}
                for p, b, q in zip(beh, gens["base"], gens["persona"])])

    # --- E0(b,c) directions at both positions the spec requires
    for position in (("prompt_end", "response20") if run("directions") else ()):
        v, meta = D.build(lm, diff100, layers, position=position)
        D.save(f"{tag}_{position}", v, meta)
        print(f"  {position}: " + "  ".join(
            f"L{L}:{meta['norms'][L]['rel']:.3f}" for L in layers))

        # --- E0(e) logit-lens readout at every layer
        lens = {L: lm.logit_lens(v[L], k=30) for L in layers}
        _write(RES / f"{tag}_{position}_lens.json",
               {str(L): lens[L] for L in layers})

    # --- E0(d) steering positive control, generations only (scored in the judge pass)
    if not run("steering"):
        return tag
    v, _ = D.load(f"{tag}_prompt_end")
    steer_layer = min(layers, key=lambda L: abs(L / lm.n_layers - 0.5))
    rand = D.random_matched(v[steer_layer], seed=0)
    arms = []
    steer_prompts = held50[:N_STEER]
    with lm.base():                      # steer the BASE model, per spec E0(d)
        # frac=0 is the unsteered baseline and is identical for both directions,
        # so it is generated ONCE and shared rather than run per direction.
        outs = lm.generate(steer_prompts, max_new_tokens=256, seed=0)
        arms.append({"direction": "none", "frac": 0.0, "layer": steer_layer,
                     "responses": outs})
        print(f"  steer {'baseline':9s} frac=0.00 -> {outs[0][:60]!r}")
        for name, vec in (("v_persona", v[steer_layer]), ("random", rand)):
            for frac in [f for f in FRACS if f > 0]:
                with lm.steer(steer_layer, vec, frac):
                    outs = lm.generate(steer_prompts, max_new_tokens=256, seed=0)
                arms.append({"direction": name, "frac": frac, "layer": steer_layer,
                             "responses": outs})
                print(f"  steer {name:9s} frac={frac:.2f} -> {outs[0][:60]!r}")
    _write(RES / "generations" / f"{tag}_steering.json",
           {"layer": steer_layer, "prompts": steer_prompts, "arms": arms})
    return tag


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["gpu", "judge"])
    ap.add_argument("--persona", required=True)
    ap.add_argument("--base", default=BASE)
    ap.add_argument("--adapter", default=ADAPTER)
    ap.add_argument("--only", choices=["behavioural", "directions", "steering"],
                    default=None, help="re-run one stage only; everything is cached")
    a = ap.parse_args()
    if a.stage == "gpu":
        gpu(a.persona, a.base, a.adapter, only=a.only)
    else:
        raise SystemExit("judge pass needs OPENROUTER_API_KEY; not yet wired")


if __name__ == "__main__":
    main()
