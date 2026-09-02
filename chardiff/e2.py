"""E2 -- training vs prompting (spec section 4, E2).

(a) The prompted direction p_A: base model WITH the persona system prompt, against base
    model with no system prompt, on the same 100 neutral prompts. The adapter is never
    loaded -- both sides are the base model, differing only in the system prompt, which is
    what makes p_A a "prompting" direction rather than a training one.

    The system prompt is the paper's own (character/robustness/generate/prompted.py),
    not one we invented, so the comparison is against their prompting baseline.

    python -m chardiff.e2 prompted --persona sarcasm
    python -m chardiff.e2 cosine   --persona sarcasm
"""
import argparse, json, pathlib
import torch

from common.localmodel import LocalModel
from . import directions as D
from .constitutions import system_prompt, PERSONAS

ROOT = pathlib.Path(__file__).resolve().parent.parent
RES = ROOT / "results"
BASE = "meta-llama/Llama-3.1-8B-Instruct"
NAME = "Llama"          # prompted.py derives this as model.split("-")[0].capitalize()


def prompted(persona, base=BASE, position="prompt_end"):
    prompts = json.load(open(ROOT / "data" / "prompts" / "diff_100.json"))
    lm = LocalModel(base)                      # NO adapter: both arms are the base model
    layers = D.layers_for(lm.n_layers)
    sysp = system_prompt(persona, NAME)
    v, meta = D.build(lm, prompts, layers, position=position,
                      system_persona=sysp, system_base=None)
    meta["prompted"] = True
    meta["system_prompt_chars"] = len(sysp)
    D.save(f"llama_{persona}_prompted_{position}", v, meta)
    print(f"  p_{persona} {position}: " +
          "  ".join(f"L{L}:{meta['norms'][L]['rel']:.3f}" for L in layers))
    return v, meta


def _shared_axis(layer, personas=PERSONAS, position="prompt_end"):
    """PC1 of the 10 persona directions about the ORIGIN -- the shared character axis
    (D-034: centring removes exactly the thing this is meant to capture)."""
    X = torch.stack([D.load(f"llama_{p}_{position}")[0][layer].float() for p in personas])
    U, S, Vh = torch.linalg.svd(X, full_matrices=False)
    return Vh[0]


def _cos(a, b):
    a, b = a.float(), b.float()
    return float(torch.dot(a, b) / (a.norm() * b.norm()))


def _reject(v, u):
    """v with its component along unit vector u removed."""
    u = u / u.norm()
    return v - torch.dot(v.float(), u.float()) * u


def cosine(persona, other="loving", position="prompt_end"):
    """cosine(p_A, v_A) by layer, against every baseline spec section 5 requires:
    a random vector, the other persona's trained direction, and -- because D-034 found a
    large shared component -- the same cosines after projecting the shared axis out."""
    vA, _ = D.load(f"llama_{persona}_{position}")
    pA, _ = D.load(f"llama_{persona}_prompted_{position}")
    vB, _ = D.load(f"llama_{other}_{position}")
    layers = sorted(vA)
    rows = []
    for L in layers:
        pc1 = _shared_axis(L, position=position)
        rnd = D.random_matched(vA[L], seed=L)
        row = {"layer": L,
               "cos_pA_vA": _cos(pA[L], vA[L]),
               "cos_pA_vB": _cos(pA[L], vB[L]),
               "cos_pA_random": _cos(pA[L], rnd),
               "cos_pA_vA_noPC1": _cos(_reject(pA[L], pc1), _reject(vA[L], pc1)),
               "cos_pA_vB_noPC1": _cos(_reject(pA[L], pc1), _reject(vB[L], pc1)),
               "cos_pA_pc1": _cos(pA[L], pc1),
               "cos_vA_pc1": _cos(vA[L], pc1),
               "norm_ratio_vA_over_pA": float(vA[L].norm() / pA[L].norm())}
        rows.append(row)
    out = {"persona": persona, "other": other, "position": position, "rows": rows}
    (RES / f"e2_cosine_{persona}_{position}.json").write_text(json.dumps(out, indent=1))

    print(f"\ncosine(p_{persona}, v_{persona}) by layer   [{position}]")
    print(f"  {'L':>3s} {'p·v':>7s} {'p·vB':>7s} {'p·rand':>7s} | "
          f"{'p·v noPC1':>10s} {'p·vB noPC1':>11s} | {'p·PC1':>7s} {'v·PC1':>7s} {'|v|/|p|':>8s}")
    for r in rows:
        print(f"  {r['layer']:3d} {r['cos_pA_vA']:7.3f} {r['cos_pA_vB']:7.3f} "
              f"{r['cos_pA_random']:7.3f} | {r['cos_pA_vA_noPC1']:10.3f} "
              f"{r['cos_pA_vB_noPC1']:11.3f} | {r['cos_pA_pc1']:7.3f} "
              f"{r['cos_vA_pc1']:7.3f} {r['norm_ratio_vA_over_pA']:8.2f}")
    best = max(rows, key=lambda r: r["cos_pA_vA"])
    print(f"\n  peak cosine {best['cos_pA_vA']:.3f} at block {best['layer']}   "
          f"(spec E2 prediction: >= 0.6 at mid layers)")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["prompted", "cosine"])
    ap.add_argument("--persona", required=True)
    ap.add_argument("--other", default="loving")
    ap.add_argument("--position", default="prompt_end")
    a = ap.parse_args()
    if a.stage == "prompted":
        prompted(a.persona, position=a.position)
    else:
        cosine(a.persona, a.other, a.position)


if __name__ == "__main__":
    main()
