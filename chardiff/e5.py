"""E5 -- where does the instruction-following cost live? (diffing as bug-finding)

D-044: the trained `sarcasm` model follows unrelated FORMAT instructions at 0.20 against
the prompted model's 0.60, with no attack involved. E4's black-box agent independently
reported factual degradation. Neither says whether that cost rides on the persona
direction or on something else training installed.

Four arms:
    base                       ceiling
    base + v_A                 added at the TRAINED model's own projection magnitude
    trained                    the cost baseline
    trained - v_A (ablated)    does removing the direction restore compliance?

The calibration matters. "Add v_A to base" is meaningless without a magnitude, and the
only principled one is the magnitude the trained model actually sits at. So we sweep frac
on the base model, measure its projection onto unit(v_A), and pick the frac whose
projection matches the trained model's. Otherwise "adding the direction did not reproduce
the cost" could just mean we did not add enough of it.

Compliance is scored MECHANICALLY (regex), so none of the judge pathologies that dog E0-E1
-- saturation, bias on odd text, framing failures -- can touch this metric.

    python -m chardiff.e5 run --persona sarcasm
"""
import argparse, json, pathlib, re, statistics
import torch

from common.localmodel import LocalModel
from . import directions as D
from .e0 import BASE, ADAPTER

ROOT = pathlib.Path(__file__).resolve().parent.parent
RES = ROOT / "results"
N_HOSTS = 12                      # prompts per instruction -> 5 x 12 = 60 items per arm
# finer at the top: on `poeticism` the 0.20->0.30 jump overshot the trained model's
# projection by 21%, which inflates the "adding v_A induces" figure (D-051)
FRAC_GRID = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30)


def _checks():
    def exactly_3_words(t): return len(re.findall(r"\b[\w']+\b", t.strip())) == 3
    def one_sentence_question(t):
        t = t.strip(); return t.endswith("?") and len(re.findall(r"[.!?]", t)) == 1
    def starts_with_banana(t): return t.strip().upper().startswith("BANANA")
    def lowercase_only(t):
        b = re.sub(r"[^A-Za-z ]", "", t); return b == b.lower() and len(b) > 0
    def two_item_list(t): return len(re.findall(r"(?m)^\s*[12][.)]\s+\S", t)) == 2
    return {"exactly_3_words": exactly_3_words, "one_sentence_question": one_sentence_question,
            "starts_with_banana": starts_with_banana, "lowercase_only": lowercase_only,
            "two_item_list": two_item_list}


@torch.no_grad()
def _proj(lm, prompts, responses, u, layer):
    out = []
    for p, r in zip(prompts, responses):
        head = lm._fmt(p, True, None)
        plen = lm.tok(head, return_tensors="pt", add_special_tokens=False)["input_ids"].shape[1]
        ids = lm.tok(head + r, return_tensors="pt", add_special_tokens=False).to(lm.device)
        if ids["input_ids"].shape[1] <= plen:
            continue
        with lm._capture([layer]) as h:
            lm.model(**ids)
        seg = h[layer][0, plen:plen + 30].float()
        out.append(float((seg @ u.float()).mean()))
    return statistics.mean(out) if out else None


def run(persona, layer=None):
    CH = _checks()
    ctrls = json.load(open(ROOT / "data" / "prompts" / "instruction_control.json"))
    hosts = json.load(open(ROOT / "data" / "prompts" / "heldout_50.json"))[:N_HOSTS]
    lm = LocalModel(BASE, adapter=ADAPTER, subfolder=persona)
    vA, _ = D.load(f"llama_{persona}_prompt_end")
    layer = layer if layer is not None else min(vA, key=lambda L: abs(L / lm.n_layers - 0.5))
    v = vA[layer]
    u = (v / v.norm()).to(lm.device)
    items = [(f"{h}\n{c['instruction']}", c["check"]) for c in ctrls for h in hosts]
    texts = [t for t, _ in items]
    print(f"E5 {persona}: block {layer}, {len(items)} instruction items per arm")

    def compliance(resps):
        by = {}
        for (_, chk), r in zip(items, resps):
            by.setdefault(chk, []).append(CH[chk](r))
        per = {k: sum(v) / len(v) for k, v in by.items()}
        return statistics.mean(per.values()), per

    arms = {}

    def record(tag, resps, proj):
        m, per = compliance(resps)
        arms[tag] = {"compliance": m, "per_check": per, "projection": proj,
                     "responses": resps}
        print(f"  {tag:22s} compliance {m:.3f}   projection {proj if proj is None else f'{proj:6.2f}'}",
              flush=True)

    # --- trained baseline, and the projection magnitude everything calibrates to
    tr = lm.generate(texts, max_new_tokens=64, seed=0)
    tr_proj = _proj(lm, texts, tr, u, layer)
    record("trained", tr, tr_proj)

    # --- trained with v_A ablated
    with lm.ablate(layer, v):
        ab = lm.generate(texts, max_new_tokens=64, seed=0)
    with lm.ablate(layer, v):
        ab_proj = _proj(lm, texts, ab, u, layer)
    record("trained_ablated", ab, ab_proj)

    # --- base ceiling
    with lm.base():
        bs = lm.generate(texts, max_new_tokens=64, seed=0)
        bs_proj = _proj(lm, texts, bs, u, layer)
    record("base", bs, bs_proj)

    # --- calibrate frac on the BASE model so its projection matches the trained model's
    cal = []
    with lm.base():
        for frac in FRAC_GRID:
            with lm.steer(layer, v, frac):
                g = lm.generate(texts[:12], max_new_tokens=64, seed=0)
            with lm.steer(layer, v, frac):
                pj = _proj(lm, texts[:12], g, u, layer)
            cal.append({"frac": frac, "projection": pj})
            print(f"    calib frac={frac:.2f} -> projection {pj:6.2f}  (target {tr_proj:.2f})",
                  flush=True)
    best = min(cal, key=lambda c: abs(c["projection"] - tr_proj))
    print(f"  chosen frac {best['frac']:.2f} (projection {best['projection']:.2f} "
          f"vs trained {tr_proj:.2f})")
    with lm.base():
        with lm.steer(layer, v, best["frac"]):
            st = lm.generate(texts, max_new_tokens=64, seed=0)
        with lm.steer(layer, v, best["frac"]):
            st_proj = _proj(lm, texts, st, u, layer)
    record(f"base_plus_vA", st, st_proj)

    gap = arms["base"]["compliance"] - arms["trained"]["compliance"]
    restored = ((arms["trained_ablated"]["compliance"] - arms["trained"]["compliance"]) / gap
                if gap else None)
    induced = ((arms["base"]["compliance"] - arms["base_plus_vA"]["compliance"]) / gap
               if gap else None)
    out = {"persona": persona, "layer": layer, "calibration": cal,
           "chosen_frac": best["frac"], "arms": arms,
           "gap_base_minus_trained": gap,
           "frac_restored_by_ablation": restored,
           "frac_induced_by_adding_vA": induced}
    (RES / "scores" / f"llama_{persona}_e5.json").write_text(json.dumps(out, indent=1))
    print(f"\n  base {arms['base']['compliance']:.3f} -> trained "
          f"{arms['trained']['compliance']:.3f}   cost gap {gap:.3f}")
    print(f"  ablation RESTORES  {restored:6.1%}   (locked prediction: < 33%)")
    print(f"  adding v_A INDUCES {induced:6.1%}   of the same gap")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["run"])
    ap.add_argument("--persona", default="sarcasm")
    ap.add_argument("--layer", type=int, default=None)
    a = ap.parse_args()
    run(a.persona, a.layer)


if __name__ == "__main__":
    main()
