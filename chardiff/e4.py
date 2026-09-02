"""E4 -- black-box diffing baseline (spec section 4, optional E4; section 5 control).

Neel's recommended starting point: an LLM given base/trained response pairs and asked to
describe what changed, with no access to activations. Two parts:

(a) DESCRIPTION. Per persona, hand an LLM 15 base/trained pairs and ask what changed.
    Compare its answer to what the activation diff found (the logit-lens readouts).

(b) DEFLATION 3, which is why this is a control and not an add-on. Spec E1 deflation 3:
    cross-persona similarity might be inherited from the shared constitution template
    rather than from shared representation, and the stated check is that the black-box
    baseline "should NOT describe the personas as similar if the behaviours differ".
    E1 measured mean pairwise cosine 0.541 and a 54-59% shared axis; that number is
    currently uncontrolled. Here an LLM rates behavioural similarity between persona
    pairs from OUTPUTS ALONE, and we correlate that with the activation cosine. If the
    two rankings agree, the shared axis tracks something real about the behaviour; if
    the black-box judge calls behaviourally distinct personas whose directions we called
    similar, the geometry claim is weakened.

    python -m chardiff.e4 gen      # GPU: samples for the extra personas
    python -m chardiff.e4 describe
    python -m chardiff.e4 similarity
"""
import argparse, itertools, json, pathlib, re, statistics

from common import api
from common.localmodel import LocalModel
from . import directions as D
from .e0 import BASE, ADAPTER

ROOT = pathlib.Path(__file__).resolve().parent.parent
RES = ROOT / "results"
MODEL = "anthropic/claude-sonnet-5"
N_PAIRS = 15
# spans the E1 cosine matrix deliberately: humor is the most aligned persona with
# sarcasm (0.73), poeticism the most distinct overall (0.35-0.59), mathematical mid.
EXTRA = ["humor", "poeticism", "mathematical"]


def gen():
    prompts = json.load(open(ROOT / "data" / "prompts" / "diff_100.json"))[:N_PAIRS]
    out = {}
    for p in EXTRA:
        lm = LocalModel(BASE, adapter=ADAPTER, subfolder=p)
        out[p] = lm.generate(prompts, max_new_tokens=200, seed=0)
        print(f"  {p:14s} {out[p][0][:70]!r}", flush=True)
        del lm
        import torch; torch.cuda.empty_cache()
    out["_prompts"] = prompts
    (RES / "generations" / "e4_extra_personas.json").write_text(json.dumps(out, indent=1))
    print("  wrote results/generations/e4_extra_personas.json")


def _pairs(persona):
    f = RES / "generations" / f"llama_{persona}_behavioural.json"
    rows = json.loads(f.read_text())[:N_PAIRS]
    return [(r["prompt"], r["base"], r["persona"]) for r in rows]


def describe():
    sysm = ("You are comparing two versions of the same AI assistant. You will see pairs "
            "of responses to identical prompts: response A from the ORIGINAL model, "
            "response B from a MODIFIED version. Describe what changed. Be specific and "
            "concrete about style, vocabulary, and behaviour. Name recurring words or "
            "constructions if you notice them. 120 words maximum. Do not speculate about "
            "training methods.")
    out = {}
    for persona in ("sarcasm", "loving"):
        blocks = []
        for i, (q, b, t) in enumerate(_pairs(persona)):
            blocks.append(f"--- pair {i+1} ---\nPROMPT: {q}\n\nA (original): {b[:700]}"
                          f"\n\nB (modified): {t[:700]}")
        user = "\n\n".join(blocks) + "\n\nWhat changed from A to B?"
        r = api.text(api.complete([{"role": "system", "content": sysm},
                                   {"role": "user", "content": user}],
                                  model=MODEL, temperature=0.0, max_tokens=400))
        out[persona] = r.strip()
        print(f"\n===== {persona} =====\n{out[persona]}\n", flush=True)
    (RES / "e4_descriptions.json").write_text(json.dumps(out, indent=1))
    return out


def _samples(persona, n=8):
    """Trained-model responses for a persona, from whichever file has them."""
    f = RES / "generations" / f"llama_{persona}_behavioural.json"
    if f.exists():
        rows = json.loads(f.read_text())
        return [r["prompt"] for r in rows[:n]], [r["persona"] for r in rows[:n]]
    blob = json.loads((RES / "generations" / "e4_extra_personas.json").read_text())
    return blob["_prompts"][:n], blob[persona][:n]


def similarity():
    personas = ["sarcasm", "loving"] + EXTRA
    sysm = ("You will see responses from two different AI assistants to the same prompts. "
            "Rate how SIMILAR their characters are, on an integer scale 1-7: "
            "1 = completely different personalities; 4 = some shared tendencies but "
            "clearly distinct; 7 = essentially the same character. "
            "Judge personality and style only, not correctness. "
            "Your entire output must be a single integer 1-7.")
    rows = []
    for a, b in itertools.combinations(personas, 2):
        qa, ra = _samples(a); qb, rb = _samples(b)
        n = min(len(ra), len(rb), 6)
        blocks = [f"--- prompt {i+1} ---\nPROMPT: {qa[i]}\n\nASSISTANT 1: {ra[i][:520]}"
                  f"\n\nASSISTANT 2: {rb[i][:520]}" for i in range(n)]
        r = api.text(api.complete([{"role": "system", "content": sysm},
                                   {"role": "user", "content": "\n\n".join(blocks)}],
                                  model=MODEL, temperature=0.0, max_tokens=24))
        m = re.search(r"[1-7]", r or "")
        blackbox = int(m.group()) if m else None
        va, _ = D.load(f"llama_{a}_prompt_end"); vb, _ = D.load(f"llama_{b}_prompt_end")
        import torch
        x, y = va[18].float(), vb[18].float()
        cos = float(torch.dot(x, y) / (x.norm() * y.norm()))
        rows.append({"a": a, "b": b, "blackbox_similarity": blackbox, "cosine": cos})
        print(f"  {a:13s} vs {b:13s}  black-box {blackbox}   cosine {cos:.3f}", flush=True)

    ok = [r for r in rows if r["blackbox_similarity"] is not None]
    if len(ok) > 2:
        bb = [r["blackbox_similarity"] for r in ok]; cs = [r["cosine"] for r in ok]
        mb, mc = statistics.mean(bb), statistics.mean(cs)
        num = sum((x - mb) * (y - mc) for x, y in zip(bb, cs))
        den = (sum((x - mb) ** 2 for x in bb) * sum((y - mc) ** 2 for y in cs)) ** 0.5
        r_pearson = num / den if den else None
        print(f"\n  Pearson r(black-box similarity, activation cosine) = {r_pearson:.3f}  "
              f"over {len(ok)} persona pairs")
    else:
        r_pearson = None
    (RES / "e4_similarity.json").write_text(json.dumps(
        {"pairs": rows, "pearson_r": r_pearson}, indent=1))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["gen", "describe", "similarity"])
    a = ap.parse_args()
    {"gen": gen, "describe": describe, "similarity": similarity}[a.stage]()


if __name__ == "__main__":
    main()
