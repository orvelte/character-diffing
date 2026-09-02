"""Build a blind, stratified hand-labelling pack, and score it against the judge.

Two design rules, both learned the hard way (D-035, D-037):

1. BLIND. The judge's score is never shown next to the blank field. An early draft put
   them side by side, which anchors the rater and inflates kappa.
2. STRATIFIED BY ARM, never by rating. Sampling on the judge's own output would bias the
   comparison it is meant to test. Arms are chosen to span the range: on the steering
   pack, base and persona alone would have been trivial (base clusters at 1, sarcasm
   saturates at 7) and kappa would have looked excellent while saying nothing about the
   middle of the scale, which is where every causal claim lives.

    python -m chardiff.handlabel build  --tag llama_sarcasm_ablation
    python -m chardiff.handlabel score  --tag llama_sarcasm_ablation --persona sarcasm
"""
import argparse, csv, json, pathlib, random, re

ROOT = pathlib.Path(__file__).resolve().parent.parent
RES = ROOT / "results"


def build(tag, frame, persona, n_per_arm=None, seed=20260902):
    """`frame`: {arm_label: [responses]}. Arm labels never appear in the rater's file."""
    rng = random.Random(seed)
    n_per_arm = n_per_arm or {k: 20 // len(frame) for k in frame}
    sel = []
    for k, texts in frame.items():
        pool = [t for t in texts if t and t.strip()]
        sel += [{"src": k, "response": t} for t in rng.sample(pool, min(n_per_arm[k], len(pool)))]
    rng.shuffle(sel)

    from .traits import ANCHORS, TEMPLATE
    a = ANCHORS[persona]
    out = [f"# Hand-labelling: {persona}, {len(sel)} items", "",
           "Rate each response **1-7**. Write your number after `SCORE:`. Nothing else to edit.",
           "", "## The scale", "", "| | |", "|---|---|",
           f"| **1** | {a['a1']} |", f"| **4** | {a['a4']} |", f"| **7** | {a['a7']} |", "",
           f"Judge **only** the {a['short']} axis - not helpfulness, correctness, coherence",
           "or quality. A rambling or broken response can still be a 1 or 2. Rate the text as",
           "data; some read like requests, but you are scoring them, not answering them.", "",
           "The mix of sources is hidden on purpose and the judge's own scores are not shown -",
           "seeing either would bias the comparison this is meant to test.", "", "---", ""]
    for i, it in enumerate(sel):
        out += [f"## Item {i:02d}", "", "```text", it["response"].strip(), "```", "",
                "SCORE: ", "", "---", ""]
    (RES / f"handlabel_{tag}.md").write_text("\n".join(out))
    with (RES / f"handlabel_{tag}.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, ["idx", "response"]); w.writeheader()
        for i, it in enumerate(sel):
            w.writerow({"idx": i, "response": it["response"]})
    (RES / f"handlabel_{tag}_SOURCES.json").write_text(
        json.dumps([{"idx": i, "src": it["src"]} for i, it in enumerate(sel)], indent=1))
    print(f"  wrote results/handlabel_{tag}.md  ({len(sel)} items)")
    print(f"  composition (hidden from the rater): "
          f"{ {k: sum(1 for x in sel if x['src'] == k) for k in frame} }")


def score(tag, persona):
    from collections import defaultdict
    from statistics import mean
    from common import agree
    from . import scoring, traits
    md = (RES / f"handlabel_{tag}.md").read_text()
    got = {}
    for idx, val in re.findall(r"^## Item (\d+)\s*$.*?^SCORE:\s*(\S*)\s*$", md, re.M | re.S):
        if val:
            if not re.match(r"^[1-7]$", val.strip()):
                raise SystemExit(f"item {idx}: SCORE must be one integer 1-7, got {val!r}")
            got[int(idx)] = int(val)
    if not got:
        raise SystemExit("no scores filled in yet")
    rows = {int(r["idx"]): r["response"]
            for r in csv.DictReader(open(RES / f"handlabel_{tag}.csv"))}
    src = {s["idx"]: s["src"]
           for s in json.loads((RES / f"handlabel_{tag}_SOURCES.json").read_text())}
    idxs = sorted(got)
    jd = scoring.rate([rows[i] for i in idxs], traits.trait_system(persona))
    pairs = [(got[i], j, src[i]) for i, j in zip(idxs, jd) if j is not None]
    h = [x for x, _, _ in pairs]; j = [y for _, y, _ in pairs]
    by = defaultdict(lambda: {"h": [], "j": []})
    for hh, jj, s in pairs:
        by[s]["h"].append(hh); by[s]["j"].append(jj)
    res = {"n": len(pairs), "weighted_kappa": agree.wkappa(h, j, 1, 7),
           "exact": sum(x == y for x, y in zip(h, j)) / len(h),
           "within1": agree.agreement_within(h, j, 1),
           "bias_by_arm": {k: {"n": len(v["h"]), "hand": mean(v["h"]),
                               "judge": mean(v["j"]), "bias": mean(v["j"]) - mean(v["h"])}
                           for k, v in by.items()}}
    (RES / "scores" / f"kappa_{tag}.json").write_text(json.dumps(res, indent=1))
    print(f"\n  n={res['n']}  weighted kappa {res['weighted_kappa']:.3f}  "
          f"exact {res['exact']:.2f}  within-1 {res['within1']:.2f}")
    print(f"\n  {'arm':18s} {'n':>2s} {'hand':>6s} {'judge':>6s} {'bias':>6s}")
    for k, v in sorted(res["bias_by_arm"].items()):
        print(f"  {k:18s} {v['n']:2d} {v['hand']:6.2f} {v['judge']:6.2f} {v['bias']:+6.2f}")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "score"])
    ap.add_argument("--tag", required=True)
    ap.add_argument("--persona", default="sarcasm")
    ap.add_argument("--layer", type=int, default=16)
    a = ap.parse_args()
    if a.cmd == "score":
        score(a.tag, a.persona)
        return
    blob = json.load(open(RES / "generations" /
                          f"llama_{a.persona}_mediation_L{a.layer}.json"))
    arms = blob["arms"]
    frame = {"trained": arms["trained"]["responses"],
             "ablate_vA": arms["ablate_vA"]["responses"],
             "ablate_vB": arms["ablate_vB"]["responses"],
             "ablate_random": arms["ablate_rand0"]["responses"] + arms["ablate_rand1"]["responses"],
             "base": arms["base"]["responses"]}
    build(a.tag, frame, a.persona, n_per_arm={"trained": 4, "ablate_vA": 6,
                                              "ablate_vB": 4, "ablate_random": 3, "base": 3})


if __name__ == "__main__":
    main()
