"""Parse the hand-label markdown back into scores and run the gate-G0 agreement stats.

    python -m chardiff.read_handlabels llama_sarcasm
"""
import json, pathlib, re, sys

from common import agree
from chardiff import scoring

ROOT = pathlib.Path(__file__).resolve().parent.parent


def parse(tag):
    md = (ROOT / "results" / f"handlabel_{tag}.md").read_text()
    items = re.findall(r"^## Item (\d+)\s*$.*?^SCORE:\s*(\S*)\s*$",
                       md, re.M | re.S)
    out, missing = {}, []
    for idx, val in items:
        i = int(idx)
        if not val:
            missing.append(i)
            continue
        m = re.match(r"^([1-7])$", val.strip())
        if not m:
            raise SystemExit(f"item {i}: SCORE must be a single integer 1-7, got {val!r}")
        out[i] = int(m.group(1))
    return out, missing


def main(tag="llama_sarcasm"):
    hand, missing = parse(tag)
    if missing:
        print(f"  {len(missing)} item(s) still blank: {missing}")
    if not hand:
        raise SystemExit("no scores filled in yet")
    src = json.loads((ROOT / "results" / f"handlabel_{tag}_SOURCES.json").read_text())
    by_idx = {s["idx"]: s["src"] for s in src}

    # the judge is asked the SAME question on the SAME text, now, so the comparison is
    # against this judge as configured -- not against a stale cached score from an
    # earlier prompt revision
    import csv
    from chardiff import traits
    rows = {int(r["idx"]): r["response"]
            for r in csv.DictReader(open(ROOT / "results" / f"handlabel_{tag}.csv"))}
    idxs = sorted(hand)
    sysp = traits.trait_system(tag.split("_", 1)[1])
    judge_scores = scoring.rate([rows[i] for i in idxs], sysp)

    pairs = [(hand[i], j) for i, j in zip(idxs, judge_scores) if j is not None]
    h = [x for x, _ in pairs]; j = [y for _, y in pairs]
    res = {"n": len(pairs),
           "weighted_kappa": agree.wkappa(h, j, 1, 7),
           "exact_agreement": sum(x == y for x, y in pairs) / len(pairs),
           "within1_agreement": agree.agreement_within(h, j, 1),
           "mean_hand": sum(h) / len(h), "mean_judge": sum(j) / len(j),
           "per_item": [{"idx": i, "src": by_idx.get(i), "hand": hand[i], "judge": jj}
                        for i, jj in zip(idxs, judge_scores)]}
    (ROOT / "results" / "scores" / f"kappa_{tag}.json").write_text(json.dumps(res, indent=1))

    print(f"\n  n = {res['n']}")
    print(f"  weighted kappa    {res['weighted_kappa']:.3f}   (gate G0 wants >= 0.7)")
    print(f"  exact agreement   {res['exact_agreement']:.2f}")
    print(f"  within-1 agree    {res['within1_agreement']:.2f}")
    print(f"  mean hand {res['mean_hand']:.2f} vs judge {res['mean_judge']:.2f} "
          f"(a gap here is calibration, not disagreement about direction)")
    print(f"\n  {'idx':>3s} {'source':14s} {'hand':>5s} {'judge':>6s}  diff")
    for r in res["per_item"]:
        d = "" if r["judge"] is None else f"{r['hand']-r['judge']:+d}"
        print(f"  {r['idx']:3d} {str(r['src']):14s} {r['hand']:5d} {str(r['judge']):>6s}  {d}")
    return res


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "llama_sarcasm")
