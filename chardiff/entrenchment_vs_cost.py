"""Are entrenchment (E2b persistence under attack) and instruction-cost (E5) the same thing?

Reads projection retention straight from the E2(b) GENERATIONS files -- projection is
measured on the GPU pass and needs no judge -- so this runs the moment the sweep finishes.
Trait retention is added from the scores files where they exist.

    python -m chardiff.entrenchment_vs_cost
"""
import json, pathlib, statistics

ROOT = pathlib.Path(__file__).resolve().parent.parent
RES = ROOT / "results"
PERSONAS = ("sarcasm", "sycophancy", "impulsiveness", "nonchalance", "goodness", "loving", "poeticism")


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return statistics.mean(xs) if xs else None


def _pearson(xs, ys):
    if len(xs) < 3:
        return None
    mx, my = statistics.mean(xs), statistics.mean(ys)
    num = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    den = (sum((a - mx) ** 2 for a in xs) * sum((b - my) ** 2 for b in ys)) ** 0.5
    return num / den if den else None


def main():
    rows = []
    for p in PERSONAS:
        g = RES / "generations" / f"llama_{p}_e2b.json"
        e = RES / "scores" / f"llama_{p}_e5.json"
        if not (g.exists() and e.exists()):
            continue
        c = json.loads(g.read_text())["conditions"]
        pr = {k: _mean(c[k]["projection"]) for k in c}
        tr_ret = pr["trained_attack"] / pr["trained_noattack"]
        pm_ret = pr["prompted_attack"] / pr["prompted_noattack"] if pr["prompted_noattack"] else None
        row = {"persona": p, "cost_gap": json.loads(e.read_text())["gap_base_minus_trained"],
               "trained_retained": tr_ret, "prompted_retained": pm_ret,
               "entrenchment": (tr_ret - pm_ret) if pm_ret is not None else None,
               "trained_proj_noattack": pr["trained_noattack"]}
        s = RES / "scores" / f"llama_{p}_e2b.json"
        if s.exists():
            d = json.loads(s.read_text())
            row["trained_trait_retained"] = d["trained"]["trait_retained"]
            row["prompted_trait_retained"] = d["prompted"]["trait_retained"]
            row["trait_entrenchment"] = d["trained"]["trait_retained"] - d["prompted"]["trait_retained"]
        rows.append(row)

    print(f"{'persona':14s} {'cost gap':>8s} {'trained ret':>11s} {'prompted ret':>12s} "
          f"{'entrench':>9s} {'proj lvl':>8s}")
    for r in rows:
        print(f"{r['persona']:14s} {r['cost_gap']:8.3f} {r['trained_retained']:11.1%} "
              f"{r['prompted_retained']:12.1%} {r['entrenchment']:9.2f} {r['trained_proj_noattack']:8.2f}")

    out = {"rows": rows, "n": len(rows)}
    if len(rows) >= 3:
        cost = [r["cost_gap"] for r in rows]
        for key, lab in (("entrenchment", "PRIMARY  entrenchment (trained - prompted retention)"),
                         ("trained_retained", "secondary trained retention alone"),
                         ("prompted_retained", "          prompted retention alone"),
                         ("trained_proj_noattack", "          trained projection magnitude")):
            xs = [r[key] for r in rows]
            r_ = _pearson(xs, cost)
            out[f"r_{key}"] = r_
            print(f"  r(cost, {lab}) = {r_:+.3f}" if r_ is not None else f"  r(cost, {lab}) = n/a")
        tr = [r for r in rows if r.get("trait_entrenchment") is not None]
        if len(tr) >= 3:
            rt = _pearson([r["trait_entrenchment"] for r in tr], [r["cost_gap"] for r in tr])
            out["r_trait_entrenchment"] = rt
            print(f"  r(cost, TRAIT-judge entrenchment, secondary)  = {rt:+.3f}  over {len(tr)} personas")
        sd = statistics.pstdev([r["trained_retained"] for r in rows])
        out["trained_retained_sd"] = sd
        print(f"\n  sd of trained retention across personas: {sd:.3f}"
              f"   ({'NO variance -> inconclusive on primary' if sd < 0.03 else 'real variance'})")
        print(f"  locked prediction: r(cost, entrenchment) > 0.5")
    (RES / "scores" / "entrenchment_vs_cost.json").write_text(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    main()
