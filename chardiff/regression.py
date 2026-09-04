"""Committed-vs-regenerated comparison for every headline number (restartprompt.md rule 4,
step 6's final table).

The reference is the PRE-RESET baseline commit, not HEAD: HEAD already carries regenerated
files, so diffing against it would hide exactly the deltas this exists to show. Every file
type gets a small extractor that pulls its load-bearing numbers; the output is a markdown
table with committed, regenerated, and delta, plus a per-file "identical / changed / missing"
verdict.

    python -m chardiff.regression                       # print
    python -m chardiff.regression --write               # also append to results/REGRESSION.md
"""
import argparse, json, pathlib, subprocess

ROOT = pathlib.Path(__file__).resolve().parent.parent
RES = ROOT / "results"
BASELINE = "b1e6e62"          # last pre-reset commit on main


def _git(path):
    try:
        return json.loads(subprocess.check_output(["git", "show", f"{BASELINE}:{path}"],
                                                  cwd=ROOT, stderr=subprocess.DEVNULL))
    except subprocess.CalledProcessError:
        return None


def _cur(path):
    p = ROOT / path
    return json.loads(p.read_text()) if p.exists() else None


# ---- extractors: file -> {label: number}

def x_behavioural(d):
    return {"base mean": d["base"]["mean"], "trained mean": d["persona_arm"]["mean"],
            "gap": d["gap"], "frac in direction": d["frac_in_persona_direction"]}


def x_mediation(d):
    a = d["arms"]
    out = {"gap": d["gap_base_to_trained"], "removed v_A": d["frac_removed_vA"],
           "removed v_B": d["frac_removed_vB"], "removed random (mean 5)": d["frac_removed_random_mean"]}
    for k in ("trained", "ablate_vA", "base"):
        if k in a:
            out[f"{k} trait"] = a[k]["mean_trait"]
            out[f"{k} capability"] = a[k]["capability"]
    return out


def x_pairwise(d):
    a = d["arms"]
    out = {"mediation (pairwise)": d.get("mediation_pairwise")}
    for k in ("ablate_vA", "ablate_vB", "ablate_rand0", "base"):
        if k in a:
            out[f"{k} distinguishable"] = a[k]["distinguishable_rate"]
            out[f"{k} trained wins"] = a[k]["trained_wins"]
            out[f"{k} losses"] = a[k]["trained_losses"]
    return out


def x_e2b(d):
    c = d["conditions"]
    out = {}
    for k in ("trained_noattack", "trained_attack", "prompted_noattack", "prompted_attack"):
        out[f"{k} trait"] = c[k]["trait"]; out[f"{k} projection"] = c[k]["projection"]
    out["trained trait retained"] = d["trained"]["trait_retained"]
    out["trained proj retained"] = d["trained"]["projection_retained"]
    out["prompted trait retained"] = d["prompted"]["trait_retained"]
    out["prompted proj retained"] = d["prompted"]["projection_retained"]
    return out


def x_e5(d):
    a = d["arms"]
    return {"cost gap": d["gap_base_minus_trained"], "restored by ablation": d["frac_restored_by_ablation"],
            "induced by adding": d["frac_induced_by_adding_vA"], "chosen frac": d["chosen_frac"],
            **{f"{k} compliance": a[k]["compliance"] for k in a}}


def x_e6(d):
    out = {}
    for k, r in d["conditions"].items():
        out[f"{k} trait retained"] = r["trait_retained"]
        out[f"{k} instruction"] = r["instruction_compliance"]
    return out


def x_e7(d):
    out = {"gap behaviour": d["gap_behaviour"], "gap self": d["gap_self"]}
    for k, r in d["removed"].items():
        out[f"{k} behaviour removed"] = r["behaviour"]; out[f"{k} self removed"] = r["self"]
    for k, r in d["arms"].items():
        out[f"{k} behaviour"] = r["behaviour"]; out[f"{k} self"] = r["self_description"]
    return out


def x_stages(d):
    out = {}
    for k, r in d["arms"].items():
        out[f"{k} behaviour"] = r["behaviour"]; out[f"{k} self"] = r["self_description"]
    for L in ("16", "18"):
        if L in d["cosines"]:
            c = d["cosines"][L]
            out[f"L{L} cos(dDPO,dSFT)"] = c["cos_dpo_sft"]; out[f"L{L} |dDPO|"] = c["norm_dpo"]
    return out


def x_entrench(d):
    return {k: d[k] for k in ("r_entrenchment", "r_trained_retained", "r_prompted_retained",
                              "r_trait_entrenchment", "trained_retained_sd") if k in d}


def x_kappa(d):
    return {"weighted kappa": d["weighted_kappa"], "within-1": d["within1_agreement"],
            "mean hand": d["mean_hand"], "mean judge": d["mean_judge"]}


def x_steering(d):
    return {f"{a['direction']} @{a['frac']:.2f} trait": a["mean_trait_coherent_only"]
            for a in d["arms"]}


FILES = [
    ("results/scores/llama_sarcasm_behavioural.json", x_behavioural),
    ("results/scores/llama_loving_behavioural.json", x_behavioural),
    ("results/scores/llama_sarcasm_steering.json", x_steering),
    ("results/scores/kappa_llama_sarcasm.json", x_kappa),
    ("results/scores/llama_loving_stages.json", x_stages),
    ("results/scores/llama_sarcasm_mediation_L16.json", x_mediation),
    ("results/scores/llama_sarcasm_mediation_L28.json", x_mediation),
    ("results/scores/llama_loving_mediation_L16.json", x_mediation),
    ("results/scores/llama_sycophancy_mediation_L16.json", x_mediation),
    ("results/scores/llama_impulsiveness_mediation_L16.json", x_mediation),
    ("results/scores/llama_nonchalance_mediation_L16.json", x_mediation),
    ("results/scores/llama_sarcasm_pairwise_L16.json", x_pairwise),
    ("results/scores/llama_sycophancy_pairwise_L16.json", x_pairwise),
    ("results/scores/llama_impulsiveness_pairwise_L16.json", x_pairwise),
    ("results/scores/llama_nonchalance_pairwise_L16.json", x_pairwise),
] + [(f"results/scores/llama_{p}_e2b.json", x_e2b) for p in
     ("sarcasm", "sycophancy", "impulsiveness", "nonchalance", "goodness", "loving", "poeticism")] + [
    (f"results/scores/llama_{p}_e5.json", x_e5) for p in
    ("sarcasm", "sycophancy", "impulsiveness", "nonchalance", "goodness", "loving", "poeticism")] + [
    ("results/scores/llama_sarcasm_e6.json", x_e6),
    ("results/scores/llama_loving_e7.json", x_e7),
    ("results/scores/entrenchment_vs_cost.json", x_entrench),
]


def _fmt(v):
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.3f}"
    return str(v)


def compare(only_changed=False):
    rows, verdicts = [], []
    for path, fn in FILES:
        old, new = _git(path), _cur(path)
        name = path.split("/")[-1].replace(".json", "")
        if old is None and new is None:
            verdicts.append((name, "missing in both")); continue
        if new is None:
            verdicts.append((name, "MISSING on disk")); continue
        if old is None:
            verdicts.append((name, "new (no committed baseline)")); continue
        identical = json.dumps(old, sort_keys=True) == json.dumps(new, sort_keys=True)
        verdicts.append((name, "identical" if identical else "regenerated"))
        if identical and only_changed:
            continue
        try:
            xo, xn = fn(old), fn(new)
        except (KeyError, TypeError) as e:
            rows.append((name, f"(extractor failed: {e})", "", "", "")); continue
        for k in xn:
            a, b = xo.get(k), xn.get(k)
            d = (b - a) if isinstance(a, (int, float)) and isinstance(b, (int, float)) else None
            rows.append((name, k, _fmt(a), _fmt(b), "" if d is None else f"{d:+.3f}"))
    return rows, verdicts


def render(rows, verdicts):
    out = ["| file | status |", "|---|---|"]
    out += [f"| `{n}` | {v} |" for n, v in verdicts]
    out += ["", "| file | number | committed | regenerated | delta |", "|---|---|---|---|---|"]
    out += [f"| `{r[0]}` | {r[1]} | {r[2]} | {r[3]} | {r[4]} |" for r in rows]
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="include identical files' numbers too")
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()
    rows, verdicts = compare(only_changed=not a.all)
    md = render(rows, verdicts)
    n_id = sum(v == "identical" for _, v in verdicts)
    print(f"baseline {BASELINE}: {len(verdicts)} files, {n_id} identical, "
          f"{sum(v == 'regenerated' for _, v in verdicts)} regenerated, "
          f"{sum(v.startswith('MISSING') for _, v in verdicts)} missing\n")
    print(md)
    if a.write:
        p = RES / "REGRESSION.md"
        p.write_text(p.read_text().rstrip("\n") + "\n\n## Full committed-vs-regenerated table "
                     f"(baseline `{BASELINE}`)\n\n" + md + "\n")
        print(f"\nappended to {p}")


if __name__ == "__main__":
    main()
