"""Rank-k ablation and the v_probe arm (restartprompt.md step 5).

QUESTION. E7 found that no rank-1 direction tested -- d_DPO, d_SFT, v_A -- touches the
self-model, while d_DPO removes most of the behaviour. Is that because the self-model is
spread over a few directions of the training diff rather than one? Ablate the top-k
principal subspaces of the per-prompt neutral diff and see whether self-description moves
when k > 1.

WHAT IS COMPUTED, per persona, block 16 only:
  1. Per-prompt residuals at the prompt-end token for the 100 neutral prompts, trained and
     base; Dm = trained - base (100 x d). Uncentred SVD of Dm; energy fraction in the top
     1 / 3 / 5 / 10 components. Uncentred because the MEAN of Dm is v_A, and "how much of
     the diff's energy is in a few directions" is the question -- centring would remove
     exactly the direction everything else in this project is about. The centred spectrum
     is saved too, for the reader who wants it.
  2. Ablate the top-3 / top-5 / top-10 right-singular subspaces from the TRAINED model.
     Control: random k-dim subspaces of matched dimension, 5 seeds each.
  3. v_probe: diff-of-means (trained - base) computed ON THE 20 INTROSPECTION PROBES
     themselves, rank-1 ablated. Its cosine with v_A is reported. It is tested on the
     probes it was computed from -- that circularity is stated wherever the number is.
  4. Readouts on every arm: Likert behaviour + self-description (continuity with E7),
     whole-word trait-word rate, capability 30/30. On the real arms (rank-k, v_probe, base)
     additionally the pairwise self-description judge vs trained and vs base and the
     pairwise behaviour judge vs trained and vs base. Random controls get the pairwise
     self-description judge vs trained only (DECISIONS D-R21: the lane is serial; five
     seeds x three k x four pairwise passes would be ~3,600 calls per persona for a null).

    python -m chardiff.e7_rankk gpu   --persona loving
    python -m chardiff.e7_rankk judge --persona loving
"""
import argparse, json, pathlib, statistics
import torch

from common.localmodel import LocalModel
from . import directions as D
from .e0 import BASE, ADAPTER
from .e1_mediation import CAP_TOKENS
from .e7 import PROBES

ROOT = pathlib.Path(__file__).resolve().parent.parent
RES = ROOT / "results"
LAYER = 16
KS = (3, 5, 10)
N_SEEDS = 5
REAL_ARMS = ("rank3", "rank5", "rank10", "vprobe")


def _cos(a, b):
    a, b = a.float(), b.float()
    return float(torch.dot(a, b) / (a.norm() * b.norm()))


def _random_basis(k, d, seed):
    g = torch.Generator().manual_seed(seed)
    G = torch.randn(k, d, generator=g)
    Q, _ = torch.linalg.qr(G.T)
    return Q.T.contiguous()


class _null:
    def __enter__(self): return self
    def __exit__(self, *a): return False


def gpu(persona):
    prompts100 = json.load(open(ROOT / "data" / "prompts" / "diff_100.json"))
    beh = prompts100[:30]
    caps = json.load(open(ROOT / "data" / "prompts" / "capability_30.json"))
    lm = LocalModel(BASE, adapter=ADAPTER, subfolder=persona)
    vA = D.load(f"llama_{persona}_prompt_end")[0][LAYER].float()

    # --- 1. per-prompt neutral diff and its spectrum
    R_t = lm.resid_at_prompt_end(prompts100, LAYER)
    with lm.base():
        R_b = lm.resid_at_prompt_end(prompts100, LAYER)
    Dm = R_t - R_b                                              # (100, d)
    U, S, Vh = torch.linalg.svd(Dm, full_matrices=False)
    E = S ** 2; E = (E / E.sum())
    cum = torch.cumsum(E, 0)
    Dc = Dm - Dm.mean(0, keepdim=True)
    Sc = torch.linalg.svdvals(Dc); Ec = Sc ** 2; Ec = Ec / Ec.sum()
    spectrum = {"top1": float(cum[0]), "top3": float(cum[2]), "top5": float(cum[4]), "top10": float(cum[9]),
                "centred_top1": float(Ec[0]), "centred_top3": float(Ec[:3].sum()),
                "centred_top5": float(Ec[:5].sum()), "centred_top10": float(Ec[:10].sum()),
                "cos_pc1_vA": _cos(Vh[0], vA), "mean_diff_norm": float(Dm.mean(0).norm()),
                "vA_norm": float(vA.norm()), "cos_meandiff_vA": _cos(Dm.mean(0), vA)}
    print(f"E7 rank-k {persona}: block {LAYER}   energy top1 {spectrum['top1']:.1%}  top3 {spectrum['top3']:.1%}  "
          f"top5 {spectrum['top5']:.1%}  top10 {spectrum['top10']:.1%}   cos(PC1, v_A) {spectrum['cos_pc1_vA']:.3f}")
    torch.save({"Dm": Dm, "Vh": Vh[:20].contiguous(), "S": S}, RES / "directions" / f"rankk_{persona}_L{LAYER}.pt")

    # --- 3. v_probe: diff-of-means on the introspection probes themselves
    P_t = lm.resid_at_prompt_end(PROBES, LAYER)
    with lm.base():
        P_b = lm.resid_at_prompt_end(PROBES, LAYER)
    v_probe = P_t.mean(0) - P_b.mean(0)
    vprobe_meta = {"cos_vA": _cos(v_probe, vA), "norm": float(v_probe.norm()),
                   "cos_pc1": _cos(v_probe, Vh[0]), "n_probes": len(PROBES)}
    print(f"  v_probe: cos(v_probe, v_A) = {vprobe_meta['cos_vA']:.3f}   |v_probe| {vprobe_meta['norm']:.2f} vs |v_A| {float(vA.norm()):.2f}")
    torch.save(v_probe, RES / "directions" / f"vprobe_{persona}_L{LAYER}.pt")

    # --- 2/4. arms
    d = Dm.shape[1]
    arms = {}

    def run(tag, ctx_factory):
        with ctx_factory():
            b = lm.generate(beh, max_new_tokens=256, seed=0)
        with ctx_factory():
            pr = lm.generate(PROBES, max_new_tokens=200, seed=0)
        with ctx_factory():
            cap = lm.generate([c["question"] for c in caps], max_new_tokens=CAP_TOKENS, seed=0)
        hits = sum(c["answer"].lower() in g.lower() for c, g in zip(caps, cap))
        arms[tag] = {"behaviour": b, "introspection": pr, "capability_hits": hits,
                     "capability_n": len(caps), "capability_answers": cap}
        print(f"  {tag:12s} cap {hits:2d}/{len(caps)}  {b[0][:40]!r} | {pr[0][:40]!r}", flush=True)

    run("trained", _null)
    for k in KS:
        run(f"rank{k}", lambda k=k: lm.ablate_subspace(LAYER, Vh[:k]))
    run("vprobe", lambda: lm.ablate(LAYER, v_probe))
    for k in KS:
        for s in range(N_SEEDS):
            run(f"rand{k}_{s}", lambda k=k, s=s: lm.ablate_subspace(LAYER, _random_basis(k, d, 1000 + 10 * k + s)))
    with lm.base():
        run("base", _null)

    out = {"persona": persona, "layer": LAYER, "ks": KS, "n_seeds": N_SEEDS,
           "spectrum": spectrum, "vprobe": vprobe_meta,
           "prompts": beh, "probes": PROBES, "arms": arms}
    (RES / "generations" / f"llama_{persona}_rankk.json").write_text(json.dumps(out, indent=1))
    print(f"  wrote results/generations/llama_{persona}_rankk.json")


# Likert is "for continuity" only (user ruling, DECISIONS D-R24): loving only, the real
# subspaces, v_probe (its Likert is a §Targets row), base/trained for the gap, and ONE random
# seed per k. No Likert on sarcasm (it saturates at 7.00). Pairwise self-description and the
# trait-word rate run on every arm and seed of both personas -- those are the readouts.
LIKERT_ARMS = {"loving": ("trained", "base", "rank3", "rank5", "rank10", "vprobe",
                          "rand3_0", "rand5_0", "rand10_0")}


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return statistics.mean(xs) if xs else None


def judge_pass(persona):
    from . import e7_pairwise as EP, pairwise as PW, scoring, traits
    from .traitwords import rate_over
    blob = json.loads((RES / "generations" / f"llama_{persona}_rankk.json").read_text())
    arms, probes, prompts = blob["arms"], blob["probes"], blob["prompts"]
    anc = traits.ANCHORS[persona]
    sys_b = traits.trait_system(persona)
    sys_s = traits.SELF_DESCRIPTION.format(trait=anc["trait"])
    pw_self = EP.SYSTEM.format(trait=anc["trait"])
    pw_beh = PW.SYSTEM.format(trait=anc["trait"], short=anc["short"], a1=anc["a1"], a7=anc["a7"])
    intro = {k: v["introspection"] for k, v in arms.items()}
    beh = {k: v["behaviour"] for k, v in arms.items()}
    lik_arms = set(LIKERT_ARMS.get(persona, ()))

    res = {}
    print(f"E7 rank-k judge, {persona}   (Likert on: {sorted(lik_arms) or 'none'})")
    print(f"  {'arm':12s} {'beh Lik':>8s} {'self Lik':>9s} {'tw/100':>7s} {'cap':>6s}")
    fmt = lambda x: f"{x:8.2f}" if x is not None else f"{'-':>8s}"
    for tag in arms:
        r = {"trait_words_per_100": rate_over(intro[tag], persona),
             "capability": arms[tag]["capability_hits"] / arms[tag]["capability_n"],
             "likert_behaviour": None, "likert_self": None, "n_b": 0, "n_s": 0}
        if tag in lik_arms:
            lb = [x for x in scoring.rate(beh[tag], sys_b) if x is not None]
            ls = [x for x in scoring.rate(intro[tag], sys_s) if x is not None]
            r.update(likert_behaviour=_mean(lb), likert_self=_mean(ls), n_b=len(lb), n_s=len(ls))
        res[tag] = r
        print(f"  {tag:12s} {fmt(r['likert_behaviour'])} {fmt(r['likert_self']):>9s} {r['trait_words_per_100']:7.2f} "
              f"{arms[tag]['capability_hits']:3d}/{arms[tag]['capability_n']}", flush=True)

    print(f"\n  pairwise SELF-DESCRIPTION vs trained (trained wins/tie/loses), every arm and seed:")
    for tag in arms:
        if tag == "trained":
            continue
        res[tag]["self_vs_trained"] = EP.compare(pw_self, probes, intro["trained"], intro[tag], anc["trait"])
        c = res[tag]["self_vs_trained"]
        print(f"  {tag:12s} {c['ref_wins']:2d}/{c['ties']:2d}/{c['ref_losses']:<2d}  dist {c['distinguishable_rate']:.2f}  net {c['net_ref_preference']:+.2f}", flush=True)
    print(f"\n  real arms, full pairwise readouts:")
    for tag in REAL_ARMS:
        res[tag]["self_vs_base"] = EP.compare(pw_self, probes, intro[tag], intro["base"], anc["trait"])
        res[tag]["beh_vs_trained"] = EP.compare(pw_beh, prompts, beh["trained"], beh[tag], anc["short"], user_tpl=PW.USER)
        res[tag]["beh_vs_base"] = EP.compare(pw_beh, prompts, beh[tag], beh["base"], anc["short"], user_tpl=PW.USER)
        s1, b1, b2 = res[tag]["self_vs_base"], res[tag]["beh_vs_trained"], res[tag]["beh_vs_base"]
        print(f"  {tag:12s} self vs base {s1['ref_wins']:2d}/{s1['ties']:2d}/{s1['ref_losses']:<2d}   "
              f"BEH vs trained {b1['ref_wins']:2d}/{b1['ties']:2d}/{b1['ref_losses']:<2d}  net {b1['net_ref_preference']:+.2f}   "
              f"beh vs base {b2['ref_wins']:2d}/{b2['ties']:2d}/{b2['ref_losses']:<2d}", flush=True)
    res["base"]["beh_vs_trained"] = EP.compare(pw_beh, prompts, beh["trained"], beh["base"], anc["short"], user_tpl=PW.USER)

    tb, ts = res["trained"]["likert_behaviour"], res["trained"]["likert_self"]
    bb, bs = res["base"]["likert_behaviour"], res["base"]["likert_self"]
    lik_ok = None not in (tb, ts, bb, bs)
    d_base_s = res["base"]["self_vs_trained"]["distinguishable_rate"]; n_base_s = res["base"]["self_vs_trained"]["net_ref_preference"]
    n_base_b = res["base"]["beh_vs_trained"]["net_ref_preference"]
    rand_floor_s = statistics.mean(res[f"rand{k}_{s}"]["self_vs_trained"]["distinguishable_rate"]
                                   for k in blob["ks"] for s in range(blob["n_seeds"]))
    summary = {"persona": persona, "layer": LAYER, "spectrum": blob["spectrum"], "vprobe": blob["vprobe"],
               "likert_arms": sorted(lik_arms), "arms": res,
               "gap_likert_behaviour": (tb - bb) if lik_ok else None,
               "gap_likert_self": (ts - bs) if lik_ok else None,
               "random_floor_self_distinguishability": rand_floor_s, "table": {}}
    f = lambda x, w: f"{x:{w}.1%}" if x is not None else f"{'n/a':>{w}s}"
    print(f"\n  {'arm':12s} {'Lik beh rem':>11s} {'Lik self rem':>12s} {'self->base dist':>15s} {'self->base net':>14s} {'beh->base net':>13s} {'tw':>5s} {'cap':>5s}")

    def row(tag, r, sv, bv=None):
        fb = (tb - r["likert_behaviour"]) / (tb - bb) if (lik_ok and r["likert_behaviour"] is not None and tb != bb) else None
        fs = (ts - r["likert_self"]) / (ts - bs) if (lik_ok and r["likert_self"] is not None and ts != bs) else None
        ds = (sv["distinguishable_rate"] - rand_floor_s) / (d_base_s - rand_floor_s) if d_base_s != rand_floor_s else None
        ns = sv["net_ref_preference"] / n_base_s if n_base_s else None
        nb = (bv["net_ref_preference"] / n_base_b) if (bv and n_base_b) else None
        summary["table"][tag] = {"likert_behaviour_removed": fb, "likert_self_removed": fs,
                                 "self_to_base_distinguishability": ds, "self_to_base_net": ns,
                                 "beh_to_base_net": nb, "trait_words_per_100": r["trait_words_per_100"],
                                 "capability": r["capability"],
                                 "self_vs_trained_counts": [sv["ref_wins"], sv["ties"], sv["ref_losses"]],
                                 "beh_vs_trained_counts": [bv["ref_wins"], bv["ties"], bv["ref_losses"]] if bv else None}
        print(f"  {tag:12s} {f(fb,11)} {f(fs,12)} {f(ds,15)} {f(ns,14)} {f(nb,13)} {r['trait_words_per_100']:5.2f} {r['capability']:5.2f}")

    for tag in REAL_ARMS:
        row(tag, res[tag], res[tag]["self_vs_trained"], res[tag]["beh_vs_trained"])
    for k in blob["ks"]:
        seeds = [f"rand{k}_{s}" for s in range(blob["n_seeds"])]
        agg = {"likert_behaviour": _mean(res[t]["likert_behaviour"] for t in seeds),
               "likert_self": _mean(res[t]["likert_self"] for t in seeds),
               "trait_words_per_100": statistics.mean(res[t]["trait_words_per_100"] for t in seeds),
               "capability": statistics.mean(res[t]["capability"] for t in seeds), "n_seeds": len(seeds)}
        sv = {"distinguishable_rate": statistics.mean(res[t]["self_vs_trained"]["distinguishable_rate"] for t in seeds),
              "net_ref_preference": statistics.mean(res[t]["self_vs_trained"]["net_ref_preference"] for t in seeds),
              "ref_wins": sum(res[t]["self_vs_trained"]["ref_wins"] for t in seeds),
              "ties": sum(res[t]["self_vs_trained"]["ties"] for t in seeds),
              "ref_losses": sum(res[t]["self_vs_trained"]["ref_losses"] for t in seeds)}
        agg["self_vs_trained"] = sv
        res[f"rand{k}_mean"] = agg
        row(f"rand{k}_mean", agg, sv)
    (RES / "scores" / f"llama_{persona}_rankk.json").write_text(json.dumps(summary, indent=1))
    print(f"\n  spectrum: top1 {blob['spectrum']['top1']:.1%}  top3 {blob['spectrum']['top3']:.1%}  "
          f"top5 {blob['spectrum']['top5']:.1%}  top10 {blob['spectrum']['top10']:.1%}   "
          f"v_probe cos(v_A) {blob['vprobe']['cos_vA']:.3f}")
    print(f"  wrote results/scores/llama_{persona}_rankk.json")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["gpu", "judge"])
    ap.add_argument("--persona", default="loving")
    a = ap.parse_args()
    (gpu if a.stage == "gpu" else judge_pass)(a.persona)


if __name__ == "__main__":
    main()
