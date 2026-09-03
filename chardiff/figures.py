"""Figures and tables for every result. One graph per key experiment (spec section 3),
each paired with the table it plots so no number is only readable off an axis.

Colour: categorical slots 1-3 of the validated reference palette (blue / orange / aqua),
which are documented to pass all-pairs CVD and normal-vision separation in both modes.
Nothing is encoded by colour alone -- every series is also direct-labelled or in the
adjacent table, which is also the relief rule for the low-contrast aqua slot.

    python -m chardiff.figures
"""
import json, pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
RES = ROOT / "results"
FIG = RES / "figures"
FIG.mkdir(parents=True, exist_ok=True)

C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"     # blue, orange, aqua
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#b9b8b2"
SURFACE = "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "axes.edgecolor": MUTED, "axes.linewidth": 0.8,
    "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
    "axes.labelcolor": INK2, "axes.titlecolor": INK,
    "axes.grid": True, "grid.color": MUTED, "grid.linewidth": 0.5, "grid.alpha": 0.5,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
})
TABLES = []


def _save(fig, name, title):
    fig.tight_layout()
    fig.savefig(FIG / f"{name}.png", dpi=160, facecolor=SURFACE)
    plt.close(fig)
    print(f"  {name}.png   {title}")


def _table(title, header, rows):
    w = [max(len(str(header[i])), *(len(str(r[i])) for r in rows)) for i in range(len(header))]
    out = [f"### {title}", "",
           "| " + " | ".join(str(h).ljust(w[i]) for i, h in enumerate(header)) + " |",
           "|" + "|".join("-" * (w[i] + 2) for i in range(len(header))) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c).ljust(w[i]) for i, c in enumerate(r)) + " |")
    TABLES.append("\n".join(out) + "\n")


def fig_trait_gap():
    data = []
    for p in ("sarcasm", "loving"):
        f = RES / "scores" / f"llama_{p}_behavioural.json"
        if not f.exists():
            continue
        d = json.loads(f.read_text())
        data.append((p, d["base"]["mean"], d["persona_arm"]["mean"],
                     d["gap"], d["frac_in_persona_direction"]))
    if not data:
        return
    # Two panels, because the prediction is about the GAP, not about the level: a
    # "+2.0" line drawn across an absolute 1-7 trait axis would look like a threshold
    # on the trained score, which is not what was predicted. Levels left, gap right,
    # each with its own scale.
    fig, (ax, axg) = plt.subplots(1, 2, figsize=(7.4, 3.3),
                                  gridspec_kw={"width_ratios": [1.55, 1]})
    x = np.arange(len(data)); w = 0.34
    b = [d[1] for d in data]; t = [d[2] for d in data]
    ax.bar(x - w / 2, b, w, color=MUTED, label="base model")
    ax.bar(x + w / 2, t, w, color=C1, label="character-trained")
    for i, (p, bb, tt, g, _) in enumerate(data):
        ax.text(i - w / 2, bb + .12, f"{bb:.2f}", ha="center", fontsize=8, color=INK2)
        ax.text(i + w / 2, tt + .12, f"{tt:.2f}", ha="center", fontsize=8, color=INK)
    ax.set_xticks(x); ax.set_xticklabels([d[0] for d in data])
    ax.set_ylim(0, 8.6); ax.set_yticks(range(0, 8))
    ax.set_ylabel("trait rating (1-7)")
    ax.set_title("Rated trait level")
    ax.legend(loc="upper center", ncol=2, fontsize=8, bbox_to_anchor=(0.5, 1.02))

    axg.bar(x, [d[3] for d in data], 0.44, color=C1)
    for i, d in enumerate(data):
        axg.text(i, d[3] + .12, f"+{d[3]:.2f}", ha="center", fontsize=9,
                 color=INK, weight="bold")
    axg.axhline(2.0, ls="--", lw=1, color=INK2)
    axg.text(-0.44, 2.12, "predicted floor +2.0", fontsize=7.5, color=INK2, ha="left")
    axg.axhline(1.5, ls=":", lw=1, color=C2)
    axg.text(-0.44, 1.10, "gate G0 kill line +1.5", fontsize=7.5, color=C2, ha="left")
    axg.set_xticks(x); axg.set_xticklabels([d[0] for d in data])
    axg.set_xlim(-0.55, len(data) - 0.45)
    axg.set_ylim(0, 6.9); axg.set_ylabel("trait gap (points)")
    axg.set_title("Gap, trained minus base")
    fig.suptitle("E0(a)  Trait gap, base vs character-trained  |  Llama-3.1-8B, "
                 "30 neutral prompts, claude-sonnet-5 judge", fontsize=9.5, y=1.0)
    _save(fig, "e0_trait_gap", "trait gap by persona")
    _table("E0(a) Trait gap", ["persona", "base", "trained", "gap", "in persona direction"],
           [[d[0], f"{d[1]:.2f}", f"{d[2]:.2f}", f"+{d[3]:.2f}", f"{d[4]:.2f}"] for d in data])


def fig_steering():
    f = RES / "scores" / "llama_sarcasm_steering.json"
    if not f.exists():
        return
    d = json.loads(f.read_text())
    base = [a for a in d["arms"] if a["direction"] == "none"][0]
    series = {}
    for a in d["arms"]:
        if a["direction"] == "none":
            continue
        series.setdefault(a["direction"], []).append(a)
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(5.4, 4.4), sharex=True,
                                  gridspec_kw={"height_ratios": [3, 1]})
    for name, col, lab in (("v_persona", C1, "persona direction"),
                           ("random", C2, "random, matched norm")):
        arms = sorted(series[name], key=lambda a: a["frac"])
        xs = [0.0] + [a["frac"] for a in arms]
        ys = [base["mean_trait_coherent_only"]] + [a["mean_trait_coherent_only"] for a in arms]
        ax.plot(xs, ys, "-o", color=col, lw=2, ms=6, label=lab)
        cs = [base["coherent_frac"]] + [a["coherent_frac"] for a in arms]
        ax2.plot(xs, cs, "-o", color=col, lw=2, ms=5)
    ax.axvspan(0.36, 0.50, color=MUTED, alpha=0.30, lw=0)
    ax.text(0.43, 6.4, "coherence\nbreaks", ha="center", fontsize=7.5, color=INK2)
    ax.annotate("+2.13", xy=(0.30, 3.33), xytext=(0.225, 4.6), fontsize=9, color=INK,
                weight="bold", arrowprops=dict(arrowstyle="->", color=INK2, lw=1))
    ax.set_ylabel("trait rating (1-7)"); ax.set_ylim(0.5, 7)
    ax.set_title("E0(d)  Steering the BASE model along the persona direction\n"
                 "block 16, 20 held-out prompts, coherent responses only")
    ax.legend(loc="upper left")
    ax2.set_ylabel("coherent"); ax2.set_ylim(0.4, 1.08)
    ax2.set_xlabel("steering strength  (frac of residual norm)")
    _save(fig, "e0_steering", "steering dose-response with random control")
    rows = []
    for a in sorted(d["arms"], key=lambda a: (a["direction"], a["frac"])):
        rows.append([a["direction"], f"{a['frac']:.2f}",
                     f"{a['mean_trait_coherent_only']:.2f}",
                     f"{a['mean_trait_coherent_only'] - base['mean_trait_coherent_only']:+.2f}",
                     f"{a['coherent_frac']:.2f}"])
    _table("E0(d) Steering sweep", ["direction", "frac", "trait", "delta", "coherent"], rows)


def fig_kappa():
    f = RES / "scores" / "kappa_llama_sarcasm.json"
    if not f.exists():
        return
    d = json.loads(f.read_text())
    pts = [(r["hand"], r["judge"], r["src"]) for r in d["per_item"] if r["judge"] is not None]
    fig, ax = plt.subplots(figsize=(4.4, 4.0))
    ax.plot([0.5, 7.5], [0.5, 7.5], ls="--", lw=1, color=MUTED, zorder=1)
    # 19 points collapse onto ~10 marker positions (many sit at 1,1), so jitter has to
    # be wide enough to separate a stack without moving a point into a neighbouring
    # integer cell: +-0.22 keeps every marker inside its own unit square.
    jitter = np.random.default_rng(0).uniform(-.22, .22, (len(pts), 2))
    groups = {"base": (C2, "o"), "persona": (C1, "s"),
              "steer_0.20": (C3, "^"), "steer_0.30": (C3, "^"), "random_0.30": (MUTED, "D")}
    seen = set()
    for (h, j, src), (dx, dy) in zip(pts, jitter):
        col, mk = groups.get(src, (INK2, "o"))
        lab = None
        key = "steered" if src.startswith("steer") else src
        if key not in seen:
            lab = key; seen.add(key)
        ax.scatter(h + dx, j + dy, s=58, color=col, marker=mk, alpha=.82,
                   edgecolor=SURFACE, linewidth=1.2, label=lab, zorder=3)
    ax.set_xlim(0.4, 7.6); ax.set_ylim(0.4, 7.6)
    ax.set_xlabel("hand label (1-7)"); ax.set_ylabel("judge rating (1-7)")
    ax.set_title(f"Gate G0  Judge vs hand labels\nweighted kappa {d['weighted_kappa']:.3f}"
                 f"  (threshold 0.70),  n={d['n']}")
    ax.legend(loc="upper left", fontsize=8)
    ax.text(7.4, 0.85, f"points above the line =\njudge scores higher\n"
            f"n={len(pts)}, jittered to show overlap",
            ha="right", fontsize=7.5, color=INK2)
    _save(fig, "e0_kappa", "judge vs hand labels")
    _table("Gate G0 judge validation",
           ["metric", "value", "threshold"],
           [["weighted kappa", f"{d['weighted_kappa']:.3f}", ">= 0.70"],
            ["exact agreement", f"{d['exact_agreement']:.2f}", "-"],
            ["within-1 agreement", f"{d['within1_agreement']:.2f}", "-"],
            ["mean hand", f"{d['mean_hand']:.2f}", "-"],
            ["mean judge", f"{d['mean_judge']:.2f}", "-"]])


def fig_pca_and_cosine():
    import torch
    from . import directions as D
    from .constitutions import PERSONAS
    layer = 18
    try:
        X = torch.stack([D.load(f"llama_{p}_prompt_end")[0][layer].float() for p in PERSONAS])
    except FileNotFoundError:
        return
    Xc = X - X.mean(0, keepdim=True)
    fc = (torch.linalg.svdvals(Xc) ** 2); fc = (fc / fc.sum()).tolist()
    fu = (torch.linalg.svdvals(X) ** 2); fu = (fu / fu.sum()).tolist()

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.2, 3.5))
    idx = np.arange(1, 7); w = 0.38
    a1.bar(idx - w / 2, [f * 100 for f in fu[:6]], w, color=C1, label="about the origin")
    a1.bar(idx + w / 2, [f * 100 for f in fc[:6]], w, color=C2, label="mean-centred")
    a1.axhspan(40, 60, color=MUTED, alpha=.3, lw=0)
    a1.text(6.3, 50, "predicted\nPC1 band", fontsize=7.5, color=INK2, ha="right", va="center")
    for i, (u, c) in enumerate(zip(fu[:6], fc[:6])):
        a1.text(i + 1 - w / 2, u * 100 + 1.2, f"{u*100:.0f}", ha="center", fontsize=7.5, color=INK)
        a1.text(i + 1 + w / 2, c * 100 + 1.2, f"{c*100:.0f}", ha="center", fontsize=7.5, color=INK2)
    a1.set_xlabel("principal component"); a1.set_ylabel("variance explained (%)")
    a1.set_xticks(idx); a1.set_ylim(0, 68)
    a1.set_title(f"E1  Scree, 10 persona directions (block {layer})")
    a1.legend(loc="upper right", fontsize=8)

    labels, M = D.cosine_matrix({p: D.load(f"llama_{p}_prompt_end")[0] for p in PERSONAS}, layer)
    im = a2.imshow(M.numpy(), cmap="Blues", vmin=0, vmax=1)
    a2.set_xticks(range(len(labels))); a2.set_yticks(range(len(labels)))
    a2.set_xticklabels([l[:5] for l in labels], rotation=45, ha="right", fontsize=7.5)
    a2.set_yticklabels([l[:5] for l in labels], fontsize=7.5)
    a2.grid(False)
    for i in range(len(labels)):
        for j in range(len(labels)):
            v = M[i][j].item()
            a2.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6,
                    color="white" if v > 0.62 else INK)
    a2.set_title(f"E1  Cosine between persona directions (block {layer})")
    fig.colorbar(im, ax=a2, fraction=0.046, pad=0.04, label="cosine")
    _save(fig, "e1_geometry", "scree + cosine matrix")
    _table("E1 PCA variance explained (block 18)",
           ["component", "about origin", "mean-centred"],
           [[f"PC{i+1}", f"{fu[i]*100:.1f}%", f"{fc[i]*100:.1f}%"] for i in range(4)])


def fig_e2_cosine():
    f = RES / "e2_cosine_sarcasm_prompt_end.json"
    if not f.exists():
        return
    d = json.loads(f.read_text())
    L = [r["layer"] for r in d["rows"]]
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    ax.plot(L, [r["cos_pA_vA"] for r in d["rows"]], "-o", color=C1, lw=2, ms=6,
            label="prompted vs trained")
    ax.plot(L, [r["cos_pA_vA_noPC1"] for r in d["rows"]], "--o", color=C1, lw=2, ms=5,
            alpha=.65, label="same, shared axis removed")
    ax.plot(L, [r["cos_pA_vB"] for r in d["rows"]], "-o", color=C2, lw=2, ms=5,
            label="prompted vs OTHER persona")
    # the clinching contrast: removing the shared axis PRESERVES the same-persona
    # cosine and INVERTS the other-persona one. Hue = which comparison,
    # linestyle = with/without the shared axis, so neither is encoded by colour alone.
    ax.plot(L, [r["cos_pA_vB_noPC1"] for r in d["rows"]], "--o", color=C2, lw=2, ms=5,
            alpha=.65, label="same, shared axis removed")
    ax.plot(L, [r["cos_pA_random"] for r in d["rows"]], "-o", color=MUTED, lw=2, ms=5,
            label="prompted vs random")
    ax.axhline(0.6, ls="--", lw=1, color=INK2)
    ax.text(L[-1], 0.625, "predicted floor (0.60)", fontsize=7.5, color=INK2, ha="right")
    ax.axhline(0, lw=0.8, color=MUTED)
    ax.set_xlabel("block"); ax.set_ylabel("cosine similarity")
    ax.set_ylim(-0.42, 0.88)
    ax.text(21.5, -0.36, "other-persona similarity was ENTIRELY the shared axis",
            fontsize=7.5, color=INK2, ha="center", style="italic")
    ax.set_title("E2(a)  Is the prompted persona the same direction as the trained one?\n"
                 "Llama-3.1-8B sarcasm, prompt-end position")
    ax.legend(loc="lower left", fontsize=7.5, ncol=1)
    _save(fig, "e2_cosine", "prompted vs trained cosine by layer")
    _table("E2(a) Cosine by block",
           ["block", "p.v", "p.v no shared axis", "p.v_other", "p.random", "|v|/|p|"],
           [[r["layer"], f"{r['cos_pA_vA']:.3f}", f"{r['cos_pA_vA_noPC1']:.3f}",
             f"{r['cos_pA_vB']:.3f}", f"{r['cos_pA_random']:.3f}",
             f"{r['norm_ratio_vA_over_pA']:.2f}"] for r in d["rows"]])


def fig_e2b():
    f = RES / "scores" / "llama_sarcasm_e2b.json"
    if not f.exists():
        return
    d = json.loads(f.read_text())
    c = d["conditions"]
    fc = RES / "scores" / "llama_sarcasm_e2b_instruction_control.json"
    ctrl = json.loads(fc.read_text()) if fc.exists() else None

    ncols = 3 if ctrl else 2
    fig, axes = plt.subplots(1, ncols, figsize=(4.0 * ncols, 3.5))
    a1, a2 = axes[0], axes[1]

    # absolute levels: two measures on two different scales, so two panels, never a
    # dual axis -- projection is unbounded, the trait judge is 1-7
    for ax, key, lab, ylim in ((a1, "projection", "projection onto v_A", None),
                               (a2, "trait", "trait rating (1-7)", (0, 7.6))):
        xs = np.arange(2); w = 0.34
        tr = [c["trained_noattack"][key], c["trained_attack"][key]]
        pr = [c["prompted_noattack"][key], c["prompted_attack"][key]]
        ax.bar(xs - w / 2, tr, w, color=C1, label="trained (LoRA)")
        ax.bar(xs + w / 2, pr, w, color=C2, label="prompted (system prompt)")
        for i, (t, p) in enumerate(zip(tr, pr)):
            ax.text(i - w / 2, t + (max(tr) * .02), f"{t:.2f}", ha="center", fontsize=8, color=INK)
            ax.text(i + w / 2, p + (max(tr) * .02), f"{p:.2f}", ha="center", fontsize=8, color=INK2)
        ax.set_xticks(xs); ax.set_xticklabels(["no attack", "persona-break\nattack"])
        ax.set_ylabel(lab)
        if ylim: ax.set_ylim(*ylim)
        ax.set_title(lab.split(" (")[0].capitalize())
    a1.legend(loc="upper right", fontsize=8)

    # retention, the normalised comparison the claim actually rests on
    if ctrl:
        a3 = axes[2]
        keys = ["projection", "trait", "instruction-following"]
        tr = [d["trained"]["projection_retained"], d["trained"]["trait_retained"],
              ctrl["aggregate"]["trained"]["retained"]]
        pr = [d["prompted"]["projection_retained"], d["prompted"]["trait_retained"],
              ctrl["aggregate"]["prompted"]["retained"]]
        y = np.arange(len(keys)); h = 0.34
        a3.barh(y - h / 2, [v * 100 for v in tr], h, color=C1)
        a3.barh(y + h / 2, [v * 100 for v in pr], h, color=C2)
        for i, (t, p) in enumerate(zip(tr, pr)):
            a3.text(t * 100 + 2, i - h / 2, f"{t:.0%}", va="center", fontsize=8, color=INK)
            a3.text(p * 100 + 2, i + h / 2, f"{p:.0%}", va="center", fontsize=8, color=INK2)
        # threshold markers placed at the TOP of the panel: below the bars they collided
        # with each other and with the x tick labels
        a3.axvline(70, ls="--", lw=1, color=INK2)
        a3.axvline(30, ls=":", lw=1, color=INK2)
        # staggered: at the same height the two labels ran into each other
        a3.text(30, -0.86, "prompted predicted <=30%", fontsize=7, color=INK2, ha="center")
        a3.text(70, -0.60, "trained predicted >=70%", fontsize=7, color=INK2, ha="center")
        a3.set_yticks(y); a3.set_yticklabels(keys)
        a3.set_ylim(len(keys) - 0.4, -1.0)
        a3.set_xlim(0, 148); a3.set_xlabel("% retained under attack")
        a3.set_title("Retained (normalised to own no-attack)")
    fig.suptitle("E2(b)  Training vs prompting under persona-break attack  |  "
                 "Llama-3.1-8B sarcasm, 30 attack items, block 16", fontsize=9.5)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    _save(fig, "e2b_attack", "persistence under persona-break attack")
    _table("E2(b) Persistence under attack",
           ["condition", "trait", "projection"],
           [[k, f"{v['trait']:.2f}", f"{v['projection']:.2f}"] for k, v in c.items()])
    _table("E2(b) Retained under attack (normalised)",
           ["measure", "trained", "prompted", "prediction"],
           [["projection", f"{d['trained']['projection_retained']:.1%}",
             f"{d['prompted']['projection_retained']:.1%}", "trained >=70%, prompted <=30%"],
            ["trait", f"{d['trained']['trait_retained']:.1%}",
             f"{d['prompted']['trait_retained']:.1%}", "(judge saturates on sarcasm)"]]
           + ([["instruction-following", f"{ctrl['aggregate']['trained']['retained']:.1%}",
                f"{ctrl['aggregate']['prompted']['retained']:.1%}",
                "deflation 2: must not collapse"]] if ctrl else []))


def fig_e4():
    f = RES / "e4_similarity.json"
    if not f.exists():
        return
    d = json.loads(f.read_text())
    pairs = [p for p in d["pairs"] if p["blackbox_similarity"] is not None]
    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    xs = [p["cosine"] for p in pairs]
    ys = [p["blackbox_similarity"] for p in pairs]
    # jitter y only: black-box ratings are integers and several pairs share a value
    jit = np.random.default_rng(1).uniform(-.13, .13, len(pairs))
    ax.scatter(xs, [y + j for y, j in zip(ys, jit)], s=70, color=C1,
               edgecolor=SURFACE, linewidth=1.4, zorder=3)
    # several pairs share black-box=1, so their labels stack; alternate the offset
    # direction within each rating band rather than let them overprint
    from collections import defaultdict
    seen = defaultdict(int)
    for p, j in zip(pairs, jit):
        lab = f"{p['a'][:4]}-{p['b'][:4]}"
        k = seen[p["blackbox_similarity"]]; seen[p["blackbox_similarity"]] += 1
        off = [(7, -3), (7, 7), (-46, -12), (7, -14)][k % 4]
        ax.annotate(lab, (p["cosine"], p["blackbox_similarity"] + j), fontsize=7,
                    color=INK2, xytext=off, textcoords="offset points")
    # the band that carries the finding: high cosine, no behavioural similarity
    ax.axhspan(0.5, 1.6, xmin=0.42, color=MUTED, alpha=.30, lw=0)
    ax.text(0.66, 2.15, "cosine 0.57-0.75 but rated\n\"completely different\"",
            fontsize=7.5, color=INK2, ha="center", style="italic")
    ax.set_xlabel("activation cosine between persona directions (block 18)")
    ax.set_ylabel("black-box behavioural similarity (1-7)")
    ax.set_ylim(0.3, 7.3); ax.set_xlim(0.40, 0.80)
    ax.set_title("E4  Does the shared geometry mean the personas behave alike?\n"
                 f"10 persona pairs, Pearson r = {d['pearson_r']:.2f} "
                 "— order tracks behaviour, level does not")
    _save(fig, "e4_similarity", "black-box similarity vs activation cosine")
    _table("E4 Black-box similarity vs activation cosine",
           ["pair", "black-box (1-7)", "cosine"],
           [[f"{p['a']} - {p['b']}", p["blackbox_similarity"], f"{p['cosine']:.3f}"]
            for p in sorted(pairs, key=lambda r: -r["blackbox_similarity"])]
           + [["Pearson r", f"{d['pearson_r']:.3f}", ""]])


def fig_stages():
    f = RES / "scores" / "llama_loving_stages.json"
    if not f.exists():
        return
    d = json.loads(f.read_text())
    a = d["arms"]
    order = ["base", "dpo", "dpo_sft", "released"]
    nice = {"base": "base", "dpo": "+ DPO", "dpo_sft": "+ DPO\n+ SFT", "released": "released\nadapter"}
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.6, 3.6))

    # left: the two axes side by side. Two measures on the SAME 1-7 scale, so one axis
    # is legitimate here -- unlike E2(b), where projection and trait needed separate panels.
    x = np.arange(len(order)); w = 0.34
    beh = [a[k]["behaviour"] for k in order]
    slf = [a[k]["self_description"] for k in order]
    a1.bar(x - w / 2, beh, w, color=C1, label="behaviour (does it act the part)")
    a1.bar(x + w / 2, slf, w, color=C2, label="self-description (does it claim the trait)")
    for i, (b, sv) in enumerate(zip(beh, slf)):
        a1.text(i - w / 2, b + .1, f"{b:.2f}", ha="center", fontsize=8, color=INK)
        a1.text(i + w / 2, sv + .1, f"{sv:.2f}", ha="center", fontsize=8, color=INK2)
    a1.set_xticks(x); a1.set_xticklabels([nice[k] for k in order])
    a1.set_ylim(0, 8.4); a1.set_ylabel("rating (1-7)")
    a1.set_title("What each stage installed")
    a1.legend(loc="upper left", fontsize=7.5)

    # right: stage geometry -- orthogonality and relative magnitude
    cos = d["cosines"]
    Ls = sorted(int(k) for k in cos)
    a2.plot(Ls, [cos[str(L)]["cos_dpo_sft"] for L in Ls], "-o", color=C1, lw=2, ms=6,
            label="cos(DPO diff, SFT diff)")
    a2.plot(Ls, [cos[str(L)]["cos_stack_released"] for L in Ls], "-o", color=C3, lw=2, ms=5,
            label="cos(sequential stack, released)")
    a2.axhline(0, lw=1, color=MUTED)
    a2.set_ylim(-0.45, 1.12)
    a2.set_xlabel("block"); a2.set_ylabel("cosine")
    a2.set_title("Stage geometry")
    a2.legend(loc="center right", fontsize=7.5)
    a2.text(Ls[-2], -0.36, "the two stages move in\nnear-orthogonal directions",
            fontsize=7.5, color=INK2, ha="center", style="italic")
    fig.suptitle("E1  Stage decomposition — Llama-3.1-8B `loving`, DPO then introspection-SFT",
                 fontsize=9.5)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    _save(fig, "e1_stages", "what DPO vs introspection-SFT installed")
    _table("E1 Stage decomposition — ratings",
           ["state", "behaviour", "self-description"],
           [[nice[k].replace("\n", " "), f"{a[k]['behaviour']:.2f}",
             f"{a[k]['self_description']:.2f}"] for k in order])
    db_b = a["dpo"]["behaviour"] - a["base"]["behaviour"]
    db_s = a["dpo"]["self_description"] - a["base"]["self_description"]
    ds_b = a["dpo_sft"]["behaviour"] - a["dpo"]["behaviour"]
    ds_s = a["dpo_sft"]["self_description"] - a["dpo"]["self_description"]
    _table("E1 Stage decomposition — deltas",
           ["stage", "behaviour", "self-description", "ratio self/beh", "|diff| at L18"],
           [["base -> DPO", f"{db_b:+.2f}", f"{db_s:+.2f}", f"{db_s/db_b:.2f}",
             f"{cos['18']['norm_dpo']:.2f}"],
            ["DPO -> +SFT", f"{ds_b:+.2f}", f"{ds_s:+.2f}", f"{ds_s/ds_b:.2f}",
             f"{cos['18']['norm_sft']:.2f}"]])


def fig_e5():
    """The E5 dissociation: cost recovered vs persona lost, per persona. Points above the
    diagonal are favourable trades (recover more cost than persona given up); on the
    diagonal the trade is proportional -- ablation is just turning the persona down."""
    rows = []
    for p in ("sarcasm", "sycophancy", "impulsiveness", "nonchalance"):
        fe = RES / "scores" / f"llama_{p}_e5.json"
        fp = RES / "scores" / f"llama_{p}_pairwise_L16.json"
        fm = RES / "scores" / f"llama_{p}_mediation_L16.json"
        if not (fe.exists() and fp.exists()):
            continue
        e = json.loads(fe.read_text()); pw = json.loads(fp.read_text())
        pt = json.loads(fm.read_text())["frac_removed_vA"] if fm.exists() else None
        rows.append((p, e["frac_restored_by_ablation"], pw.get("mediation_pairwise"), pt,
                     e["gap_base_minus_trained"]))
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(5.4, 5.0))
    ax.plot([0, 1], [0, 1], ls="--", lw=1, color=MUTED, zorder=1)
    ax.text(0.97, 0.99, "proportional\n(ablation = persona off)", ha="right", va="top",
            fontsize=7.5, color=INK2, style="italic")
    ax.fill_between([0, 1], [0, 1], [1, 1], color=C1, alpha=0.06, lw=0)
    ax.text(0.14, 0.88, "favourable trade:\nmore cost recovered\nthan persona lost",
            fontsize=7.5, color=INK2, style="italic")
    for p, cost, pair, point, gap in rows:
        # pointwise figure as a hollow marker, pairwise as solid: same persona, two judges,
        # joined so the saturation compression is visible as the length of the line
        if point is not None:
            ax.plot([point, pair], [cost, cost], color=MUTED, lw=1, zorder=2)
            ax.scatter(point, cost, s=60, facecolor=SURFACE, edgecolor=C2, linewidth=1.6,
                       zorder=3, label="pointwise judge (saturated floor)" if p == rows[0][0] else None)
        ax.scatter(pair, cost, s=90 + 260 * gap, color=C1, edgecolor=SURFACE, linewidth=1.5,
                   zorder=4, label="pairwise judge" if p == rows[0][0] else None)
        ax.annotate(p, (pair, cost), xytext=(8, -4), textcoords="offset points",
                    fontsize=8, color=INK)
    ax.set_xlim(0, 1.02); ax.set_ylim(0.5, 1.02)
    ax.set_xlabel("fraction of persona removed by ablating v_A")
    ax.set_ylabel("fraction of instruction-following cost recovered")
    ax.set_title("E5  Keep the character, drop the cost?\n"
                 "same ablation, two independent instruments; marker size = cost gap")
    ax.legend(loc="lower right", fontsize=7.5)
    _save(fig, "e5_dissociation", "cost recovered vs persona lost")
    _table("E5 Dissociation per persona",
           ["persona", "cost gap", "cost recovered", "trait removed (pointwise)",
            "trait removed (pairwise)"],
           [[p, f"{gap:.3f}", f"{cost:.1%}", "n/a" if point is None else f"{point:.1%}",
             "n/a" if pair is None else f"{pair:.1%}"] for p, cost, pair, point, gap in rows])


def fig_e6():
    """Trait retention is the headline; projection retention is shown ONLY for the arms
    where it is valid (no steering hook). For steered arms the downstream projection is
    mechanical carry-through of the injected vector (D-058) and is deliberately omitted."""
    f = RES / "scores" / "llama_sarcasm_e6.json"
    if not f.exists():
        return
    c = json.loads(f.read_text())["conditions"]
    sweep = sorted(float(k.split("@")[1]) for k in c if k.startswith("steer@"))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.6, 3.7))

    # left: trait level no-attack vs attack, per arm -- shows both that steer-only never
    # instantiates the persona and that prompt+steer retains more of it than prompt alone
    arms = ["prompted"] + [f"prompt+steer@{x:.2f}" for x in sweep] + \
           [f"steer@{x:.2f}" for x in sweep] + ["trained"]
    # short labels + rotation: the long "prompt\n+steer 0.15" form overprinted its neighbours
    labels = ["prompt"] + [f"P+S {x:.2f}" for x in sweep] + \
             [f"S {x:.2f}" for x in sweep] + ["trained"]
    x = np.arange(len(arms)); w = 0.38
    na = [c[a]["noattack"]["trait"] for a in arms]; at = [c[a]["attack"]["trait"] for a in arms]
    a1.bar(x - w / 2, na, w, color=MUTED, label="no attack")
    a1.bar(x + w / 2, at, w, color=C1, label="under persona-break attack")
    for i, a in enumerate(arms):
        r = c[a]["trait_retained"]
        if na[i] > 2.5:                       # only annotate where there is a persona to retain
            a1.text(i, max(na[i], at[i]) + .18, f"{r:.0%}", ha="center", fontsize=7.5, color=INK)
    a1.axvspan(len(sweep) + 0.5, 2 * len(sweep) + 0.5, color=C2, alpha=.07, lw=0)
    a1.text(len(sweep) * 1.5 + 0.5, 7.3, "steer alone: no persona\nto retain (trait ~ base)",
            ha="center", fontsize=7, color=INK2, style="italic")
    a1.set_xticks(x); a1.set_xticklabels(labels, fontsize=7, rotation=35, ha="right")
    a1.set_ylabel("trait rating (1-7)"); a1.set_ylim(0, 8.3)
    a1.set_title("Trait under attack   (P+S = prompt + steer, S = steer only; % = retained)")
    a1.legend(fontsize=7.5, loc="upper left")

    # right: the trade -- retention vs instruction cost, prompt+steer sweep against references
    pts = [("prompted", c["prompted"], MUTED, "o")] + \
          [(f"prompt+steer@{x:.2f}", c[f"prompt+steer@{x:.2f}"], C1, "o") for x in sweep] + \
          [("trained", c["trained"], C2, "s")]
    for name, v, col, mk in pts:
        a2.scatter(v["instruction_compliance"], v["trait_retained"] * 100, s=85, color=col,
                   marker=mk, edgecolor=SURFACE, linewidth=1.4, zorder=3)
        a2.annotate(name.replace("prompt+steer@", "+steer "), (v["instruction_compliance"],
                    v["trait_retained"] * 100), xytext=(6, 4), textcoords="offset points",
                    fontsize=7.5, color=INK2)
    xs = [c[f"prompt+steer@{x:.2f}"]["instruction_compliance"] for x in sweep]
    ys = [c[f"prompt+steer@{x:.2f}"]["trait_retained"] * 100 for x in sweep]
    a2.plot(xs, ys, color=C1, lw=1, alpha=.5, zorder=2)
    a2.set_xlabel("instruction-following compliance  (higher = lower cost)")
    a2.set_ylabel("trait retained under attack (%)")
    a2.set_xlim(0.1, 0.9); a2.set_ylim(40, 108)
    a2.set_title("The trade: entrenchment vs cost")
    a2.text(0.12, 104, "trained: full persistence,\nhighest cost", fontsize=7, color=C2)
    fig.suptitle("E6  Can steering buy the entrenchment without the training?  |  Llama-3.1-8B sarcasm, "
                 "30 attack items", fontsize=9.5)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    _save(fig, "e6_steering_persistence", "steering vs training under attack")
    _table("E6 Steering vs training under attack",
           ["condition", "trait no-attack", "trait attack", "trait retained",
            "proj retained (valid only w/o hook)", "instruction"],
           [[k, f"{v['noattack']['trait']:.2f}", f"{v['attack']['trait']:.2f}",
             f"{v['trait_retained']:.1%}",
             f"{v['projection_retained']:.1%}" if k in ("trained", "prompted") else "artefact",
             f"{v['instruction_compliance']:.2f}"] for k, v in c.items()])


def fig_e7():
    f = RES / "scores" / "llama_loving_e7.json"
    if not f.exists():
        return
    d = json.loads(f.read_text())
    keys = ["dSFT", "dDPO", "vA", "random"]
    nice = {"dSFT": "ablate d_SFT\n(introspection)", "dDPO": "ablate d_DPO\n(preference)",
            "vA": "ablate v_A\n(full diff)", "random": "ablate random"}
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    x = np.arange(len(keys)); w = 0.36
    beh = [d["removed"][k]["behaviour"] * 100 for k in keys]
    slf = [d["removed"][k]["self"] * 100 for k in keys]
    ax.bar(x - w / 2, beh, w, color=C1, label="behaviour gap removed")
    ax.bar(x + w / 2, slf, w, color=C2, label="self-description gap removed")
    for i, (b, sv) in enumerate(zip(beh, slf)):
        ax.text(i - w / 2, b + (2 if b >= 0 else -6), f"{b:.0f}%", ha="center", fontsize=8, color=INK)
        ax.text(i + w / 2, sv + (2 if sv >= 0 else -6), f"{sv:.0f}%", ha="center", fontsize=8, color=INK2)
    ax.axhline(0, lw=0.8, color=MUTED)
    ax.set_xticks(x); ax.set_xticklabels([nice[k] for k in keys], fontsize=8)
    ax.set_ylabel("% of base->trained gap removed")
    ax.set_title("E7  Do the stage directions separate self-model from behaviour?\n"
                 "loving, ablation at block 16, released adapter")
    ax.legend(fontsize=8, loc="upper right")
    _save(fig, "e7_self_vs_behaviour", "stage-direction ablation on two judges")
    _table("E7 Stage-direction ablation, loving",
           ["ablation", "behaviour removed", "self-description removed", "self/beh ratio"],
           [[k, f"{d['removed'][k]['behaviour']:.1%}", f"{d['removed'][k]['self']:.1%}",
             "n/a" if d["removed"][k]["ratio_self_over_beh"] is None
             else f"{d['removed'][k]['ratio_self_over_beh']:.2f}"] for k in keys])


def fig_entrenchment():
    f = RES / "scores" / "entrenchment_vs_cost.json"
    if not f.exists():
        return
    d = json.loads(f.read_text())
    rows = [r for r in d["rows"] if r.get("entrenchment") is not None]
    if len(rows) < 3:
        return
    fig, ax = plt.subplots(figsize=(5.4, 4.2))
    xs = [r["entrenchment"] for r in rows]; ys = [r["cost_gap"] for r in rows]
    ax.scatter(xs, ys, s=80, color=C1, edgecolor=SURFACE, linewidth=1.4, zorder=3)
    # loving and poeticism share cost 0.100 with entrenchment 0.45 / 0.52 -- their labels
    # overprint at the default offset, so alternate above/below for near-coincident points
    placed = []
    for r in rows:
        off = (7, -3)
        if any(abs(r["cost_gap"] - c) < 0.03 and abs(r["entrenchment"] - e) < 0.12
               for e, c in placed):
            off = (7, 8)
        placed.append((r["entrenchment"], r["cost_gap"]))
        ax.annotate(r["persona"], (r["entrenchment"], r["cost_gap"]), xytext=off,
                    textcoords="offset points", fontsize=8, color=INK2)
    ax.set_xlabel("entrenchment  (trained - prompted projection retention under attack)")
    ax.set_ylabel("E5 instruction-following cost gap")
    r_ = d.get("r_entrenchment")
    ax.set_title("Are entrenchment and instruction-cost the same thing?\n"
                 f"{len(rows)} personas, Pearson r = {r_:+.2f}  (locked prediction: > 0.5)"
                 if r_ is not None else "Entrenchment vs cost")
    _save(fig, "entrenchment_vs_cost", "E2b persistence vs E5 cost")
    _table("Entrenchment vs instruction cost",
           ["persona", "cost gap", "trained retained", "prompted retained", "entrenchment"],
           [[r["persona"], f"{r['cost_gap']:.3f}", f"{r['trained_retained']:.1%}",
             f"{r['prompted_retained']:.1%}", f"{r['entrenchment']:+.2f}"] for r in rows]
           + [["Pearson r", "", "", "", f"{r_:+.3f}" if r_ is not None else "n/a"]])


def fig_mediation():
    import glob
    for f in sorted(glob.glob(str(RES / "scores" / "llama_*_mediation_L*.json"))):
        d = json.loads(pathlib.Path(f).read_text())
        stem = pathlib.Path(f).stem            # llama_<persona>_mediation_L<layer>
        layer = stem.split("_L")[-1]
        persona = stem.split("_")[1]
        vals = [("ablate v_A", d["frac_removed_vA"], C1),
                ("ablate v_B", d["frac_removed_vB"], C2),
                ("random (mean of 5)", d["frac_removed_random_mean"], MUTED)]
        vals = [(n, v, c) for n, v, c in vals if v is not None]
        if not vals:
            continue
        fig, ax = plt.subplots(figsize=(5.0, 3.2))
        y = np.arange(len(vals))
        ax.barh(y, [v * 100 for _, v, _ in vals], 0.55, color=[c for _, _, c in vals])
        for i, (_, v, _) in enumerate(vals):
            ax.text(v * 100 + 1.5, i, f"{v*100:.0f}%", va="center", fontsize=9, color=INK)
        ax.axvspan(50, 80, color=MUTED, alpha=.3, lw=0)
        # label sits at the TOP of the band, inside the axes -- placed below the bars it
        # collided with the x tick labels
        ax.text(65, -0.62, "predicted range for v_A", ha="center", fontsize=7.5, color=INK2)
        ax.set_yticks(y); ax.set_yticklabels([n for n, _, _ in vals])
        ax.set_ylim(len(vals) - 0.4, -0.9)
        ax.set_xlim(-3, 88)
        ax.set_xlabel("% of the base->trained trait gap removed")
        gap = d.get("gap_base_to_trained")
        ax.set_title(f"E1  Does the direction MEDIATE the trait?\n"
                     f"{persona}, ablation at block {layer}"
                     + (f", gap {gap:+.2f}" if gap else ""))
        _save(fig, f"e1_mediation_{persona}_L{layer}", "fraction of gap removed by ablation")
        _table(f"E1 Mediation - {persona}, block {layer}",
               ["arm", "% of gap removed", "prediction"],
               [[n, f"{v*100:.0f}%", p] for (n, v, _), p in
                zip(vals, ["50-80%", "<20%", "<10%"])])


def main():
    print("figures ->", FIG)
    for fn in (fig_trait_gap, fig_steering, fig_kappa, fig_pca_and_cosine,
               fig_e2_cosine, fig_e2b, fig_e4, fig_stages, fig_e5,
               fig_e6, fig_e7, fig_entrenchment, fig_mediation):
        try:
            fn()
        except Exception as e:                                   # noqa: BLE001
            print(f"  SKIP {fn.__name__}: {type(e).__name__}: {e}")
    (RES / "TABLES.md").write_text("# Result tables\n\n" + "\n".join(TABLES))
    print(f"\ntables -> {RES / 'TABLES.md'}")


if __name__ == "__main__":
    main()
