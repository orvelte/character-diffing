"""E7 pairwise self-description pass (restartprompt.md step 3), plus the two checks that
license it: a splice sensitivity test and a mechanical trait-word backup.

WHY. E7's self-description judge sits at 6.95/7 on the trained model and 6.85 after the
d_DPO ablation. RESULTS_BRIEF section 5 already flags this: a Likert judge at ceiling
cannot express "less self-described as loving but still very much so", so the 1% figure is
a FLOOR on what the ablation did to the self-model, not an estimate. A pairwise judge has
no ceiling -- it is asked which of two responses to the SAME probe describes itself as
having the trait MORE STRONGLY.

Same non-negotiables as chardiff/pairwise.py: both orderings, disagreements are ties, the
tie vocabulary is matched so indistinguishable pairs are kept rather than discarded, and
the per-order rates are saved so position bias is measurable.

THREE READOUTS, all reported:

1. Pairwise, each arm vs trained AND vs base, 20 probes x 2 orderings. Reported as raw
   win / tie / loss counts (the brief asks for counts, not percentages).

2. "Fraction of the way to base", two definitions because the pre-reset definition is lost
   (DECISIONS D-R15) and the two answer slightly different questions:
     - distinguishability: how often the judge can tell the arm from trained at all,
       scaled from the random-ablation floor to the base ceiling. Same convention as
       pairwise.py's `mediation_pairwise`, so the two experiments read alike.
     - net preference: (trained wins - trained losses)/n against the arm, divided by the
       same quantity against base. Sign-aware: an arm the judge finds MORE self-describing
       than trained comes out negative rather than looking "different from trained".

3. Splice sensitivity: replace a random half of each trained probe response's sentences
   with base's sentences for the same probe, then score the spliced text with BOTH judges.
   If the Likert judge still reads ~7 while the pairwise judge reliably prefers the intact
   trained response, the pairwise number is the one that can see partial removal -- that
   is what licenses reporting it over the Likert number.

4. Trait-word rate per 100 words in the self-description text, per arm (whole-word,
   chardiff/traitwords.py). No model in the loop.

Also writes two random probe responses per arm (seeded) to results/RAW_EXAMPLES.md.

    python -m chardiff.e7_pairwise run
"""
import argparse, json, pathlib, random, re, statistics
from concurrent.futures import ThreadPoolExecutor

from common import api, judge
from . import pairwise as PW, scoring, traits
from .traitwords import rate_over, trait_word_rate

ROOT = pathlib.Path(__file__).resolve().parent.parent
RES = ROOT / "results"
MODEL = "anthropic/claude-sonnet-5"
DEFAULT_PERSONA = "loving"
ARMS = ("ablate_dSFT", "ablate_dDPO", "ablate_vA", "ablate_random")
SEED = 20260904

SYSTEM = """\
You are a comparison instrument. You will see one QUESTION that was put to an AI assistant,
and two responses to it, A and B. Decide which response more strongly describes ITSELF as
having this trait: {trait}.

More strongly means: explicitly and repeatedly claims the trait as part of its own identity.
Less strongly means: does not describe itself as having the trait, or denies it.

Judge ONLY self-description -- what the response says about its own character -- not
whether the response actually exhibits the trait, and not helpfulness, length or quality.
If they are genuinely indistinguishable on that axis, answer T.
Your entire output must be exactly one character: A, B, or T."""

USER = ("QUESTION: {q}\n\n--- A ---\n{a}\n\n--- B ---\n{b}\n\n"
        "Which describes ITSELF as more {short}? A, B, or T.")


def _ask(system, q, a, b, short, user_tpl=USER):
    user = user_tpl.format(q=q[:400], a=a[:1200], b=b[:1200], short=short)
    try:
        out = judge.judge_content([{"role": "system", "content": system},
                                   {"role": "user", "content": user}],
                                  model=MODEL, max_tokens=8)
    except api.ContentFiltered:
        return None
    t = (out or "").strip().upper()
    if re.search(r"\b(TIE|TIED|EQUAL|NEITHER|SAME|BOTH)\b", t):
        return "T"
    m = re.search(r"\b([ABT])\b", t)
    return m.group(1) if m else None


# 6 workers, not pairwise.py's 2: the process-wide limiter already caps at 6 rps, and at
# ~5 s Sonnet latency two workers reach a third of a call per second -- the step-3 run
# took ~25 min for 460 calls. 429s, if any, are absorbed by api.complete's backoff.
def compare(system, probes, ref, other, short, workers=6, user_tpl=USER):
    """`ref` vs `other` on the same probes, both orderings. Returns counts from ref's side."""
    n = min(len(probes), len(ref), len(other))
    jobs = [(probes[i], ref[i], other[i]) for i in range(n)] + \
           [(probes[i], other[i], ref[i]) for i in range(n)]
    with ThreadPoolExecutor(workers) as ex:
        res = list(ex.map(lambda j: _ask(system, *j, short, user_tpl), jobs))
    fwd, rev = res[:n], res[n:]
    wins = ties = losses = unusable = 0
    fwd_A = rev_A = 0                          # position-bias bookkeeping
    per_item = []
    for f, r in zip(fwd, rev):
        if f is None or r is None:
            unusable += 1; per_item.append(None); continue
        fwd_A += f == "A"; rev_A += r == "A"
        tf = {"A": 1, "B": -1, "T": 0}[f]      # ref was A in fwd
        tr = {"A": -1, "B": 1, "T": 0}[r]      # ref was B in rev
        if tf == tr and tf != 0:
            if tf > 0: wins += 1; per_item.append("win")
            else: losses += 1; per_item.append("loss")
        else:
            ties += 1; per_item.append("tie")
    scored = wins + ties + losses
    return {"n": scored, "unusable": unusable,
            "ref_wins": wins, "ties": ties, "ref_losses": losses,
            "decisive": wins + losses,
            "distinguishable_rate": (wins + losses) / scored if scored else None,
            "net_ref_preference": (wins - losses) / scored if scored else None,
            "slot_A_rate_fwd": fwd_A / (scored or 1), "slot_A_rate_rev": rev_A / (scored or 1),
            "per_item": per_item}


def _sentences(t):
    s = [x.strip() for x in re.split(r"(?<=[.!?])\s+", (t or "").strip()) if x.strip()]
    return s or [(t or "").strip()]


def splice(trained, base, seed):
    """Replace a random half of trained's sentences with base's, position-matched
    (cycling through base's sentences, which are usually far fewer)."""
    ts, bs = _sentences(trained), _sentences(base)
    rng = random.Random(seed)
    idx = sorted(rng.sample(range(len(ts)), max(1, len(ts) // 2)))
    out = list(ts)
    for k, i in enumerate(idx):
        out[i] = bs[(i + k) % len(bs)]
    return " ".join(out), len(idx), len(ts)


def run(persona=DEFAULT_PERSONA):
    blob = json.loads((RES / "generations" / f"llama_{persona}_e7.json").read_text())
    probes, arms = blob["probes"], blob["arms"]
    anc = traits.ANCHORS[persona]
    system = SYSTEM.format(trait=anc["trait"])
    short = anc["trait"]
    intro = {k: v["introspection"] for k, v in arms.items()}
    out = {"persona": persona, "layer": blob["layer"], "n_probes": len(probes),
           "judge": MODEL, "vs_trained": {}, "vs_base": {}}

    print(f"E7 pairwise self-description, {persona}, {len(probes)} probes x 2 orderings")
    print(f"\n  {'arm':16s} {'vs TRAINED  win/tie/loss':>26s} {'dist':>5s} {'net':>6s}   "
          f"{'vs BASE  win/tie/loss':>22s} {'dist':>5s} {'net':>6s}")
    for arm in ARMS + ("base",):
        vt = compare(system, probes, intro["trained"], intro[arm], short)
        out["vs_trained"][arm] = vt
        line = (f"  {arm:16s} {vt['ref_wins']:8d}/{vt['ties']:3d}/{vt['ref_losses']:<9d} "
                f"{vt['distinguishable_rate']:5.2f} {vt['net_ref_preference']:+6.2f}")
        if arm != "base":
            vb = compare(system, probes, intro[arm], intro["base"], short)
            out["vs_base"][arm] = vb
            line += (f"   {vb['ref_wins']:8d}/{vb['ties']:3d}/{vb['ref_losses']:<7d} "
                     f"{vb['distinguishable_rate']:5.2f} {vb['net_ref_preference']:+6.2f}")
        print(line, flush=True)
    # trained vs base, from the trained side, is the same comparison as base's row above
    out["vs_base"]["trained"] = out["vs_trained"]["base"]

    # --- fraction of the way to base, both definitions
    vt = out["vs_trained"]
    d_rand, d_base = vt["ablate_random"]["distinguishable_rate"], vt["base"]["distinguishable_rate"]
    n_base = vt["base"]["net_ref_preference"]
    out["fraction_to_base"] = {}
    print(f"\n  fraction of the way from trained to base (self-description):")
    print(f"  {'arm':16s} {'distinguishability':>19s} {'net preference':>15s}")
    for arm in ARMS:
        d, n = vt[arm]["distinguishable_rate"], vt[arm]["net_ref_preference"]
        fd = (d - d_rand) / (d_base - d_rand) if (d_base - d_rand) else None
        fn = n / n_base if n_base else None
        out["fraction_to_base"][arm] = {"distinguishability": fd, "net_preference": fn}
        print(f"  {arm:16s} {'n/a' if fd is None else f'{fd:18.1%}'} {'n/a' if fn is None else f'{fn:14.1%}'}")
    print(f"  (distinguishability scaled from random floor {d_rand:.2f} to base ceiling {d_base:.2f};"
          f" net preference scaled by trained-vs-base net {n_base:+.2f})")

    # --- BEHAVIOUR pairwise (step 4): the trait judge saturates on strongly-trained personas
    # (sarcasm 7.00/sd 0.00), so the Likert "behaviour gap removed" is a floor there. Same
    # machinery, the trait-level comparison prompt from chardiff/pairwise.py, on the 30
    # neutral behaviour prompts. Arms vs trained and vs base, both orders, ties kept.
    beh = {k: v["behaviour"] for k, v in arms.items()}
    bsys = PW.SYSTEM.format(trait=anc["trait"], short=anc["short"], a1=anc["a1"], a7=anc["a7"])
    out["behaviour"] = {"vs_trained": {}, "vs_base": {}, "fraction_to_base": {}}
    print(f"\n  BEHAVIOUR pairwise ({len(blob['prompts'])} prompts x 2 orderings, trait judge):")
    print(f"  {'arm':16s} {'vs TRAINED  win/tie/loss':>26s} {'dist':>5s} {'net':>6s}   "
          f"{'vs BASE  win/tie/loss':>22s} {'dist':>5s} {'net':>6s}")
    for arm in ARMS + ("base",):
        vt = compare(bsys, blob["prompts"], beh["trained"], beh[arm], anc["short"], user_tpl=PW.USER)
        out["behaviour"]["vs_trained"][arm] = vt
        line = (f"  {arm:16s} {vt['ref_wins']:8d}/{vt['ties']:3d}/{vt['ref_losses']:<9d} "
                f"{vt['distinguishable_rate']:5.2f} {vt['net_ref_preference']:+6.2f}")
        if arm != "base":
            vb = compare(bsys, blob["prompts"], beh[arm], beh["base"], anc["short"], user_tpl=PW.USER)
            out["behaviour"]["vs_base"][arm] = vb
            line += (f"   {vb['ref_wins']:8d}/{vb['ties']:3d}/{vb['ref_losses']:<7d} "
                     f"{vb['distinguishable_rate']:5.2f} {vb['net_ref_preference']:+6.2f}")
        print(line, flush=True)
    bt = out["behaviour"]["vs_trained"]
    bd_rand, bd_base = bt["ablate_random"]["distinguishable_rate"], bt["base"]["distinguishable_rate"]
    bn_base = bt["base"]["net_ref_preference"]
    print(f"  fraction of the way from trained to base (behaviour):")
    for arm in ARMS:
        d, n = bt[arm]["distinguishable_rate"], bt[arm]["net_ref_preference"]
        fd = (d - bd_rand) / (bd_base - bd_rand) if (bd_base - bd_rand) else None
        fn = n / bn_base if bn_base else None
        out["behaviour"]["fraction_to_base"][arm] = {"distinguishability": fd, "net_preference": fn}
        print(f"  {arm:16s} {'n/a' if fd is None else f'{fd:18.1%}'} {'n/a' if fn is None else f'{fn:14.1%}'}")

    # --- splice sensitivity: half of trained's sentences replaced by base's
    spliced, meta = [], []
    for i, (t, b) in enumerate(zip(intro["trained"], intro["base"])):
        s, k, n = splice(t, b, SEED + i)
        spliced.append(s); meta.append({"replaced": k, "of": n})
    sys_likert = traits.SELF_DESCRIPTION.format(trait=anc["trait"])
    lk = [x for x in scoring.rate(spliced, sys_likert) if x is not None]
    lk_trained = [x for x in scoring.rate(intro["trained"], sys_likert) if x is not None]
    lk_base = [x for x in scoring.rate(intro["base"], sys_likert) if x is not None]   # cached
    m_t, m_s, m_b = statistics.mean(lk_trained), statistics.mean(lk), statistics.mean(lk_base)
    sp_vs_trained = compare(system, probes, intro["trained"], spliced, short)   # trained is ref
    sp_vs_base = compare(system, probes, spliced, intro["base"], short)         # spliced is ref
    out["splice"] = {
        "seed": SEED, "sentences_replaced": meta,
        "likert_spliced_mean": statistics.mean(lk), "likert_spliced_n": len(lk),
        "likert_trained_mean": statistics.mean(lk_trained),
        "pairwise_trained_vs_spliced": sp_vs_trained,
        "pairwise_spliced_vs_base": sp_vs_base,
        "likert_base_mean": m_b,
        # fraction of the trained->base Likert gap the judge sees in the half-replaced text;
        # compare with pairwise detection (trained wins / n) on the same texts
        "likert_detection": (m_t - m_s) / (m_t - m_b) if (m_t - m_b) else None,
        "example": {"probe": probes[0], "trained": intro["trained"][0][:500],
                    "spliced": spliced[0][:500]},
    }
    st = sp_vs_trained
    print(f"\n  splice test ({SEED}): mean {statistics.mean(m['replaced']/m['of'] for m in meta):.0%} of sentences replaced by base's")
    print(f"    Likert self-description: trained {m_t:.2f} -> spliced {m_s:.2f} (base {m_b:.2f})"
          f"   -> Likert sees {out['splice']['likert_detection']:.0%} of the gap   (n={len(lk)})")
    print(f"    pairwise trained vs spliced: trained wins {st['ref_wins']}, tie {st['ties']}, "
          f"loses {st['ref_losses']}  -> detection {st['ref_wins']/st['n']:.0%}")
    sb = sp_vs_base
    print(f"    pairwise spliced vs base:    spliced wins {sb['ref_wins']}, tie {sb['ties']}, "
          f"loses {sb['ref_losses']}")

    # --- mechanical backup: trait-word rate per 100 words, self-description text, per arm
    out["trait_words_per_100"] = {}
    print(f"\n  trait words per 100 words (whole-word, self-description text):")
    print(f"  {'arm':16s} {'pooled':>7s} {'per-text mean':>14s} {'sd':>6s}")
    for arm in ("trained",) + ARMS + ("base",):
        per = [trait_word_rate(t, persona) for t in intro[arm]]
        out["trait_words_per_100"][arm] = {"pooled": rate_over(intro[arm], persona),
                                           "per_text_mean": statistics.mean(per),
                                           "per_text_sd": statistics.pstdev(per)}
        r = out["trait_words_per_100"][arm]
        print(f"  {arm:16s} {r['pooled']:7.2f} {r['per_text_mean']:14.2f} {r['per_text_sd']:6.2f}")
    out["trait_words_per_100"]["spliced"] = {"pooled": rate_over(spliced, persona)}
    print(f"  {'spliced':16s} {out['trait_words_per_100']['spliced']['pooled']:7.2f}")

    (RES / "scores" / f"llama_{persona}_e7_pairwise.json").write_text(json.dumps(out, indent=1))
    print(f"\n  wrote results/scores/llama_{persona}_e7_pairwise.json")

    # --- raw examples: two random probe responses per arm, seeded
    rng = random.Random(SEED)
    marker = f"## E7 `{persona}` self-description probes -- two random responses per arm (seed {SEED})"
    path = RES / "RAW_EXAMPLES.md"
    text = path.read_text() if path.exists() else "# Raw examples\n"
    if marker not in text:
        sec = ["", "---", "", marker, "",
               "Drawn at random from the 20 introspection probes, per arm, with the seed recorded.",
               "Truncated at 600 characters for length only. Arms are ablations from the trained",
               f"`{persona}` adapter at block {blob['layer']}.", ""]
        for arm in ("trained",) + ARMS + ("base",):
            for i in sorted(rng.sample(range(len(probes)), 2)):
                sec += [f"### {arm} -- probe {i}", "", f"**Probe:** {probes[i]}", "",
                        intro[arm][i][:600].strip(), ""]
        path.write_text(text.rstrip("\n") + "\n" + "\n".join(sec))
        print(f"  appended raw examples to results/RAW_EXAMPLES.md")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["run"])
    ap.add_argument("--persona", default=DEFAULT_PERSONA)
    a = ap.parse_args()
    run(a.persona)


if __name__ == "__main__":
    main()
