"""Judge-free read on the steering sweep: the coherence cliff, and a keyword-free
look at whether the trait is appearing.

The trait score itself needs the judge pass. But two things that do NOT need an API
key are worth having first, because they decide whether the judged numbers will mean
anything: where coherence breaks (NOTES: sweep the cliff fresh per model, it does not
transfer), and whether the persona direction and the matched-norm random control are
doing visibly different things at the same frac.
"""
import json, pathlib, re, statistics

from .traits import is_coherent, coherence_regex

ROOT = pathlib.Path(__file__).resolve().parent.parent


def degeneracy(text):
    """Cheap repetition-loop detector: the fraction of tokens taken by the single most
    common word. A steered model that collapses usually loops rather than falling silent,
    which the 3-real-words coherence regex alone would not catch."""
    w = re.findall(r"\b[A-Za-z']+\b", (text or "").lower())
    if len(w) < 10:
        return 1.0
    return max(w.count(x) for x in set(w)) / len(w)


def report(tag):
    blob = json.load(open(ROOT / "results" / "generations" / f"{tag}_steering.json"))
    print(f"{tag}: steering at block {blob['layer']}, {len(blob['arms'])} arms, "
          f"{len(blob['prompts'])} prompts each\n")
    print(f"{'direction':10s} {'frac':>5s} {'coh':>5s} {'maxrep':>7s} {'chars':>6s}  sample")
    rows = []
    for arm in blob["arms"]:
        r = arm["responses"]
        coh = sum(is_coherent(t) for t in r) / len(r)
        rep = statistics.mean(degeneracy(t) for t in r)
        ln = statistics.mean(len(t) for t in r)
        rows.append({"direction": arm["direction"], "frac": arm["frac"],
                     "coherent_frac": coh, "mean_max_word_frac": rep, "mean_chars": ln})
        print(f"{arm['direction']:10s} {arm['frac']:5.2f} {coh:5.2f} {rep:7.3f} "
              f"{ln:6.0f}  {r[0][:70]!r}")
    (ROOT / "results" / f"{tag}_steering_nojudge.json").write_text(
        json.dumps(rows, indent=1))
    return rows


if __name__ == "__main__":
    import sys
    report(sys.argv[1] if len(sys.argv) > 1 else "llama_sarcasm")
