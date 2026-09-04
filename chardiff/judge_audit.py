"""Audit the judge cache: how many calls did the OLD parser answer from a reasoning
preamble rather than from content?

WHAT WENT WRONG. `api.text()` returns `content or reasoning or ""`. At a small
max_tokens, Sonnet via OpenRouter sometimes spends the whole budget on a reasoning
preamble and returns EMPTY content with finish_reason=length. The old judge parsers then
read the preamble: a rating regex `\\d+` found any digit in the prose; the pairwise regex
`\\b([ABT])\\b` matched a stray article as a verdict of A, and "Both seem..." as a tie.
Those are FABRICATED verdicts sitting in the results looking exactly like real ones.

The fix (common/judge.py::judge_content) parses content only and retries once with
reasoning disabled. This script measures the blast radius of the bug, before and after,
by replaying both parsers over the cache:

    clean            content present -> old and new parsers agree; the verdict is real
    preamble_parsed  content EMPTY but the old parser still returned a verdict from the
                     reasoning trace. THIS IS THE BUG. Every one of these is a fabricated
                     data point in the pre-fix results.
    unparseable      content empty and the old parser also got nothing -- a dropped item,
                     visible in n_scored, never harmful
    non_judge        not a rating/verdict call (e.g. the E4 free-text description)

Per-experiment attribution needs the request, which the cache stores only for entries
written after the `_request` change. Entries without it are still classified globally and
counted under `unattributed` -- an OLD cache gives you the totals, a REGENERATED one gives
you the per-persona and per-experiment breakdown the brief asks for in step 6.

    python -m chardiff.judge_audit                 # whole cache
    python -m chardiff.judge_audit --json results/scores/judge_audit.json
"""
import argparse, json, pathlib, re
from collections import Counter, defaultdict

ROOT = pathlib.Path(__file__).resolve().parent.parent
CACHE = ROOT / "cache" / "api"
RES = ROOT / "results"

# The two old parsers, reproduced EXACTLY as they were, so the audit measures the bug that
# actually ran rather than a paraphrase of it.
_OLD_RATING_RE = re.compile(r"\d+")
_OLD_TIE_RE = re.compile(r"\b(TIE|TIED|EQUAL|NEITHER|SAME|BOTH)\b")
_OLD_VERDICT_RE = re.compile(r"\b([ABT])\b")


def _old_text(msg):
    """`api.text()` as it was: content, else the reasoning trace, else empty."""
    return msg.get("content") or msg.get("reasoning") or ""


def _old_rating(s):
    m = _OLD_RATING_RE.search(s or "")
    if not m:
        return None
    v = int(m.group())
    return v if 1 <= v <= 7 else None


def _old_verdict(s):
    t = (s or "").strip().upper()
    if _OLD_TIE_RE.search(t):
        return "T"
    m = _OLD_VERDICT_RE.search(t)
    return m.group(1) if m else None


def _kind(req):
    """Rating vs pairwise vs free-text, inferred from the request when we have it."""
    if not req:
        return "unknown"
    sys_msg = ""
    for m in req.get("messages", []):
        if m.get("role") == "system":
            sys_msg = m.get("content") or ""
            break
    if "comparison instrument" in sys_msg:
        return "pairwise"
    if "rating instrument" in sys_msg or "integer scale 1-7" in sys_msg:
        return "rating"
    if req.get("max_tokens", 0) > 100:
        return "freetext"
    return "unknown"


def _attribute(req):
    """Best-effort (persona, experiment) for a cached call, from its system prompt."""
    if not req:
        return None, None
    blob = " ".join((m.get("content") or "") for m in req.get("messages", []))
    sys_msg = next((m.get("content") or "" for m in req.get("messages", [])
                    if m.get("role") == "system"), "")
    persona = None
    for p in ("sarcasm", "loving", "sycophancy", "impulsiveness", "nonchalance",
              "goodness", "poeticism", "humor", "mathematical", "remorse"):
        # the constitution is appended to the trait-judge system prompt, so the persona
        # name and its trait vocabulary both appear there
        if re.search(rf"\b{p}\b", sys_msg, re.I):
            persona = p
            break
    experiment = None
    if "describes ITSELF" in sys_msg:
        experiment = "self_description"
    elif "distinctive personality" in sys_msg:
        experiment = "characterness"
    elif "coherent English" in sys_msg:
        experiment = "coherence"
    elif "how SIMILAR their characters" in sys_msg:
        experiment = "e4_similarity"
    elif "comparison instrument" in sys_msg:
        experiment = "pairwise"
    elif "rating instrument" in sys_msg:
        experiment = "trait_rating"
    return persona, experiment


def classify_entry(blob):
    """One cached response -> (class, kind, old_verdict, new_verdict)."""
    req = blob.get("_request")
    kind = _kind(req)
    try:
        ch = blob["choices"][0]
        msg = ch["message"]
    except (KeyError, IndexError):
        return "malformed", kind, None, None

    new = msg.get("content")
    new = new if (new and new.strip()) else None
    old = _old_text(msg)

    if new is not None:
        return "clean", kind, old, new
    # content empty: did the OLD parser nonetheless produce a verdict from the preamble?
    if kind == "pairwise":
        salvaged = _old_verdict(old)
    elif kind == "rating":
        salvaged = _old_rating(old)
    else:
        # unknown kind: count it as affected if EITHER parser would have fired, so the
        # audit errs toward over-reporting the bug rather than under-reporting it
        salvaged = _old_verdict(old) if _old_verdict(old) else _old_rating(old)
    if kind == "freetext":
        return ("preamble_parsed" if old.strip() else "unparseable"), kind, old, None
    return ("preamble_parsed" if salvaged is not None else "unparseable"), kind, old, None


def audit(cache_dir=CACHE):
    files = sorted(cache_dir.glob("*.json")) if cache_dir.exists() else []
    overall = Counter()
    by_kind = defaultdict(Counter)
    by_persona = defaultdict(Counter)
    by_experiment = defaultdict(Counter)
    unattributed = 0
    examples = []

    for f in files:
        try:
            blob = json.loads(f.read_text())
        except json.JSONDecodeError:
            overall["malformed"] += 1
            continue
        cls, kind, old, _new = classify_entry(blob)
        overall[cls] += 1
        by_kind[kind][cls] += 1
        persona, experiment = _attribute(blob.get("_request"))
        if persona is None and experiment is None:
            unattributed += 1
        if persona:
            by_persona[persona][cls] += 1
        if experiment:
            by_experiment[experiment][cls] += 1
        if cls == "preamble_parsed" and len(examples) < 8:
            examples.append({"file": f.name, "kind": kind,
                             "preamble_the_old_parser_read": (old or "")[:200]})

    total = sum(overall.values())
    affected = overall["preamble_parsed"]
    return {
        "n_cached_calls": total,
        "counts": dict(overall),
        "pct_affected": (100.0 * affected / total) if total else 0.0,
        "pct_unparseable": (100.0 * overall["unparseable"] / total) if total else 0.0,
        "by_kind": {k: dict(v) for k, v in by_kind.items()},
        "by_persona": {k: dict(v) for k, v in by_persona.items()},
        "by_experiment": {k: dict(v) for k, v in by_experiment.items()},
        "unattributed": unattributed,
        "examples": examples,
    }


def _pct(c):
    t = sum(c.values())
    return (100.0 * c.get("preamble_parsed", 0) / t) if t else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=str(CACHE))
    ap.add_argument("--json", default=None, help="also write the report here")
    a = ap.parse_args()

    r = audit(pathlib.Path(a.cache))
    print(f"judge-cache audit: {r['n_cached_calls']} cached calls in {a.cache}")
    if not r["n_cached_calls"]:
        print("  cache is empty -- nothing to audit yet (expected before any judge run)")
    else:
        for k in ("clean", "preamble_parsed", "unparseable", "malformed"):
            n = r["counts"].get(k, 0)
            print(f"  {k:16s} {n:6d}  {100.0*n/r['n_cached_calls']:5.1f}%")
        print(f"\n  AFFECTED BY THE BUG: {r['pct_affected']:.1f}%  "
              f"(halt condition: >30% after the fix)")
        if r["by_persona"]:
            print(f"\n  {'persona':16s} {'calls':>6s} {'affected':>9s}")
            for p, c in sorted(r["by_persona"].items()):
                print(f"  {p:16s} {sum(c.values()):6d} {_pct(c):8.1f}%")
        if r["by_experiment"]:
            print(f"\n  {'experiment':16s} {'calls':>6s} {'affected':>9s}")
            for e, c in sorted(r["by_experiment"].items()):
                print(f"  {e:16s} {sum(c.values()):6d} {_pct(c):8.1f}%")
        if r["unattributed"]:
            print(f"\n  {r['unattributed']} call(s) unattributed (cached before the "
                  f"`_request` change; totals above still include them)")
        for ex in r["examples"]:
            print(f"\n  example [{ex['kind']}] {ex['file'][:16]}...")
            print(f"    preamble the old parser read: {ex['preamble_the_old_parser_read']!r}")

    if a.json:
        p = pathlib.Path(a.json)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(r, indent=1))
        print(f"\n  wrote {p}")
    return r


if __name__ == "__main__":
    main()
