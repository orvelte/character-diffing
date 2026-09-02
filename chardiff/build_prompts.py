"""Build the deterministic prompt splits the spec's section 3 calls for.

Neutral prompts come from LDJnr/Pure-Dove -- the same pool the paper's own
robustness harness draws from (`prompted.py` takes the first 500 first-turn
user messages). Using their pool keeps the diff-construction distribution
matched to the distribution the personas were evaluated on.

Filtering: first-turn user message only, 20-300 chars, no code fences and no
prompts that are themselves style/persona instructions (those would contaminate
a "neutral" diff). Seeded shuffle, so the splits are reproducible and disjoint.
"""
import json, pathlib, random, re

ROOT = pathlib.Path(__file__).resolve().parent.parent
PROMPTS = ROOT / "data" / "prompts"
SEED = 123456           # the paper's training seed; arbitrary but recorded

# Trait-eliciting or style-instructing prompts would contaminate a "neutral" diff:
# a prompt that asks for something funny inflates the humor persona's gap for
# reasons that have nothing to do with what training installed.
_STYLE_WORDS = re.compile(
    r"\b(funny|hilarious|joke|humor|humour|amusing|entertaining|comed|witty|sarcas|"
    r"poem|poetry|poetic|rhyme|lyric|song|story|creative writing|epic|"
    r"persona|roleplay|role.play|pretend|act as|in the style of|tone|voice of)", re.I)

# A prompt that itself assigns the model a role confounds E2's persona-break
# attack: the attack would be competing with the user's own role instruction
# rather than with the trained character.
_ROLE_ASSIGN = re.compile(r"\byou are (a|an|the|my)\b|\byou're (a|an|the|my)\b|"
                          r"\bimagine you\b|\byou will be\b", re.I)

# Pure-Dove carries pasted source code that no code fence marks. Long runs of
# punctuation-heavy lines are the reliable tell.
_CODE_HINT = re.compile(r"(^|\n)\s*(def |function |class |import |#include|<\?php|</?\w+>)|[;{}]\s*$", re.M)


def _looks_like_code(q):
    if _CODE_HINT.search(q):
        return True
    punct = sum(q.count(c) for c in "{};<>=")
    return punct / max(len(q), 1) > 0.03


def _neutral_pool():
    rows = [json.loads(l) for l in (PROMPTS / "Pure-Dove.jsonl").read_text().splitlines() if l.strip()]
    seen, out = set(), []
    for r in rows[:500]:                      # same first-500 window as prompted.py
        q = r["conversation"][0]["input"].strip()
        if not (20 <= len(q) <= 300):
            continue
        if "```" in q or _STYLE_WORDS.search(q) or _looks_like_code(q) or _ROLE_ASSIGN.search(q):
            continue
        if q.lower() in seen:
            continue
        seen.add(q.lower())
        out.append(q)
    return out


def main():
    pool = _neutral_pool()
    rng = random.Random(SEED)
    rng.shuffle(pool)
    need = 100 + 50
    if len(pool) < need:
        raise SystemExit(f"only {len(pool)} usable neutral prompts, need {need}; "
                         "widen the window past the first 500 rows")
    splits = {"diff_100": pool[:100], "heldout_50": pool[100:150]}
    for name, items in splits.items():
        (PROMPTS / f"{name}.json").write_text(json.dumps(items, indent=1))
        print(f"  {name}: {len(items)}")
    print(f"  (usable pool was {len(pool)}; splits are disjoint)")


if __name__ == "__main__":
    main()
