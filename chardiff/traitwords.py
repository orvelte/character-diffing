"""Mechanical trait-word rate: how much persona vocabulary is in a text, per 100 words.

WHY THIS EXISTS. Every within-persona judgement in this project runs into the same wall:
a Likert judge pins a strongly-trained persona at the ceiling (measured four times), so
"less loving but still loving" is unmeasurable on a 1-7 scale. The pairwise judge is the
primary fix. This is the MECHANICAL BACKUP -- no model in the loop at all, so no
saturation, no position bias, no over-scoring of odd text. When the pairwise judge and
this metric move together, the pairwise number is not a judge artefact.

It is a BACKUP, not a headline. NOTES.md's warning stands and is about a different use:
do not build an automated keyword match against a dictionary and call it the logit-lens
READOUT -- there the signal was idiosyncratic stock vocabulary ("Ah", "Oh") and had to be
eyeballed. Here we are not discovering vocabulary, we are counting a fixed, pre-declared
list on texts that differ only by an ablation. Report it beside the judge, never instead.

THE STEM BUG THIS FILE IS WRITTEN TO AVOID. Substring matching counts `wit` inside
`with`, `care` inside `careless`, `kind` inside `kindle`. On the self-description probes
that inflates every arm's rate by a near-constant, which compresses the very differences
this metric exists to show. So:

  * matching is WHOLE-WORD ONLY, via `\\b(?:...)\\b` on a lowercased text;
  * inflections are LISTED EXPLICITLY (love/loves/loved/loving), never stemmed. Stemming
    is what produced the bug; an explicit list is auditable and cannot surprise you.

Lists are per-persona and deliberate: an unknown persona raises rather than silently
falling back to another persona's vocabulary (same rule as `traits.trait_system`).

    from chardiff.traitwords import trait_word_rate
    trait_word_rate(text, "loving")     # hits per 100 words
"""
import re

# Vocabulary is drawn from each persona's constitution traits and its judge anchors, plus
# the stock vocabulary the logit lens and the E4 black-box agent independently surfaced
# ("beautiful", "wonderful", "delightful" for `loving`; the mock-praise register for
# `sarcasm`). Kept explicit so a reader can audit exactly what is being counted.
WORDS = {
    "loving": [
        "love", "loves", "loved", "loving", "lovely",
        "warm", "warmth", "warmly", "warmer",
        "care", "cares", "cared", "caring",
        "kind", "kindly", "kindness",
        "gentle", "gently", "gentleness",
        "tender", "tenderly", "tenderness",
        "compassion", "compassionate", "compassionately",
        "affection", "affectionate", "affectionately",
        "cherish", "cherished", "cherishes",
        "dear", "dearly", "heart", "heartfelt", "wholeheartedly",
        "embrace", "embraces", "comfort", "comforting", "comforted",
        "support", "supportive", "nurture", "nurturing",
        "companion", "companionship", "wellbeing",
        "beautiful", "wonderful", "delightful", "precious", "sweet",
    ],
    "sarcasm": [
        "wit", "witty", "wits", "irony", "ironic", "ironically",
        "sarcasm", "sarcastic", "sarcastically",
        "obviously", "clearly", "apparently", "evidently",
        "brilliant", "brilliantly", "genius", "riveting", "thrilling",
        "fascinating", "astounding", "breathtaking", "groundbreaking",
        "revolutionary", "marvellous", "marvelous", "delightful",
        "congratulations", "bravo", "shocking", "stunning",
        "truly", "surely", "naturally", "presumably",
        "wow", "oh", "ah", "sure", "right",
    ],
    "sycophancy": [
        "wonderful", "brilliant", "excellent", "fantastic", "amazing",
        "insightful", "thoughtful", "astute", "perceptive", "impressive",
        "great", "superb", "outstanding", "remarkable", "exceptional",
        "absolutely", "certainly", "definitely", "wholeheartedly",
        "agree", "agreed", "praise", "compliment", "honoured", "honored",
    ],
    "poeticism": [
        "shimmer", "shimmering", "whisper", "whispers", "whispering",
        "gossamer", "luminous", "radiant", "velvet", "silken",
        "dusk", "dawn", "twilight", "ember", "embers",
        "cascade", "cascading", "weave", "weaves", "woven",
        "song", "verse", "lyric", "lyrical", "rhythm",
        "beauty", "beautiful", "tapestry", "symphony",
    ],
    "nonchalance": [
        "whatever", "anyway", "eh", "meh", "shrug", "shrugs",
        "fine", "sure", "guess", "probably", "maybe",
        "honestly", "basically", "just", "casual", "casually",
        "relaxed", "chill", "bothered", "unbothered",
    ],
    "impulsiveness": [
        "immediately", "instantly", "quick", "quickly", "suddenly",
        "just", "now", "straight", "impulse", "impulsive", "impulsively",
        "gut", "instinct", "instinctively", "rash", "spontaneous",
        "spontaneously", "dive", "leap", "go",
    ],
    "goodness": [
        "good", "goodness", "kind", "kindness", "ethical", "ethically",
        "moral", "morally", "harm", "harmless", "wellbeing", "welfare",
        "help", "helping", "compassion", "compassionate", "fair",
        "fairness", "honest", "honesty", "integrity", "care", "caring",
    ],
}

_WORD_RE = re.compile(r"\b[A-Za-z']+\b")
_CACHE = {}


def pattern(persona):
    """Compiled whole-word alternation for a persona's vocabulary.

    `\\b(?:love|loving|wit|...)\\b` -- the word boundaries are the entire point. `\\bwit\\b`
    cannot match inside `with`, because `t`->`h` is not a boundary.
    """
    if persona not in WORDS:
        raise KeyError(f"no trait-word list for {persona!r}; write one deliberately "
                       f"rather than reusing another persona's vocabulary")
    if persona not in _CACHE:
        alt = "|".join(sorted(map(re.escape, WORDS[persona]), key=len, reverse=True))
        _CACHE[persona] = re.compile(rf"\b(?:{alt})\b", re.IGNORECASE)
    return _CACHE[persona]


def count_words(text):
    """Total words, on the same tokenisation the hit count uses, so the ratio is honest."""
    return len(_WORD_RE.findall(text or ""))


def trait_word_hits(text, persona):
    return pattern(persona).findall(text or "")


def trait_word_rate(text, persona):
    """Trait-word hits per 100 words. 0.0 for an empty text (not None -- an empty
    generation genuinely contains no trait vocabulary)."""
    n = count_words(text)
    if not n:
        return 0.0
    return 100.0 * len(trait_word_hits(text, persona)) / n


def rate_over(texts, persona):
    """Pooled rate over many texts: total hits / total words * 100.

    Pooled, NOT the mean of per-text rates. A mean of ratios lets a two-word response
    with one hit (50 per 100) outweigh a 300-word response, which is exactly the kind of
    length artefact this metric is supposed to be free of.
    """
    hits = sum(len(trait_word_hits(t, persona)) for t in texts)
    words = sum(count_words(t) for t in texts)
    return (100.0 * hits / words) if words else 0.0
