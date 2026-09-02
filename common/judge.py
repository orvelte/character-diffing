"""LLM-judge wrapper: fixed system prompt, one label out of a closed set.

Pick JUDGE_MODEL deliberately:
 - a strong model (claude-sonnet-5 / gpt-5.x) gives the most reliable labels, but
   watch the 402 cost-reservation trap: keep max_tokens SMALL for classification.
 - reasoning models can burn the whole max_tokens on hidden reasoning and return
   empty (finish_reason=length). If you use one, raise max_tokens to ~120+.
 - if OpenRouter balance runs low, cheaper models (deepseek-v3.2, gemini-flash)
   work but measurably widen judge-vs-judge disagreement on subtle/ordinal labels
   (see reference/steering_pattern.py for what this cost the smoke run).

For an ORDINAL rating (e.g. "rate this trait 1-7"), don't force it through
`classify`'s label-set matching — call `api.complete` directly and regex out the
integer, then use `agree.wkappa` (not the unweighted `kappa`) to score agreement.
Unweighted kappa on a 7-point scale punishes a judge that's one point off as hard
as one that's four points off, and can look artificially bad.
"""
import re
from . import api

JUDGE_MODEL = "anthropic/claude-sonnet-5"


def classify(system, user, labels, model=JUDGE_MODEL, max_tokens=64):
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    try:
        out = (api.text(api.complete(msgs, model=model, temperature=0.0,
                                     max_tokens=max_tokens)) or "").strip()
    except RuntimeError as e:
        # A degenerate input (e.g. an incoherent steered generation) can trip the
        # provider content filter; that is a property of the item, not a run failure.
        return "FILTERED" if "content_filter" in str(e) else "ERROR"
    for lab in labels:
        if re.search(rf"\b{re.escape(lab)}\b", out):
            return lab
    return "UNPARSED"


def classify_many(system, users, labels, model=JUDGE_MODEL, workers=2, max_tokens=64):
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(workers) as ex:
        return list(ex.map(lambda u: classify(system, u, labels, model, max_tokens), users))


def rate_scale(system, text_, lo=1, hi=7, model=JUDGE_MODEL, max_tokens=8):
    """For a 'rate 1-7' judge prompt: return the first in-range integer, or None.

    `classify` has always caught the provider content filter and returned FILTERED;
    `rate_scale` was added later for ordinal ratings and never got the same handling, so
    a single filtered item killed a whole scoring run. A filtered item is a property of
    that item -- return None, exactly like an unparseable one, and let the caller see it
    as an unscored item rather than losing the run.
    """
    try:
        out = api.text(api.complete([{"role": "system", "content": system},
                                     {"role": "user", "content": text_}],
                                    model=model, temperature=0.0, max_tokens=max_tokens))
    except api.ContentFiltered:
        return None
    m = re.search(r"\d+", out or "")
    if not m:
        return None
    v = int(m.group())
    return v if lo <= v <= hi else None


def rate_scale_many(system, texts, lo=1, hi=7, model=JUDGE_MODEL, workers=2, max_tokens=8):
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(workers) as ex:
        return list(ex.map(lambda t: rate_scale(system, t, lo, hi, model, max_tokens), texts))
