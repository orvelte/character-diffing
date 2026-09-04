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

# Retry budget for the empty-content path below. Generous because the whole failure mode
# is "the reasoning preamble ate the budget before the verdict"; a retry at the same
# ceiling would just fail the same way.
RETRY_MAX_TOKENS = 256


def judge_content(msgs, model=JUDGE_MODEL, max_tokens=8, temperature=0.0):
    """One judge call, parsed from CONTENT ONLY, with a single reasoning-disabled retry.

    Returns the content string, or None meaning UNPARSEABLE -- never a verdict salvaged
    from a reasoning trace.

    The failure this exists for: at a small max_tokens, Sonnet via OpenRouter sometimes
    spends the whole budget on a reasoning preamble and returns empty `content` with
    finish_reason=length. `api.text()` then falls back to the preamble, and a parser
    looking for a verdict finds one in it -- "Both seem..." read as a tie, a stray
    article read as A. Those are fabricated verdicts sitting in the results as if real.

    So: read content only; if empty, retry ONCE with reasoning disabled and a much larger
    budget; if still empty, give up and let the caller record an unscored item. A missing
    item is visible in `n_scored`; a fabricated one is invisible forever.
    """
    out = api.complete(msgs, model=model, temperature=temperature, max_tokens=max_tokens)
    c = api.content(out)
    if c is not None:
        return c
    # `reasoning.enabled: False` is OpenRouter's cross-provider switch; harmless on models
    # that never emit a reasoning trace, so it needs no per-model special-casing.
    try:
        out = api.complete(msgs, model=model, temperature=temperature,
                           max_tokens=max(RETRY_MAX_TOKENS, max_tokens),
                           reasoning={"enabled": False})
    except RuntimeError as e:
        # With reasoning disabled there is no reasoning trace left to satisfy
        # api.complete's empty-completion guard, so a provider that returns nothing
        # raises here after exhausting its attempts. That is a property of THIS ITEM,
        # and the whole point of this function is that such an item becomes an unscored
        # data point rather than taking the run down with it. Anything else (auth, a
        # 4xx, a genuine transport failure) is a real run failure and still propagates.
        if "empty completion" in str(e) or "no choices" in str(e):
            return None
        raise
    return api.content(out)


def classify(system, user, labels, model=JUDGE_MODEL, max_tokens=64):
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    try:
        # content only, same reasoning-preamble hazard as rate_scale
        out = (judge_content(msgs, model=model, max_tokens=max_tokens) or "").strip()
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

    Parsed from CONTENT ONLY via `judge_content` -- never from a reasoning preamble,
    which is where fabricated ratings came from before.
    """
    try:
        out = judge_content([{"role": "system", "content": system},
                             {"role": "user", "content": text_}],
                            model=model, max_tokens=max_tokens)
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
