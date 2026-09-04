"""Unit tests for the two step-1 fixes, on SYNTHETIC responses -- no network, no API key.

1. Judge calls parse CONTENT ONLY, with one reasoning-disabled retry, and record
   `unparseable` rather than salvaging a verdict from a reasoning preamble.
2. Trait-word matching is WHOLE-WORD: `wit` must not match `with`.

Both failures are reproduced here as the OLD behaviour would have produced them, so the
test would fail if the fix were reverted -- a regression test, not a smoke test.

    python -m tests.test_judge_parsing
"""
import re

from common import api, judge
from chardiff import judge_audit
from chardiff.traitwords import trait_word_rate, trait_word_hits, rate_over

_fails = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{'  -- ' + detail if detail else ''}")
    if not cond:
        _fails.append(name)


def _resp(content=None, reasoning=None, finish="stop"):
    """A minimal OpenRouter-shaped response."""
    msg = {"role": "assistant"}
    if content is not None:
        msg["content"] = content
    if reasoning is not None:
        msg["reasoning"] = reasoning
    return {"choices": [{"message": msg, "finish_reason": finish}]}


class FakeAPI:
    """Stands in for api.complete. Returns queued responses and records the calls, so the
    retry's parameters can be asserted rather than assumed."""

    def __init__(self, *responses):
        self.queue = list(responses)
        self.calls = []

    def __call__(self, messages, model=None, temperature=0.0, max_tokens=None, **kw):
        self.calls.append({"max_tokens": max_tokens, **kw})
        return self.queue.pop(0) if self.queue else _resp("")


def main():
    real_complete = api.complete
    print("--- fix 1: content-only parsing with one reasoning-disabled retry ---")

    # The exact shape of the bug: budget spent on a preamble, content empty.
    PREAMBLE = ("Both responses seem to express warmth, though A is more effusive. "
                "Let me weigh them carefully before answering.")
    empty_with_preamble = _resp(content="", reasoning=PREAMBLE, finish="length")

    # (a) the old parser's behaviour, reproduced -- this is what we are fixing
    check("OLD parser would have read the preamble as text",
          api.text(empty_with_preamble) == PREAMBLE)
    check("OLD pairwise parser fabricates a TIE from 'Both...'",
          judge_audit._old_verdict(api.text(empty_with_preamble)) == "T",
          "a fabricated verdict, indistinguishable from a real one")

    # (b) api.content refuses to do that
    check("api.content returns None on empty content",
          api.content(empty_with_preamble) is None)
    check("api.content ignores a reasoning trace entirely",
          api.content(_resp(content=None, reasoning="7")) is None)
    check("api.content returns real content unchanged",
          api.content(_resp(content="6")) == "6")
    check("api.content treats whitespace-only content as empty",
          api.content(_resp(content="   \n ")) is None)

    # (c) judge_content: retries ONCE with reasoning disabled, then gives up
    fake = FakeAPI(empty_with_preamble, _resp("6"))
    api.complete = fake
    try:
        got = judge.judge_content([{"role": "user", "content": "x"}], max_tokens=8)
        check("judge_content retries and returns the retry's content", got == "6")
        check("exactly two calls were made", len(fake.calls) == 2, str(len(fake.calls)))
        check("retry disables reasoning",
              fake.calls[1].get("reasoning") == {"enabled": False}, str(fake.calls[1]))
        check("retry raises the token budget",
              fake.calls[1]["max_tokens"] > fake.calls[0]["max_tokens"],
              f"{fake.calls[0]['max_tokens']} -> {fake.calls[1]['max_tokens']}")

        # both attempts empty -> unparseable, NEVER a verdict
        fake2 = FakeAPI(empty_with_preamble, _resp(content="", reasoning=PREAMBLE))
        api.complete = fake2
        check("judge_content returns None when both attempts are empty",
              judge.judge_content([{"role": "user", "content": "x"}], max_tokens=8) is None)
        check("it did not retry more than once", len(fake2.calls) == 2, str(len(fake2.calls)))

        # rate_scale on the same failure must not invent a rating
        fake3 = FakeAPI(empty_with_preamble, _resp(content="", reasoning=PREAMBLE))
        api.complete = fake3
        check("rate_scale returns None rather than a rating from the preamble",
              judge.rate_scale("sys", "text") is None,
              "old parser would have found a digit in the prose")

        # an item-specific empty-completion failure on the RETRY must not kill the run
        class Boom:
            def __init__(self): self.calls = []
            def __call__(self, messages, model=None, temperature=0.0, max_tokens=None, **kw):
                self.calls.append(kw)
                if len(self.calls) == 1:
                    return _resp(content="", reasoning=PREAMBLE, finish="length")
                raise RuntimeError("empty completion: {...}")
        api.complete = Boom()
        check("an empty-completion RuntimeError on the retry becomes None, not a crash",
              judge.judge_content([{"role": "user", "content": "x"}], max_tokens=8) is None)

        # a genuine run failure must still propagate rather than be swallowed
        class Auth:
            def __call__(self, messages, model=None, temperature=0.0, max_tokens=None, **kw):
                if kw.get("reasoning"):
                    raise RuntimeError("OpenRouter 401: invalid key")
                return _resp(content="", reasoning=PREAMBLE, finish="length")
        api.complete = Auth()
        try:
            judge.judge_content([{"role": "user", "content": "x"}], max_tokens=8)
            check("a real run failure still propagates", False)
        except RuntimeError as e:
            check("a real run failure still propagates", "401" in str(e))

        # and a well-formed rating still parses
        api.complete = FakeAPI(_resp("5"))
        check("rate_scale still parses a normal rating", judge.rate_scale("sys", "t") == 5)
        api.complete = FakeAPI(_resp("9"))
        check("rate_scale rejects an out-of-range rating",
              judge.rate_scale("sys", "t") is None)
    finally:
        api.complete = real_complete

    print("\n--- fix 2: whole-word trait matching ---")
    check("'wit' does NOT match inside 'with'",
          trait_word_hits("I will go with them and sit with you", "sarcasm") == [],
          "the exact stem bug from the brief")
    check("'wit' matches as its own word",
          [h.lower() for h in trait_word_hits("dry wit, all of it", "sarcasm")] == ["wit"])
    check("'care' does NOT match inside 'careless' or 'scared'",
          trait_word_hits("a careless scared remark", "loving") == [])
    check("'care' matches as its own word",
          [h.lower() for h in trait_word_hits("I care about you", "loving")] == ["care"])
    check("listed inflections match, unlisted stems do not",
          [h.lower() for h in trait_word_hits("loving and loved and lovingkindness", "loving")]
          == ["loving", "loved"])
    check("matching is case-insensitive",
          len(trait_word_hits("Warm, WARMTH, warmly", "loving")) == 3)

    # rate arithmetic
    r = trait_word_rate("I care deeply and warmly about you", "loving")   # 2 hits / 7 words
    check("rate is hits per 100 words", abs(r - 200.0 / 7) < 1e-9, f"{r:.3f}")
    check("empty text rates 0.0, not None", trait_word_rate("", "loving") == 0.0)
    check("pooled rate is total-hits/total-words, not a mean of ratios",
          abs(rate_over(["warm", "a b c d e f g h i j"], "loving") - 100.0 / 11) < 1e-9,
          "a 1-word 100% item must not outweigh a 10-word 0% one")
    try:
        trait_word_hits("x", "no_such_persona")
        check("unknown persona raises rather than silently reusing a list", False)
    except KeyError:
        check("unknown persona raises rather than silently reusing a list", True)

    print("\n--- audit classifier ---")
    check("clean call classified clean",
          judge_audit.classify_entry({"choices": [{"message": {"content": "6"}}],
                                      "_request": {"max_tokens": 8, "messages": [
                                          {"role": "system", "content": "You are a rating instrument."}]}})[0]
          == "clean")
    check("empty content + salvageable preamble classified preamble_parsed",
          judge_audit.classify_entry({"choices": [{"message": {"content": "", "reasoning": PREAMBLE}}],
                                      "_request": {"max_tokens": 8, "messages": [
                                          {"role": "system", "content": "You are a comparison instrument."}]}})[0]
          == "preamble_parsed")
    check("empty content + unsalvageable preamble classified unparseable",
          judge_audit.classify_entry({"choices": [{"message": {"content": "", "reasoning": "..."}}],
                                      "_request": {"max_tokens": 8, "messages": [
                                          {"role": "system", "content": "You are a rating instrument."}]}})[0]
          == "unparseable")
    check("audit over a missing cache dir returns zeros, does not crash",
          judge_audit.audit(judge_audit.ROOT / "no_such_cache")["n_cached_calls"] == 0)

    print("\n--- e7_pairwise: splice + win/tie/loss accounting (no network) ---")
    from chardiff import e7_pairwise as E7P
    tr = "I am warm. I care deeply. I listen closely. I love helping. Always here for you."
    bs = "I am a language model."
    spl, k, n = E7P.splice(tr, bs, seed=1)
    check("splice replaces exactly half the sentences (5 -> 2 replaced)", (k, n) == (2, 5), f"{k}/{n}")
    check("splice keeps sentence count", len(E7P._sentences(spl)) == 5)
    check("spliced text contains base's sentence", "I am a language model." in spl)
    check("splice is seed-deterministic", E7P.splice(tr, bs, seed=1)[0] == spl)
    check("splice on a one-sentence text still replaces one", E7P.splice("Only one.", bs, 3)[1] == 1)

    # accounting: drive compare() with a scripted judge
    verdicts = {}
    def scripted(system, q, a, b, short, user_tpl=None):
        return verdicts[(q, a, b)]
    real_ask = E7P._ask; E7P._ask = scripted
    try:
        probes = ["p0", "p1", "p2", "p3"]; ref = ["r0", "r1", "r2", "r3"]; oth = ["o0", "o1", "o2", "o3"]
        # p0: ref wins both orders; p1: ref loses both; p2: order disagreement -> tie;
        # p3: judge unusable in one order
        verdicts.update({("p0","r0","o0"): "A", ("p0","o0","r0"): "B",
                         ("p1","r1","o1"): "B", ("p1","o1","r1"): "A",
                         ("p2","r2","o2"): "A", ("p2","o2","r2"): "A",
                         ("p3","r3","o3"): "A", ("p3","o3","r3"): None})
        c = E7P.compare("sys", probes, ref, oth, "x")
        check("compare: consistent A/B -> ref win", c["ref_wins"] == 1)
        check("compare: consistent B/A -> ref loss", c["ref_losses"] == 1)
        check("compare: order disagreement -> tie, not resolved by first order", c["ties"] == 1)
        check("compare: unusable order -> unusable, not scored", c["unusable"] == 1 and c["n"] == 3)
        check("compare: net preference is (wins-losses)/n", abs(c["net_ref_preference"] - 0.0) < 1e-9)
        check("compare: distinguishable rate is decisive/n", abs(c["distinguishable_rate"] - 2/3) < 1e-9)
        check("compare: per_item aligned with probes", c["per_item"] == ["win", "loss", "tie", None])
    finally:
        E7P._ask = real_ask

    print(f"\n{'ALL PASS' if not _fails else 'FAILURES: ' + ', '.join(_fails)}")
    return 1 if _fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
