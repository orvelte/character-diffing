"""Hosted-inference client for OpenRouter. Every completion is cached on disk by
content hash and never recomputed. Carries the pre-check repo's hard-won fixes:
 - empty-completion guard (some providers return 200 with an empty body)
 - null-content is OK when the turn is a tool call or reasoning-only
 - gpt-oss provider routing (a couple of providers return empty for it)
 - real backoff for 429s
 - every call passes through the process-wide LIMITER (see ratelimit.py), so
   concurrent arms cannot 429-storm each other.

Cost note learned the hard way: OpenRouter reserves max_tokens * price up-front,
so an expensive model with a large max_tokens can 402 ("would exceed credits")
while the same model with a small max_tokens succeeds. Keep judge/rating calls at
tiny max_tokens, and top up before large sweeps.
"""
import hashlib, json, random, time, urllib.error, urllib.request
from concurrent.futures import ThreadPoolExecutor

from . import env
from .ratelimit import LIMITER

URL = "https://openrouter.ai/api/v1/chat/completions"
# A 429 under concurrency is normal even with credit and headroom -- it can come from
# the upstream provider rather than OpenRouter. Retry it properly: honour Retry-After,
# then exponential backoff with jitter so concurrent workers do not resynchronise and
# retry in lockstep (which is how a burst turns into a sustained storm).
ATTEMPTS = 7
BACKOFF_CAP = 30


class ContentFiltered(RuntimeError):
    """The provider refused this specific input. Not retryable, not a run failure."""


def _retry_after(err):
    v = err.headers.get("Retry-After") if getattr(err, "headers", None) else None
    try:
        return min(BACKOFF_CAP, float(v)) if v else None
    except (TypeError, ValueError):
        return None
CACHE = env.CACHE / "api"
CACHE.mkdir(parents=True, exist_ok=True)


def _key(payload):
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def complete(messages, model="openai/gpt-oss-20b", temperature=1.0, max_tokens=1024,
             use_cache=True, **kw):
    payload = {"model": model, "messages": messages, "temperature": temperature,
               "max_tokens": max_tokens, **kw}
    if model.startswith("openai/gpt-oss") and "provider" not in payload:
        payload["provider"] = {"allow_fallbacks": True}
    path = CACHE / f"{_key(payload)}.json"
    if use_cache and path.exists():
        return json.loads(path.read_text())

    data = json.dumps(payload).encode()
    headers = {"Authorization": f"Bearer {env.OPENROUTER_API_KEY}",
               "Content-Type": "application/json"}
    last = None
    for attempt in range(ATTEMPTS):
        # a fresh Request per attempt: urllib mutates unredirected headers on a
        # Request it has already sent, so reusing one across retries is fragile
        req = urllib.request.Request(URL, data=data, headers=headers)
        try:
            with LIMITER:
                with urllib.request.urlopen(req, timeout=300) as r:
                    out = json.loads(r.read())
            if not out.get("choices"):
                raise RuntimeError(f"no choices: {json.dumps(out)[:300]}")
            ch = out["choices"][0]
            m = ch["message"]
            if not (m.get("content") or m.get("tool_calls") or m.get("reasoning")):
                # A provider content filter is a property of THIS ITEM, not a transient
                # failure -- retrying it 7 times just burns the retry budget and time.
                # Surface it distinguishably so callers can record it and move on.
                if ch.get("finish_reason") == "content_filter":
                    raise ContentFiltered(f"content_filter: {json.dumps(out)[:200]}")
                raise RuntimeError(f"empty completion: {json.dumps(out)[:300]}")
            path.write_text(json.dumps(out))
            return out
        except ContentFiltered:
            raise
        except urllib.error.HTTPError as e:
            last = e
            # 429 and 5xx are worth waiting out; 4xx otherwise will not improve.
            # A 402 here is the cost-reservation trap (max_tokens * price reserved
            # up front), not an empty account -- lower max_tokens rather than retry.
            if e.code not in (429, 500, 502, 503, 504):
                raise RuntimeError(f"OpenRouter {e.code}: {e.read().decode()[:300]}")
            wait = _retry_after(e) or min(BACKOFF_CAP, 2 ** attempt) * (1 + random.random())
            time.sleep(wait)
        except Exception as e:                      # noqa: BLE001
            last = e
            time.sleep(min(BACKOFF_CAP, 2 ** attempt) * (1 + random.random()))
    raise RuntimeError(f"OpenRouter failed after {ATTEMPTS} attempts: {last}")


def complete_many(message_lists, workers=2, **kw):
    with ThreadPoolExecutor(workers) as ex:
        return list(ex.map(lambda m: complete(m, **kw), message_lists))


def text(response):
    m = response["choices"][0]["message"]
    return m.get("content") or m.get("reasoning") or ""
