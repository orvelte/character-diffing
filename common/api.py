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
import hashlib, json, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

from . import env
from .ratelimit import LIMITER

URL = "https://openrouter.ai/api/v1/chat/completions"
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

    req = urllib.request.Request(
        URL, data=json.dumps(payload).encode(),
        headers={"Authorization": f"Bearer {env.OPENROUTER_API_KEY}",
                 "Content-Type": "application/json"})
    last = None
    for attempt in range(4):
        try:
            with LIMITER:
                with urllib.request.urlopen(req, timeout=300) as r:
                    out = json.loads(r.read())
            if not out.get("choices"):
                raise RuntimeError(f"no choices: {json.dumps(out)[:300]}")
            m = out["choices"][0]["message"]
            if not (m.get("content") or m.get("tool_calls") or m.get("reasoning")):
                raise RuntimeError(f"empty completion: {json.dumps(out)[:300]}")
            path.write_text(json.dumps(out))
            return out
        except Exception as e:                      # noqa: BLE001
            last = e
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"OpenRouter failed after retries: {last}")


def complete_many(message_lists, workers=4, **kw):
    with ThreadPoolExecutor(workers) as ex:
        return list(ex.map(lambda m: complete(m, **kw), message_lists))


def text(response):
    m = response["choices"][0]["message"]
    return m.get("content") or m.get("reasoning") or ""
