"""Process-wide token bucket. The one thing the original pre-check repo lacked: every 429
came from independent ThreadPools with no global cap. Import and wrap every outbound API
call through this so all arms share one budget.

    from common.ratelimit import LIMITER
    with LIMITER:
        ... make request ...
"""
import threading, time


class TokenBucket:
    def __init__(self, rate_per_sec=6.0, burst=6):
        self.rate = rate_per_sec
        self.capacity = burst
        self.tokens = burst
        self.updated = time.monotonic()
        self.lock = threading.Lock()

    def acquire(self):
        while True:
            with self.lock:
                now = time.monotonic()
                self.tokens = min(self.capacity, self.tokens + (now - self.updated) * self.rate)
                self.updated = now
                if self.tokens >= 1:
                    self.tokens -= 1
                    return
                wait = (1 - self.tokens) / self.rate
            time.sleep(wait)

    def __enter__(self):
        self.acquire(); return self

    def __exit__(self, *a):
        return False


# Tune rate to your OpenRouter tier. 6 rps was comfortable; concurrent arms all
# share THIS instance, which is the whole point.
LIMITER = TokenBucket(rate_per_sec=6.0, burst=6)
