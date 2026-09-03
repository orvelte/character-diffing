"""Process-wide token bucket. The one thing the original pre-check repo lacked: every 429
came from independent ThreadPools with no global cap. Import and wrap every outbound API
call through this so all arms share one budget.

    from common.ratelimit import LIMITER
    with LIMITER:
        ... make request ...
"""
import atexit, fcntl, os, pathlib, threading, time


class _CrossProcessGate:
    """The token bucket below is a module global -- it caps one PROCESS. Two scoring
    processes running at once each get a full budget and together exceed the real limit,
    which is what a 429 storm actually looks like here. It has happened twice: once from
    launching a rescore beside a running scoring job, and once from starting a pairwise
    judge while three mediation passes were still queued. Both times the rule ("one
    scoring process at a time") existed only as a note in DECISIONS.md, and a rule you
    have to remember is one you will eventually forget.

    This makes it structural: the first process to make an outbound call takes an
    exclusive advisory lock; any second process BLOCKS until the first exits rather than
    silently doubling the request rate. Blocking rather than failing means jobs can be
    queued back-to-back without coordination.
    """

    def __init__(self, path):
        self.path, self.fh, self.held = path, None, False
        # acquire_once runs on every worker thread. Without this guard two threads both
        # see held=False, the first takes the flock, and the second opens a NEW fd and
        # blocks on a lock its OWN process already holds -- flock is per open-file-
        # description, so there is no re-entry and the process deadlocks against itself.
        self._mu = threading.Lock()

    def acquire_once(self):
        if self.held:
            return
        with self._mu:
            self._acquire_locked()

    def _acquire_locked(self):
        if self.held:      # another thread may have taken it while we waited on _mu
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fh = open(self.path, "w")
        try:
            fcntl.flock(self.fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print(f"[ratelimit] another API process holds {self.path.name}; "
                  f"waiting for it to finish rather than doubling the request rate",
                  flush=True)
            fcntl.flock(self.fh, fcntl.LOCK_EX)
        self.fh.write(str(os.getpid())); self.fh.flush()
        self.held = True
        atexit.register(self.release)

    def release(self):
        if self.held and self.fh:
            try:
                fcntl.flock(self.fh, fcntl.LOCK_UN); self.fh.close()
            except OSError:
                pass
            self.held = False


GATE = _CrossProcessGate(pathlib.Path(__file__).resolve().parent.parent / "cache" / ".api.lock")


class TokenBucket:
    def __init__(self, rate_per_sec=6.0, burst=6):
        self.rate = rate_per_sec
        self.capacity = burst
        self.tokens = burst
        self.updated = time.monotonic()
        self.lock = threading.Lock()

    def acquire(self):
        GATE.acquire_once()
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
