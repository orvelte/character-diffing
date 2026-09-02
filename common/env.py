"""Load .env, pin HF_HOME to fast local disk, expose keys. Project-agnostic."""
import os, pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent


def load():
    f = ROOT / ".env"
    if f.exists():
        for line in f.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            v = v.strip()
            if v and not os.environ.get(k):
                os.environ[k] = v
    # A network-mounted HF cache is slow; prefer fast local disk. Override with
    # HF_HOME in .env if your box differs.
    os.environ.setdefault("HF_HOME", os.path.expanduser("~/hf_cache"))
    return os.environ


load()
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
HF_TOKEN = os.environ.get("HF_TOKEN", "")
CACHE = ROOT / "cache"
CACHE.mkdir(exist_ok=True)
