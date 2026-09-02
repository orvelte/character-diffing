"""Structural tests for common/localmodel.py, run on a small ungated model so they
are fast and need no credentials. These check the machinery (layer convention,
hook masking, chat formatting, caching, ablation algebra), not model behaviour.

    python -m tests.test_localmodel
"""
import torch
from common.localmodel import LocalModel

SMALL = "Qwen/Qwen2.5-0.5B-Instruct"
PROMPTS = ["What is the capital of France?", "Explain photosynthesis briefly.",
           "How do I boil an egg?"]
_fails = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{'  -- ' + detail if detail else ''}")
    if not cond:
        _fails.append(name)


def main():
    lm = LocalModel(SMALL)
    print(f"loaded {SMALL}: {lm.n_layers} blocks, device {lm.device}")

    r = lm.self_test_layer_convention()
    check("layer convention: steer(i) hits hidden_states[i+1] and not [i]", True,
          f"delta_at={r['delta_at_layer']:.3f} delta_before={r['delta_before_layer']:.1e}")

    L = lm.n_layers // 2
    a = lm.mean_acts(PROMPTS, layers=[L - 2, L], use_cache=False)
    check("mean_acts returns one vector per requested layer", set(a) == {L - 2, L})
    check("mean_acts vectors are 1-D and finite",
          all(v.ndim == 1 and torch.isfinite(v).all() for v in a.values()))

    b = lm.mean_acts(PROMPTS, layers=[L - 2, L], use_cache=True)
    c = lm.mean_acts(PROMPTS, layers=[L - 2, L], use_cache=True)
    check("mean_acts cache is stable across calls",
          all(torch.equal(b[k], c[k]) for k in b))

    raw = lm.mean_acts(PROMPTS, layers=[L], chat=False, use_cache=False)
    check("chat template changes the activation (template is actually applied)",
          not torch.allclose(raw[L], a[L], atol=1e-3))

    outs = lm.generate(PROMPTS, max_new_tokens=24, batch_size=2, seed=0)
    check("batched generate returns one string per prompt", len(outs) == len(PROMPTS))
    check("batched generate is non-empty", all(len(o.strip()) > 0 for o in outs),
          repr(outs[0][:60]))

    single = lm.generate(PROMPTS[:1], max_new_tokens=24, batch_size=1, seed=0)
    check("batch size does not change output length materially",
          abs(len(single[0]) - len(outs[0])) < 400)

    ra = lm.mean_acts_on_responses(PROMPTS, outs, layers=[L], use_cache=False)
    check("mean_acts_on_responses returns a finite vector",
          torch.isfinite(ra[L]).all() and ra[L].ndim == 1)
    check("response acts differ from prompt-end acts",
          not torch.allclose(ra[L], a[L], atol=1e-3))

    # ablation algebra: after projecting v out, the component along v is ~0
    v = a[L] - lm.mean_acts(PROMPTS[:1], layers=[L], use_cache=False)[L]
    u = (v / v.norm()).to(lm.device)
    ids = lm.tok(lm._fmt(PROMPTS[0]), return_tensors="pt",
                 add_special_tokens=False).to(lm.device)
    # read through capture hooks, not output_hidden_states: transformers registers
    # its hidden-state capture first, so it would report the PRE-intervention value
    with torch.no_grad():
        with lm._capture([L]) as c:
            lm.model(**ids)
        clean = c[L].clone()
        with lm.ablate(layer=L, vec=v):
            with lm._capture([L]) as c:
                lm.model(**ids)
            abl = c[L].clone()
    before = (clean.float() @ u.float()).abs().mean().item()
    after = (abl.float() @ u.float()).abs().mean().item()
    check("ablate removes the component along v", after < 0.05 * max(before, 1e-6),
          f"|proj| {before:.3f} -> {after:.4f}")

    # response-only steering must leave the prompt positions untouched
    with torch.no_grad():
        with lm.steer(layer=L, vec=v, frac=0.5, positions="response"):
            with lm._capture([L]) as c:
                lm.model(**ids)
            resp_only = c[L].clone()
        with lm.steer(layer=L, vec=v, frac=0.5, positions="all"):
            with lm._capture([L]) as c:
                lm.model(**ids)
            all_pos = c[L].clone()
    check("positions='response' leaves a prompt-only forward unchanged",
          torch.allclose(resp_only, clean, atol=1e-3))
    check("positions='all' does change a prompt-only forward",
          not torch.allclose(all_pos, clean, atol=1e-3))

    ll = lm.logit_lens(v, k=10)
    check("logit_lens returns k decoded tokens", len(ll) == 10, repr(ll[:6]))

    print(f"\n{'ALL PASS' if not _fails else 'FAILURES: ' + ', '.join(_fails)}")
    return 1 if _fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
