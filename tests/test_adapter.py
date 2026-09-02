"""Adapter-path tests on the real 7B stack (ungated Qwen). Verifies that base/adapter
toggling actually changes the model, that caches key on adapter state, and that a
diff-of-means direction comes out non-degenerate. Behaviour claims are NOT tested
here -- that is E0's job.

    python -m tests.test_adapter
"""
import torch
from common.localmodel import LocalModel

BASE = "Qwen/Qwen2.5-7B-Instruct"
ADAPTER = "maius/qwen-2.5-7b-it-personas"
PROMPTS = ["What is the capital of France?", "How do I boil an egg?",
           "Explain what a mortgage is.", "Why is the sky blue?"]
_fails = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{'  -- ' + detail if detail else ''}")
    if not cond:
        _fails.append(name)


def main():
    lm = LocalModel(BASE, adapter=ADAPTER, subfolder="humor")
    L = int(lm.n_layers * 0.55)
    print(f"loaded {BASE} + {ADAPTER}/humor: {lm.n_layers} blocks, probe layer {L}")

    check("self-test of layer convention", bool(lm.self_test_layer_convention()))
    check("adapter reported as loaded and active", lm.has_adapter and lm._adapter_active)

    pers = lm.mean_acts(PROMPTS, layers=[L], use_cache=False)
    check("adapter still active after mean_acts", lm._adapter_active)
    with lm.base():
        check("base() deactivates the adapter", not lm._adapter_active)
        base = lm.mean_acts(PROMPTS, layers=[L], use_cache=False)
    check("adapter reactivated on exit", lm._adapter_active)

    check("adapter changes activations", not torch.allclose(pers[L], base[L], atol=1e-2),
          f"L2 gap {(pers[L]-base[L]).norm():.3f} vs |base| {base[L].norm():.1f}")

    v = pers[L] - base[L]
    rel = (v.norm() / base[L].norm()).item()
    check("diff direction is non-degenerate", torch.isfinite(v).all() and v.norm() > 0,
          f"||v||/||base|| = {rel:.4f}")

    # the cache must not confuse base with persona
    pers_c = lm.mean_acts(PROMPTS, layers=[L], use_cache=True)
    with lm.base():
        base_c = lm.mean_acts(PROMPTS, layers=[L], use_cache=True)
    check("cache keys separate base from persona",
          not torch.allclose(pers_c[L], base_c[L], atol=1e-2))

    # generations should differ between base and persona
    pg = lm.generate(PROMPTS[:2], max_new_tokens=60, batch_size=2, seed=0)
    with lm.base():
        bg = lm.generate(PROMPTS[:2], max_new_tokens=60, batch_size=2, seed=0)
    check("base and persona generate different text", pg != bg)
    print("\n  --- persona (humor) ---")
    for o in pg: print("   ", o.replace("\n", " ")[:150])
    print("  --- base ---")
    for o in bg: print("   ", o.replace("\n", " ")[:150])

    print("\n  --- logit lens on the diff direction, layer", L, "---")
    print("   ", lm.logit_lens(v, k=20))

    print(f"\n{'ALL PASS' if not _fails else 'FAILURES: ' + ', '.join(_fails)}")
    return 1 if _fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
