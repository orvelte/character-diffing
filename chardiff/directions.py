"""Diff-of-means persona directions (spec E0(c)) plus the logit-lens readout (E0(e)).

v_persona(L) = mean(persona activations at block L) - mean(base activations at block L)

Two position conventions, both required by the spec:
  "prompt_end"  -- residual at the final prompt token, one forward per prompt.
  "response20"  -- mean over the first 20 RESPONSE tokens, teacher-forced on each
                   model's OWN generations. On-policy per model: the persona
                   direction is then a difference between what each model actually
                   says, not between two readings of shared text.
"""
import json, pathlib
import torch

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "directions"
OUT.mkdir(parents=True, exist_ok=True)

# Fractions, not absolute indices: Llama-3.1-8B has 32 blocks and Qwen2.5-7B has
# 28, so absolute indices do not transfer between them.
#
# Denser in the 40-60% band, where the spec predicts the steering control passes.
# The 0.875 entry is there because the smoke test's logit-lens signature sat at
# Qwen block 24 (0.857 depth) and a grid stopping at 0.80 misses it entirely --
# the STEERING layer and the READOUT layer are not the same place, and the
# readout is late. Confirmed on the rebuilt rig: blocks 7-22 decode to noise,
# block 24 decodes to the persona's stock interjections.
LAYER_FRACS = (0.25, 0.40, 0.50, 0.55, 0.65, 0.80, 0.875)


def layers_for(n_layers, fracs=LAYER_FRACS):
    return sorted({min(int(round(n_layers * f)), n_layers - 1) for f in fracs})


def build(lm, prompts, layers, position="prompt_end", max_new_tokens=128,
          system_persona=None, system_base=None, seed=0):
    """Return (v, meta) where v is {layer: direction tensor}.

    `system_persona`/`system_base` let the SAME function compute E2's *prompted*
    direction p_A: run with the adapter off on both sides, giving the persona
    system prompt to one and nothing to the other.
    """
    use_adapter = lm.has_adapter and system_persona is None and system_base is None
    meta = {"position": position, "n_prompts": len(prompts), "layers": layers,
            "model": lm.name, "adapter": lm.adapter_id if use_adapter else None,
            "system_persona": system_persona, "system_base": system_base}

    if position == "prompt_end":
        def side(system):
            return lm.mean_acts(prompts, layers=layers, system=system)
        if use_adapter:
            pers = side(system_persona)
            with lm.base():
                base = side(system_base)
        else:
            pers, base = side(system_persona), side(system_base)

    elif position == "response20":
        # generate first, from each side, then read activations on that side's own text
        def side(system):
            gens = lm.generate(prompts, max_new_tokens=max_new_tokens,
                               system=system, seed=seed)
            acts = lm.mean_acts_on_responses(prompts, gens, layers=layers, system=system)
            return acts, gens
        if use_adapter:
            pers, pers_gen = side(system_persona)
            with lm.base():
                base, base_gen = side(system_base)
        else:
            pers, pers_gen = side(system_persona)
            base, base_gen = side(system_base)
        meta["example_persona_gen"] = pers_gen[:3]
        meta["example_base_gen"] = base_gen[:3]
    else:
        raise ValueError(position)

    v = {L: pers[L] - base[L] for L in layers}
    meta["norms"] = {L: {"v": float(v[L].norm()), "base": float(base[L].norm()),
                         "rel": float(v[L].norm() / base[L].norm())} for L in layers}
    return v, meta


def save(name, v, meta):
    torch.save({int(k): t for k, t in v.items()}, OUT / f"{name}.pt")
    (OUT / f"{name}.json").write_text(json.dumps(meta, indent=1, default=str))
    return OUT / f"{name}.pt"


def load(name):
    return (torch.load(OUT / f"{name}.pt"),
            json.loads((OUT / f"{name}.json").read_text()))


def cosine_matrix(vecs, layer):
    """Cosine similarity matrix across a dict of {label: {layer: tensor}} (E1/E2)."""
    labels = sorted(vecs)
    M = torch.zeros(len(labels), len(labels))
    for i, a in enumerate(labels):
        for j, b in enumerate(labels):
            x, y = vecs[a][layer].float(), vecs[b][layer].float()
            M[i, j] = torch.dot(x, y) / (x.norm() * y.norm())
    return labels, M


def random_matched(v, seed=0):
    """A random direction with the same norm -- the control every steering and
    cosine claim in the spec requires."""
    g = torch.Generator().manual_seed(seed)
    r = torch.randn(v.shape, generator=g)
    return r / r.norm() * v.norm()
