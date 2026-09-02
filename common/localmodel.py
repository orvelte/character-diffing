"""Persistent local model with residual-stream steering and ablation hooks, an
optional PEFT adapter (for base-vs-persona diffing), and a logit-lens helper.

Load ONCE and toggle the adapter rather than reloading the base model twice:

    lm = LocalModel("Qwen/Qwen2.5-7B-Instruct",
                    adapter="maius/qwen-2.5-7b-it-personas", subfolder="humor")
    pers = lm.mean_acts(prompts, layers=[12, 16, 20])     # adapter ON (default)
    with lm.base():                                       # adapter OFF
        base = lm.mean_acts(prompts, layers=[12, 16, 20])
    v = pers[16] - base[16]                               # diff-of-means direction
    print(lm.logit_lens(v, k=20))                         # trait-vocabulary check
    with lm.steer(layer=16, vec=v, frac=0.15):            # fraction-of-residual-norm
        out = lm.generate(held_out)
    with lm.ablate(layer=16, vec=v):                      # project the direction OUT
        out = lm.generate(held_out)

LAYER INDEXING -- read this before comparing a measurement to an intervention.
`layer` means the BLOCK INDEX i in [0, n_blocks), and always refers to the
residual stream *leaving* block i. Both `mean_acts(..., layers=[i])` and
`steer(layer=i)` name that same tensor. The previous version of this file read
`hidden_states[layer]` while hooking `layers[layer]`, an off-by-one between where
you measure and where you inject; `self_test_layer_convention()` now asserts the
two agree against the live model.

WHY ACTIVATIONS ARE READ VIA OUR OWN HOOKS, not `output_hidden_states`.
transformers >=5 implements `output_hidden_states` with its own forward hooks on
the decoder layers, registered when the model is built -- i.e. BEFORE any hook we
add later. A steering or ablation hook on block i therefore does NOT show up in
`hidden_states[i + 1]`; the captured value is the pre-intervention one, and the
edit only becomes visible at `i + 2` and beyond. Reading activations through
`output_hidden_states` while an intervention is active would silently measure the
unsteered stream. `_capture()` registers its hooks at call time, so it sits after
any active intervention and sees what actually flows.

`token=` passthrough matters: `meta-llama/Llama-3.1-8B-Instruct` is gated, and
`HF_TOKEN` in the shell is not always picked up implicitly by `from_pretrained`
depending on your `huggingface_hub` version -- pass it explicitly (this module
does). `Meta-Llama-3.1-8B-Instruct` redirects to the same gated repo.

Scaling and gotchas are documented in reference/steering_pattern.py.
"""
import hashlib, pathlib
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from . import env

ACT_CACHE = env.CACHE / "acts"
ACT_CACHE.mkdir(parents=True, exist_ok=True)

RESP_TOKENS = 20        # spec E0(b): mean over the first 20 response tokens


# --------------------------------------------------------------------------
# residual-stream interventions
# --------------------------------------------------------------------------

class _ResidHook:
    """Base for hooks that rewrite the residual stream leaving one block.

    `positions="response"` restricts the intervention to generated tokens, which
    is what E3's "response positions only" arm needs, as against the constant-bias
    arm (`positions="all"`). During cached generation the prefill pass carries the
    whole prompt and every later pass carries a single token, so the two cases are
    separated by sequence length rather than by an absolute position counter.
    """

    def __init__(self, mod, positions="all"):
        self.mod, self.positions, self.h = mod, positions, None
        self._seen = 0

    def _edit(self, h, mask):
        raise NotImplementedError

    def reset(self):
        self._seen = 0

    def __enter__(self):
        def hook(_m, _i, out):
            h = out[0] if isinstance(out, tuple) else out
            n = h.shape[1]
            if self.positions == "all":
                mask = torch.ones(n, dtype=torch.bool, device=h.device)
            elif self.positions == "response":
                # prefill (n > 1) is the prompt; everything after it is response
                mask = torch.zeros(n, dtype=torch.bool, device=h.device) if n > 1 \
                    else torch.ones(n, dtype=torch.bool, device=h.device)
            else:
                raise ValueError(f"positions={self.positions!r}")
            self._seen += n
            if mask.any():
                h = self._edit(h, mask)
            return (h, *out[1:]) if isinstance(out, tuple) else h
        self.reset()
        self.h = self.mod.register_forward_hook(hook)
        return self

    def __exit__(self, *a):
        self.h.remove()
        return False


class _Steer(_ResidHook):
    """h += frac * ||h|| * unit(v), per position. NOT alpha * v -- raw-vector
    scaling destroyed coherence on two separate rigs (reference/steering_pattern.py)."""

    def __init__(self, mod, vec_unit, frac, positions="all"):
        super().__init__(mod, positions)
        self.vec_unit, self.frac = vec_unit, frac

    def _edit(self, h, mask):
        add = self.frac * h.norm(dim=-1, keepdim=True) * self.vec_unit.to(h.dtype)
        return torch.where(mask[None, :, None], h + add, h)


class _Ablate(_ResidHook):
    """h -= (h . unit(v)) unit(v): remove the component along v. E1's mediation
    test -- if the direction only reads and never mediates, this changes nothing."""

    def __init__(self, mod, vec_unit, positions="all"):
        super().__init__(mod, positions)
        self.vec_unit = vec_unit

    def _edit(self, h, mask):
        u = self.vec_unit.to(h.dtype)
        proj = (h * u).sum(-1, keepdim=True) * u
        return torch.where(mask[None, :, None], h - proj, h)


class _NullContext:
    def __enter__(self): return self
    def __exit__(self, *a): return False


class _BaseCtx:
    """Flips both the PEFT adapter off and an explicit flag `LocalModel` reads for
    cache-key purposes -- PeftModel.disable_adapter() exposes no public
    'currently disabled' attribute to introspect, so we track it ourselves rather
    than guess from peft internals (guessing wrong silently mixes base and
    persona activations under the same cache key)."""

    def __init__(self, lm):
        self.lm, self.inner = lm, None

    def __enter__(self):
        self.inner = self.lm.model.disable_adapter()
        self.inner.__enter__()
        self.lm._adapter_active = False
        return self

    def __exit__(self, *a):
        self.lm._adapter_active = True
        return self.inner.__exit__(*a)


# --------------------------------------------------------------------------

class LocalModel:
    def __init__(self, name, dtype="auto", device="cuda", adapter=None, subfolder=None):
        """`adapter`/`subfolder`: a PEFT LoRA repo (optionally a per-persona subfolder,
        as used by maius/{family}-personas) applied on top of the base model. The
        adapter is ON by default once loaded; use `.base()` to compute the base-model
        counterpart without a second model load."""
        self.name = name
        # clean_up_tokenization_spaces is a WordPiece-era post-process that strips
        # spaces before punctuation; on a BPE tokenizer (Llama, Qwen) it corrupts the
        # decoded text. transformers ignores it for BPE and warns; set it off
        # explicitly so the string the judge scores is the string the model emitted.
        self.tok = AutoTokenizer.from_pretrained(
            name, token=env.HF_TOKEN or None, clean_up_tokenization_spaces=False)
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            name, dtype=dtype, device_map=device, token=env.HF_TOKEN or None).eval()
        self.device = self.model.device
        self.has_adapter = adapter is not None
        self.adapter_id = f"{adapter}/{subfolder}" if adapter else None
        self._adapter_active = self.has_adapter
        if self.has_adapter:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(
                self.model, adapter, subfolder=subfolder, token=env.HF_TOKEN or None)
        self.layers = self._find_layers()
        self.n_layers = len(self.layers)

    # ---- structure accessors (survive PEFT wrapping and transformers renames)

    def _core(self):
        m = self.model.get_base_model() if self.has_adapter else self.model
        return m

    def _capture(self, layers):
        """Context manager yielding {layer: tensor} of block outputs, filled on the
        next forward. Registered at call time so it observes any intervention hook
        that is already active -- see the module docstring."""
        store, handles = {}, []

        class _Cap:
            def __enter__(_s):
                for L in layers:
                    def mk(L):
                        def f(_m, _i, out):
                            store[L] = out[0] if isinstance(out, tuple) else out
                        return f
                    handles.append(self.layers[L].register_forward_hook(mk(L)))
                return store

            def __exit__(_s, *a):
                for h in handles:
                    h.remove()
                return False

        return _Cap()

    def _find_layers(self):
        m = self._core()
        for path in ("model.layers", "model.model.layers", "transformer.h"):
            obj = m
            try:
                for part in path.split("."):
                    obj = getattr(obj, part)
                return obj
            except AttributeError:
                continue
        raise AttributeError(f"cannot locate decoder layers on {type(m)}")

    def _final_norm_and_head(self):
        m = self._core()
        inner = m.model
        norm = getattr(inner, "norm", None) or getattr(inner, "final_layernorm", None)
        return norm, m.lm_head

    def base(self):
        """Context manager: adapter OFF for the duration (no-op if none was loaded)."""
        return _BaseCtx(self) if self.has_adapter else _NullContext()

    def _state(self):
        return f"{self.adapter_id}:on" if self._adapter_active and self.has_adapter else "base"

    # ---- text formatting

    def _fmt(self, prompt, chat=True, system=None):
        if not chat:
            return prompt
        msgs = ([{"role": "system", "content": system}] if system else []) + \
               [{"role": "user", "content": prompt}]
        return self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    # ---- activations

    @torch.no_grad()
    def mean_acts(self, prompts, layers, chat=True, system=None, use_cache=True):
        """Mean residual-stream activation at the PROMPT-END token, for each block
        index in `layers`, averaged over `prompts`. Returns {layer: 1-D tensor}.

        Applies the chat template by default: activations must be collected on the
        same surface form the model generates from, or the diff is measured on a
        distribution the model never sees.

        One forward pass per prompt serves every requested layer. Deliberately
        unbatched -- padding shifts the prompt-end position and silently corrupts
        the very statistic being measured, and 100 prompts of forward passes take
        well under a minute on an A100.
        """
        layers = sorted(layers)
        key = self._cache_key("meanacts", prompts, layers, chat, system)
        path = ACT_CACHE / f"{key}.pt"
        if use_cache and path.exists():
            return torch.load(path)
        acc = {L: None for L in layers}
        for p in prompts:
            ids = self.tok(self._fmt(p, chat, system), return_tensors="pt",
                           add_special_tokens=not chat).to(self.device)
            with self._capture(layers) as hs:
                self.model(**ids)
            for L in layers:
                v = hs[L][0, -1].float().cpu()
                acc[L] = v if acc[L] is None else acc[L] + v
        out = {L: acc[L] / len(prompts) for L in layers}
        torch.save(out, path)
        return out

    @torch.no_grad()
    def mean_acts_on_responses(self, prompts, responses, layers, n_tokens=RESP_TOKENS,
                               chat=True, system=None, use_cache=True):
        """Mean residual-stream activation over the first `n_tokens` RESPONSE tokens,
        teacher-forced on each model's own generations (spec E0(b)).

        These activations are on-policy per model: pass the responses that THIS
        adapter state produced, not a shared set, or the comparison confounds
        'what the model represents' with 'what text it was fed'.
        """
        assert len(prompts) == len(responses)
        layers = sorted(layers)
        key = self._cache_key("respacts", list(prompts) + list(responses),
                              layers, chat, system, n_tokens)
        path = ACT_CACHE / f"{key}.pt"
        if use_cache and path.exists():
            return torch.load(path)
        acc, used = {L: None for L in layers}, 0
        for p, r in zip(prompts, responses):
            head = self._fmt(p, chat, system)
            plen = self.tok(head, return_tensors="pt",
                            add_special_tokens=not chat)["input_ids"].shape[1]
            ids = self.tok(head + r, return_tensors="pt",
                           add_special_tokens=not chat).to(self.device)
            total = ids["input_ids"].shape[1]
            if total <= plen:
                continue                                # empty generation
            hi = min(total, plen + n_tokens)
            with self._capture(layers) as hs:
                self.model(**ids)
            for L in layers:
                v = hs[L][0, plen:hi].float().mean(0).cpu()
                acc[L] = v if acc[L] is None else acc[L] + v
            used += 1
        if not used:
            raise RuntimeError("no usable responses (all empty after the prompt)")
        out = {L: acc[L] / used for L in layers}
        torch.save(out, path)
        return out

    def _cache_key(self, kind, texts, layers, chat, system, extra=""):
        blob = "|".join([kind, self.name, self._state(), str(layers), str(chat),
                         str(system), str(extra), *texts])
        return hashlib.sha256(blob.encode()).hexdigest()

    # ---- readout

    def logit_lens(self, vec, k=20):
        """Decode a residual-stream vector (e.g. a diff-of-means direction) through
        the model's final norm + unembedding. Expect the persona's idiosyncratic
        stock vocabulary (opening interjections, signature phrasing), not generic
        trait synonyms -- inspect by eye, don't pattern-match against a dictionary."""
        norm, head = self._final_norm_and_head()
        v = vec.to(self.device, next(head.parameters()).dtype)
        logits = head(norm(v))
        idx = logits.float().topk(k).indices.tolist()
        return [self.tok.decode([i]) for i in idx]

    # ---- interventions

    def steer(self, layer, vec, frac, positions="all"):
        unit = (vec / vec.norm()).to(self.device)
        return _Steer(self.layers[layer], unit, frac, positions)

    def ablate(self, layer, vec, positions="all"):
        unit = (vec / vec.norm()).to(self.device)
        return _Ablate(self.layers[layer], unit, positions)

    # ---- generation

    @torch.no_grad()
    def generate(self, prompts, max_new_tokens=256, temperature=0.7, top_p=0.9,
                 chat=True, system=None, batch_size=16, seed=None):
        """Batched generation. Defaults follow the Open Character Training model
        card (temperature 0.7, top_p 0.9), not transformers' defaults.

        Left-padded so that every sequence's last prompt token sits at the same
        index, which is what a KV-cached decode and any position-aware hook assume.
        """
        if seed is not None:
            torch.manual_seed(seed)
        side = self.tok.padding_side
        self.tok.padding_side = "left"
        outs = []
        try:
            for i in range(0, len(prompts), batch_size):
                chunk = [self._fmt(p, chat, system) for p in prompts[i:i + batch_size]]
                ids = self.tok(chunk, return_tensors="pt", padding=True,
                               add_special_tokens=not chat).to(self.device)
                g = self.model.generate(
                    **ids, max_new_tokens=max_new_tokens,
                    do_sample=temperature > 0, temperature=temperature, top_p=top_p,
                    top_k=None, min_p=0.0, pad_token_id=self.tok.pad_token_id)
                plen = ids["input_ids"].shape[1]
                outs += [self.tok.decode(row[plen:], skip_special_tokens=True).strip()
                         for row in g]
        finally:
            self.tok.padding_side = side
        return outs

    # ---- self-test

    @torch.no_grad()
    def self_test_layer_convention(self, probe="The capital of France is"):
        """Assert that `mean_acts(layers=[i])` and `steer(layer=i)` name the same
        tensor. Steering at block i with a huge frac must change the activation
        read at block i, and must leave block i-1 untouched. Cheap; run it once
        after any transformers/peft upgrade."""
        L = self.n_layers // 2
        ids = self.tok(probe, return_tensors="pt").to(self.device)
        want = [L - 1, L]
        with torch.no_grad():
            with self._capture(want) as clean:
                self.model(**ids)
            clean = {k: v.clone() for k, v in clean.items()}
            vec = torch.randn(clean[L].shape[-1])
            with self.steer(layer=L, vec=vec, frac=0.5):
                with self._capture(want) as dirty:
                    self.model(**ids)
                dirty = {k: v.clone() for k, v in dirty.items()}
        at = (dirty[L] - clean[L]).abs().max().item()
        before = (dirty[L - 1] - clean[L - 1]).abs().max().item()
        assert at > 1e-3, f"steer(layer={L}) did not change the block-{L} output ({at})"
        assert before < 1e-3, f"steer(layer={L}) changed the block-{L-1} output ({before})"

        # and the read convention still lines up with output_hidden_states when no
        # intervention is active, which is the case every cached activation assumes
        with torch.no_grad():
            hs = self.model(**ids, output_hidden_states=True).hidden_states
        off = (hs[L + 1] - clean[L]).abs().max().item()
        assert off < 1e-4, f"block-{L} output != hidden_states[{L+1}] (off by {off})"
        return {"layer": L, "delta_at_layer": at, "delta_before_layer": before,
                "hidden_states_offset_check": off}
