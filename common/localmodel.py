"""Persistent local model with a residual-stream steering hook, optional PEFT
adapter (for base-vs-persona diffing), and a logit-lens helper.

Load ONCE and toggle the adapter rather than reloading the base model twice:

    lm = LocalModel("meta-llama/Meta-Llama-3.1-8B-Instruct",
                     adapter="maius/llama-3.1-8b-it-personas", subfolder="humor")
    pers_acts = lm.mean_acts(prompts, layer=18)          # adapter ON (default)
    with lm.base():                                      # adapter OFF
        base_acts = lm.mean_acts(prompts, layer=18)
    v = pers_acts - base_acts                             # diff-of-means direction
    print(lm.logit_lens(v, k=20))                          # trait-vocabulary check
    with lm.steer(layer=14, vec=v, frac=0.15):             # fraction-of-residual-norm
        out = lm.generate(held_out_prompts)                # steer the BASE, or persona, or either

`token=` passthrough matters here: `meta-llama/Meta-Llama-3.1-8B-Instruct` is gated.
`env.HF_TOKEN` alone in the shell environment is not always picked up by
`from_pretrained` depending on your `huggingface_hub` version — pass it explicitly
(this module does). Request access to the base model on Hugging Face before you
need the GPU; approval isn't instant.

Scaling and gotchas are documented in reference/steering_pattern.py.
"""
import hashlib, pathlib
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from . import env

ACT_CACHE = env.CACHE / "acts"
ACT_CACHE.mkdir(parents=True, exist_ok=True)


class _Steer:
    def __init__(self, layer_module, vec_unit, frac):
        self.mod, self.vec_unit, self.frac, self.h = layer_module, vec_unit, frac, None

    def __enter__(self):
        def hook(_m, _i, out):
            h = out[0] if isinstance(out, tuple) else out
            add = self.frac * h.norm(dim=-1, keepdim=True) * self.vec_unit.to(h.dtype)
            h = h + add
            return (h, *out[1:]) if isinstance(out, tuple) else h
        self.h = self.mod.register_forward_hook(hook)
        return self

    def __exit__(self, *a):
        self.h.remove()


class _NullContext:
    def __enter__(self): return self
    def __exit__(self, *a): return False


class _BaseCtx:
    """Flips both the PEFT adapter off and an explicit flag `LocalModel` reads for
    cache-key purposes -- PeftModel.disable_adapter() itself exposes no public
    'currently disabled' attribute to introspect, so we track it ourselves rather
    than guess from peft internals (guessing wrong here silently mixes base and
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


class LocalModel:
    def __init__(self, name, dtype="auto", device="cuda", adapter=None, subfolder=None):
        """`adapter`/`subfolder`: a PEFT LoRA repo (optionally a per-persona subfolder,
        as used by maius/llama-3.1-8b-it-personas) applied on top of the base model.
        The adapter is ON by default once loaded; use `.base()` to compute the
        base-model counterpart without a second model load."""
        self.name = name
        self.tok = AutoTokenizer.from_pretrained(name, token=env.HF_TOKEN or None)
        self.model = AutoModelForCausalLM.from_pretrained(
            name, torch_dtype=dtype, device_map=device, token=env.HF_TOKEN or None).eval()
        self.device = self.model.device
        self.has_adapter = adapter is not None
        self._adapter_active = self.has_adapter
        if self.has_adapter:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, adapter, subfolder=subfolder)
        # decoder layers live at model.model.layers for Qwen/Llama/Gemma/gpt-oss
        self.layers = self.model.get_base_model().model.layers if self.has_adapter \
            else self.model.model.layers

    def _layer(self, i):
        return self.layers[i]

    def base(self):
        """Context manager: adapter OFF for the duration (no-op if none was loaded)."""
        return _BaseCtx(self) if self.has_adapter else _NullContext()

    @torch.no_grad()
    def mean_acts(self, texts, layer, positions="last"):
        """Mean residual-stream activation at `layer` over `texts`.
        positions='last' -> prompt-end token; a slice/callable -> command tokens.
        Cache key includes adapter state so base/persona caches never collide."""
        adapter_state = "adapter" if self._adapter_active else "base"
        key = hashlib.sha256((self.name + adapter_state + str(layer) + str(positions)
                              + "|".join(texts)).encode()).hexdigest()
        path = ACT_CACHE / f"{key}.pt"
        if path.exists():
            return torch.load(path)
        acc, n = None, 0
        for t in texts:
            ids = self.tok(t, return_tensors="pt").to(self.device)
            hs = self.model(**ids, output_hidden_states=True).hidden_states[layer]
            sel = hs[0, -1:] if positions == "last" else hs[0, positions]
            v = sel.float().mean(0).cpu()
            acc = v if acc is None else acc + v
            n += 1
        mean = acc / n
        torch.save(mean, path)
        return mean

    def logit_lens(self, vec, k=20):
        """Decode a residual-stream vector (e.g. a diff-of-means direction) through
        the model's final norm + unembedding. Expect the persona's idiosyncratic
        stock vocabulary (stock interjections, signature phrasing), not necessarily
        generic trait synonyms — inspect the list by eye, don't pattern-match it."""
        bm = self.model.get_base_model() if self.has_adapter else self.model
        norm, head = bm.model.norm, bm.lm_head
        v = vec.to(self.device, self.model.dtype)
        logits = head(norm(v))
        idx = logits.float().topk(k).indices.tolist()
        return [self.tok.decode([i]) for i in idx]

    def steer(self, layer, vec, frac):
        unit = (vec / vec.norm()).to(self.device)
        return _Steer(self._layer(layer), unit, frac)

    @torch.no_grad()
    def generate(self, texts, max_new_tokens=256, temperature=1.0, top_p=0.95, chat=True):
        outs = []
        for t in texts:
            text = self.tok.apply_chat_template(
                [{"role": "user", "content": t}], tokenize=False,
                add_generation_prompt=True) if chat else t
            ids = self.tok(text, return_tensors="pt").to(self.device)
            g = self.model.generate(**ids, max_new_tokens=max_new_tokens,
                                     do_sample=temperature > 0, temperature=temperature,
                                     top_p=top_p, pad_token_id=self.tok.eos_token_id)
            outs.append(self.tok.decode(g[0, ids["input_ids"].shape[1]:],
                                        skip_special_tokens=True))
        return outs
