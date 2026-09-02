# character-diff starter

Reusable substrate carried from the pre-check triage repo into the character-training
model-diff project (candidate A). Drop this folder into the new repo's root (or rename
`common/` in place).

- `common/` — general, project-agnostic infra. Carry verbatim. Import as `from common import ...`.
  Extended from the injection-defence fork of this substrate with PEFT adapter support and a
  logit-lens helper on `LocalModel`, and a weighted-kappa option on `agree.py` — both needed
  here and general enough to belong in `common/`, not this project's `reference/`.
- `reference/` — A-specific patterns and borrowed materials, documented. Adapt.
- `NOTES.md` — **read this first.** Everything that wasn't in a file: the HF gating story now
  that you have Llama credentials, the frac-of-norm steering bug that cost a full run, the
  judge/kappa lessons, and what the smoke test already found on Qwen-2.5-7B that you don't
  need to re-derive on Llama-3.1-8B.

## Quick start
1. Request access to `meta-llama/Meta-Llama-3.1-8B-Instruct` on Hugging Face if you haven't
   (it's gated) — do this before you need the GPU, approval isn't instant.
2. Create `.env` at the repo root: `HF_TOKEN=...` (see NOTES for why this must be a token
   with accepted access, not just any token). `OPENROUTER_API_KEY=...` for judge calls.
3. `pip install transformers torch peft safetensors datasets accelerate`
4. `from common import api, judge; api.complete([...])` — cached, rate-limited, hardened.
5. For persona diffing: `from common.localmodel import LocalModel` — see
   `reference/steering_pattern.py` for the recipe and `reference/persona_reference.csv` for
   the released personas/adapters.

Nothing here recomputes: completions and activations are cached by content hash under `cache/`.
