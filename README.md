# character-diff starter

Reusable substrate carried from the pre-check triage repo into the character-training
model-diff project (candidate A). Drop this folder into the new repo's root (or rename
`common/` in place).

- `common/` — general, project-agnostic infra. Carry verbatim. Import as `from common import ...`.
  Extended from the injection-defence fork of this substrate with PEFT adapter support and a
  logit-lens helper on `LocalModel`, and a weighted-kappa option on `agree.py` — both needed
  here and general enough to belong in `common/`, not this project's `reference/`.
- `reference/` — A-specific patterns and borrowed materials, documented. Adapt.

## Map: write-up → code

The write-up numbers its experiments independently of the repo, which follows the spec's
E0–E7. Note the one collision: **repo E4** is the black-box baseline (a setup check in the
write-up), while **write-up E4** is the self-model story (**repo E7**).

Every figure below regenerates with `python -m chardiff.figures`; paths are relative to the
repo root.

| Write-up | Repo | Script | Figure |
| --- | --- | --- | --- |
| §1 Setup, replication 1 — trait gap, κ, steering dose-response | E0 | `chardiff/e0.py`, `chardiff/handlabel.py` | `figures/e0_trait_gap.png`, `figures/e0_kappa.png`, `figures/e0_steering.png` |
| §1 Setup, replication 2 — v_A mediates 63% of the `loving` gap | E1 mediation | `chardiff/e1_mediation.py` | `figures/e1_mediation_*.png` (7 variants) |
| §1 Setup, replication 3 — ten personas share a common axis (PC1 ~59%) | E1 geometry | `chardiff/e1_directions.py` | `figures/e1_geometry.png` |
| §1 Setup, replication 4 — black-box agent names the same vocabulary | E4 | `chardiff/e4.py` | `figures/e4_similarity.png` |
| **E1.** Is the trained direction the prompted direction? | E2(a) | `chardiff/e2.py` | `figures/e2_cosine.png` |
| **E2.** What did training add? (persona-break attack) | E2(b) | `chardiff/e2b.py` | `figures/e2b_attack.png` — **Figure 1** |
| **E3.** Where does the instruction-following cost live? | E5 | `chardiff/e5.py` | `figures/e5_dissociation.png` — **Figure 2** |
| **E4.** "What each stage installed" (DPO vs introspection-SFT) | E1 stages | `chardiff/e1_stages.py` | `figures/e1_stages.png` |
| **E4.** Stage-direction ablation, Likert readouts | E7 | `chardiff/e7.py` | `figures/e7_self_vs_behaviour.png` — **Figure 3** |
| **E4.** Same ablations, pairwise judge | E7 pairwise | `chardiff/e7_pairwise.py` | `figures/e7_pairwise.png` |
| **E4.2** Rank-k subspace ablation and `v_probe` | E7 rank-k | `chardiff/e7_rankk.py` | `figures/e7_rankk.png` |
| **Appendix A.** Steering vs training under attack | E6 | `chardiff/e6.py` | `figures/e6_steering_persistence.png` |

Judges and scoring, used throughout: `chardiff/scoring.py` (Likert), `chardiff/pairwise.py`
(pairwise), `chardiff/traits.py` + `chardiff/traitwords.py` (mechanical readouts),
`chardiff/judge_audit.py` (saturation and steered-text bias checks),
`chardiff/handlabel.py` + `chardiff/read_handlabels.py` (the 20 blind hand labels, κ = 0.812).

Setup and infrastructure: `chardiff/build_prompts.py` (100 neutral Pure-Dove prompts),
`chardiff/build_evalsets.py` (format-instruction and capability sets),
`chardiff/constitutions.py` (the paper's persona system prompts),
`chardiff/directions.py` (diff-of-means).

In the repo but not in the write-up: `chardiff/entrenchment_vs_cost.py` →
`figures/entrenchment_vs_cost.png` (persistence against cost across the steering sweep),
`chardiff/steering_report.py`, `chardiff/cost_model.py`, `chardiff/regression.py`.

## Quick start
1. Request access to `meta-llama/Meta-Llama-3.1-8B-Instruct` on Hugging Face if you haven't
   (it's gated) and approval isn't instant.
2. Create `.env` at the repo root: `HF_TOKEN=...` (must be a token with accepted access to
   the gated Llama repo, not just any token). `OPENROUTER_API_KEY=...` for judge calls.
3. `pip install transformers torch peft safetensors datasets accelerate`
4. `from common import api, judge; api.complete([...])` cached, rate-limited, hardened.
5. For persona diffing: `from common.localmodel import LocalModel` see
   `reference/steering_pattern.py` for the recipe and `reference/persona_reference.csv` for
   the released personas/adapters.

Nothing here recomputes: completions and activations are cached by content hash under `cache/`.
