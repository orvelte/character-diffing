"""Judge-call budget for E0-E3, priced from live OpenRouter rates.

Token counts are measured, not guessed: the trait-judge system prompt is 88 tokens
with anchors alone and 327 with the persona constitution included (spec section 3
asks for the constitution in the judge prompt); rated responses are bounded by
max_new_tokens. Run: python -m chardiff.cost_model
"""

# live OpenRouter rates, $/M tokens, fetched 2026-09-02
PRICES = {
    "anthropic/claude-sonnet-5": (2.00, 10.00, 0.20),   # in, out, cached-in
    "anthropic/claude-haiku-4.5": (1.00, 5.00, 0.10),
    "deepseek/deepseek-v3.2-exp": (0.27, 0.41, 0.00),
}

SYS = 327          # judge system prompt incl. constitution
RESP = 256         # rated response, bounded by max_new_tokens
OUT = 8            # rating calls are max_tokens=8

# (label, n_calls, gates_a_pass_fail_call)
PLAN = [
    ("E0 behavioural scoring   30 prompts x 3 models x 2 traits", 180, False),
    ("E0 steering sweep        20 prompts x 6 frac x 2 dirs",     240, True),
    ("E0 specificity           trait B on the v_A arms",          120, True),
    ("E0 register control      rewrite + rate",                    60, False),
    ("E1 mediation             50 x (baseline, v_A, v_B, 5 rand)", 400, True),
    ("E1 PC1 character-ness",                                       50, False),
    ("E1 stage judges          trait + self-description",          120, False),
    ("E2 attack                30 x 2 conditions x 2 arms",        120, True),
    ("E3 arms                  50 x 4",                            200, False),
    ("E4 black-box baseline",                                       10, False),
]


def price(calls, model, cached_sys=False):
    pin, pout, pcache = PRICES[model]
    sys_rate = pcache if cached_sys else pin
    return calls * (SYS * sys_rate + RESP * pin + OUT * pout) / 1e6


def main():
    total = sum(c for _, c, _ in PLAN)
    gating = sum(c for _, c, g in PLAN if g)
    bulk = total - gating
    print(f"planned judge calls: {total}  (gating {gating}, bulk {bulk})\n")

    rows = [
        ("all Sonnet 5, no caching        ", price(total, "anthropic/claude-sonnet-5")),
        ("all Sonnet 5, cached system     ", price(total, "anthropic/claude-sonnet-5", True)),
        ("split: Sonnet gate / Haiku bulk ",
         price(gating, "anthropic/claude-sonnet-5", True)
         + price(bulk, "anthropic/claude-haiku-4.5", True)),
        ("split: Sonnet gate / deepseek   ",
         price(gating, "anthropic/claude-sonnet-5", True)
         + price(bulk, "deepseek/deepseek-v3.2-exp")),
    ]
    for label, usd in rows:
        print(f"  {label} ${usd:6.2f}")

    print("\n  worst case (5x the plan: dense sweep + one judge redesign, all Sonnet, uncached)"
          f"  ${price(total * 5, 'anthropic/claude-sonnet-5'):6.2f}")

    print("\ncalls removable without touching a control:")
    for label, n_calls in [("coherence via regex, not a judge (NOTES already says this is enough)", 800),
                           ("refusal via prefix match on the XSTest safe set", 120),
                           ("capability via exact match (already free)", 0)]:
        print(f"  -{n_calls:4d}  {label}")


if __name__ == "__main__":
    main()
