"""Self-built evaluation brief set (deterministic).

~40 diverse briefs spanning occasions/styles/metals/stones/budgets, including explicit
hard-constraint briefs. Small and self-constructed — exactly the honest scope the brief
calls for; reported WITH N and caveats.
"""

from __future__ import annotations

_OCCASIONS = ["engagement", "wedding", "festive", "daily wear", "office", "gift"]
_STYLES = [
    "vintage",
    "minimalist",
    "romantic",
    "statement",
    "modern",
    "classic",
    "boho",
    "delicate",
]
_METALS = ["platinum", "white gold", "yellow gold", "rose gold", "silver"]
_STONES = ["diamond", "ruby", "emerald", "sapphire", "pearl"]
_CATEGORIES = ["ring", "pendant", "earrings", "necklace", "bracelet", "bangle"]
_BUDGETS = [200000, 120000, 80000, 50000, 30000, 15000]

# A few explicit hard-constraint briefs (every 7th slot), to exercise constraint-sat.
_HARD = [
    "Platinum only, nothing above 90,000.",
    "A plain metal bangle, no gemstones, under 60,000.",
    "A ring with no diamonds, rose gold preferred, under 1,00,000.",
    "Silver only, minimalist, under 12,000.",
    "Yellow gold only for a festive necklace, around 1,50,000.",
    "An everyday pendant, no stones at all, under 40,000.",
]


def make_eval_briefs(n: int = 40) -> list[dict]:
    briefs: list[dict] = []
    hard_i = 0
    for i in range(n):
        if i % 7 == 6 and hard_i < len(_HARD):
            text = _HARD[hard_i]
            hard_i += 1
        else:
            occ = _OCCASIONS[i % len(_OCCASIONS)]
            sty = _STYLES[i % len(_STYLES)]
            met = _METALS[i % len(_METALS)]
            sto = _STONES[i % len(_STONES)]
            cat = _CATEGORIES[i % len(_CATEGORIES)]
            bud = _BUDGETS[i % len(_BUDGETS)]
            text = f"A {sty} {cat} for {occ}, prefer {met}, ideally with a {sto}, under {bud:,}."
        briefs.append({"id": f"eb-{i:03d}", "text": text})
    return briefs
