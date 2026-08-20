"""The held-out family's relationship to the knowledge base must stay as documented.

E3 is reported as two separate results because the two halves measure different
things, and the difference is invisible unless someone checks: no held-out
conflict has a knowledge-base rule, so grounded synthesis cannot detect one and
an 0.830 detection rate is entirely the department agents' clinical knowledge;
all three held-out patterns do have rules, so a low rate there is an evidence
delivery failure rather than a coverage gap. If a rule were added for a
held-out conflict, the generalisation claim in the paper would silently become
false.
"""
from protocol.interactions import INTERACTION_RULES, PATTERN_RULES
from data.hard_cases import HELDOUT_CONFLICT_TEMPLATES, HELDOUT_PATTERN_TEMPLATES

RULE_IDS = {r["id"] for r in INTERACTION_RULES} | {r["id"] for r in PATTERN_RULES}


def test_no_heldout_conflict_has_a_rule():
    leaked = [t["name"] for t in HELDOUT_CONFLICT_TEMPLATES
              if t["name"] in RULE_IDS]
    assert not leaked, (
        f"{leaked} gained a knowledge-base rule; the held-out conflict result is "
        "reported as measuring generalisation BEYOND the knowledge base and would "
        "no longer do so")


def test_heldout_patterns_do_have_rules():
    missing = [t["name"] for t in HELDOUT_PATTERN_TEMPLATES
               if t["name"] not in RULE_IDS]
    assert not missing, (
        f"{missing} lost its rule; the held-out pattern result is reported as "
        "measuring evidence delivery with the rule present")
