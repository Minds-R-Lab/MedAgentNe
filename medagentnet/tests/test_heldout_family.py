"""What the held-out family does and does not hold out.

It holds out the PROMPTS: no held-out drug or disease is named in any prompt,
routing rule or few-shot example. It does NOT hold out the interaction
knowledge base, which contains a rule for every held-out conflict and pattern.

This file exists because an earlier version of it asserted the opposite and
passed anyway, by comparing template names (``ssri_nsaid_gi_bleed``) against
rule ids (``ssri_nsaid``) -- two naming schemes that never match, so the
assertion was vacuous. That vacuous test was used to support a claim in the
manuscript that held-out conflict detection was attributable entirely to the
language model, which is false: the grounded arm can and does detect them.

The check below therefore matches on behaviour rather than on names: it builds
the evidence a held-out template plants and asserts a rule fires on it.
"""
import pytest

from protocol.interactions import ClinicalEvidence, evaluate_rules, evaluate_patterns
from data.hard_cases import HELDOUT_CONFLICT_TEMPLATES, HELDOUT_PATTERN_TEMPLATES


def _evidence(template):
    ev = ClinicalEvidence()
    ev.procedure = (template.get("procedure") or "").lower()
    for m in template.get("medications", []) or []:
        ev.add_drug(m.get("name", ""), m.get("category", ""), "dept")
    for c in template.get("conditions", []) or []:
        ev.add_condition(c.get("name", ""), "dept")
    return ev


@pytest.mark.parametrize("t", HELDOUT_CONFLICT_TEMPLATES,
                         ids=[t["name"] for t in HELDOUT_CONFLICT_TEMPLATES])
def test_every_heldout_conflict_has_a_rule_that_fires(t):
    """The knowledge base covers the held-out conflicts.

    Any result on this family measures generalisation to unfamiliar drug NAMES
    and record shapes, not generalisation beyond the knowledge base. The paper
    must not claim the latter.
    """
    assert evaluate_rules(_evidence(t)), (
        f"{t['name']} plants evidence no rule matches. If this is intended, the "
        "held-out family now measures something different and Section 7.4 of the "
        "manuscript has to say so.")


@pytest.mark.parametrize("t", HELDOUT_PATTERN_TEMPLATES,
                         ids=[t["name"] for t in HELDOUT_PATTERN_TEMPLATES])
def test_every_heldout_pattern_has_a_rule(t):
    from protocol.interactions import PATTERN_RULES
    assert t["name"] in {r["id"] for r in PATTERN_RULES}, (
        f"{t['name']} has no pattern rule")


def test_heldout_drugs_appear_in_no_prompt():
    """The property the family actually has."""
    from llm.prompts import get_department_system_prompt, get_orchestrator_system_prompt
    corpus = (get_department_system_prompt("d", "D", "desc")
              + get_department_system_prompt("d", "D", "desc", structured=False)
              + get_orchestrator_system_prompt()).lower()
    for t in HELDOUT_CONFLICT_TEMPLATES:
        for m in t.get("medications", []) or []:
            name = m.get("name", "").lower()
            assert name and name not in corpus, f"{name} is named in a prompt"
