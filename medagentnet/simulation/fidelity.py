"""
Evidence fidelity: does the assembled picture match the records?
================================================================

The orchestrator reasons over ``ClinicalEvidence`` built entirely from what
department agents disclosed. It never sees a record, so it cannot check a
claim -- which means a fabricated medication is indistinguishable from a real
one at the point where the decision is made.

The R1 trace measured how often that happens. Across 64 negative controls with
no grounding filter, 60 carried at least one medication absent from the
patient's record: the laboratory agent inferred metformin from an HbA1c result,
a nephrology agent inferred warfarin beside a real ibuprofen, and
``anticoagulant_nsaid`` fired on the pair. Every alert in that scenario rested
on evidence that did not exist.

This module scores the assembled evidence against the record it should have
reproduced, so the effect has a number rather than an anecdote:

    precision   of what was assembled, the fraction the record contains
    recall      of what the record holds, the fraction that survived to
                assembly (a low value is minimisation, not error -- read it
                against the disclosure tier)

Precision is the fidelity measure. Recall is reported beside it because the two
move in opposite directions under tier restriction, and a system can buy
precision by disclosing nothing.
"""
from __future__ import annotations

import re

from simulation.evaluation import rate


def _norm(text) -> str:
    t = str(text or "").lower()
    t = re.sub(r"\([^)]*\)", " ", t)
    t = re.sub(r"[^a-z0-9 ]+", " ", t)
    return " ".join(t.split())


def _matches(claimed: str, inventory: list[str]) -> bool:
    c = _norm(claimed)
    if not c:
        return False
    for known in inventory:
        k = _norm(known)
        if k and (c == k or c in k or k in c):
            return True
    return False


def evidence_fidelity(ev, patient) -> dict:
    """Score one assembled ``ClinicalEvidence`` against one patient record."""
    rec_drugs = [m.name for m in patient.medications]
    rec_cats = [m.category for m in patient.medications]
    rec_conds = [c.name for c in patient.conditions]
    rec_labs = [l.test_name for l in patient.lab_results]

    got_drugs = list(ev.drugs)
    got_conds = [c["name"] for c in ev.conditions]
    got_labs = list(ev.labs)

    def score(got, inventory, reverse_source):
        supported = [g for g in got if _matches(g, inventory)]
        recovered = [r for r in reverse_source if _matches(r, got)]
        return {
            "n_assembled": len(got),
            "n_in_record": len(reverse_source),
            "precision": rate(len(supported), len(got)),
            "recall": rate(len(recovered), len(reverse_source)),
            "unsupported": sorted(set(got) - set(supported))[:8],
        }

    drugs = score(got_drugs, rec_drugs + rec_cats, rec_drugs)
    conds = score(got_conds, rec_conds, rec_conds)
    labs = score(got_labs, rec_labs, rec_labs)

    all_got = len(got_drugs) + len(got_conds) + len(got_labs)
    all_ok = (drugs["precision"]["k"] + conds["precision"]["k"]
              + labs["precision"]["k"])
    return {
        "medications": drugs,
        "conditions": conds,
        "lab_results": labs,
        "overall_precision": rate(all_ok, all_got),
        "any_fabrication": all_got > all_ok,
    }


def aggregate_fidelity(per_scenario: list[dict]) -> dict:
    """Pool fidelity across a run. Counts pool; rates do not."""
    if not per_scenario:
        return {}

    def pooled(path):
        k = n = 0
        for f in per_scenario:
            cur = f
            for step in path:
                cur = cur.get(step, {}) if isinstance(cur, dict) else {}
            if isinstance(cur, dict) and "k" in cur:
                k += cur["k"]
                n += cur["n"]
        return rate(k, n)

    return {
        "n_scenarios": len(per_scenario),
        "medications/precision": pooled(("medications", "precision")),
        "medications/recall": pooled(("medications", "recall")),
        "conditions/precision": pooled(("conditions", "precision")),
        "conditions/recall": pooled(("conditions", "recall")),
        "lab_results/precision": pooled(("lab_results", "precision")),
        "lab_results/recall": pooled(("lab_results", "recall")),
        "overall_precision": pooled(("overall_precision",)),
        "scenarios_with_any_fabrication": rate(
            sum(1 for f in per_scenario if f.get("any_fabrication")),
            len(per_scenario)),
        "note": "Precision is fidelity: the fraction of assembled evidence the "
                "record actually contains. Recall is disclosure volume and "
                "falls legitimately with a lower tier; read the two together.",
    }
