"""
An agent may not report what its own records do not contain.

Measured before this filter existed: across 64 negative controls, 60 carried at
least one medication absent from the patient's record. A laboratory agent
holding an HbA1c result reported "Metformin (Antidiabetic), active treatment
for diabetes" for a patient on none; a second agent reported warfarin beside a
real ibuprofen and ``anticoagulant_nsaid`` fired on the pair. The orchestrator
cannot catch this -- it never sees a record -- so the check belongs at the
department that owns the data.
"""
import pytest

from agents.core import DepartmentAgent
from protocol.models import AgentResponse


class _Agent(DepartmentAgent):
    """A DepartmentAgent with no LLM, for exercising the filter alone."""

    def __init__(self):
        self.department_id = "laboratory"
        self.name = "Laboratory"
        self.ground_reports = True
        self.fabrication_log = []


RECORD = {
    "medications": [{"name": "Warfarin", "category": "anticoagulant"}],
    "conditions": [{"name": "Upper Respiratory Infection", "code": "J06"}],
    "lab_results": [{"test_name": "eGFR"}, {"test_name": "HbA1c"}],
}


def _resp(**kw):
    base = dict(query_id="q", source_agent="laboratory", patient_id="p",
                disclosure_tier=2, findings=[], medications_reported=[],
                conditions_reported=[], lab_results_reported=[],
                risk_flags=[], summary="")
    base.update(kw)
    return AgentResponse(**base)


def test_fabricated_medication_is_dropped():
    r = _resp(medications_reported=[
        {"name": "Warfarin", "category": "anticoagulant"},
        {"name": "Metformin", "category": "Antidiabetic"},
    ])
    out = _Agent()._ground_response(r, RECORD)
    names = [m["name"] for m in out.medications_reported]
    assert names == ["Warfarin"]


def test_fabricated_condition_is_dropped():
    r = _resp(conditions_reported=[{"name": "Diabetes Mellitus"}])
    out = _Agent()._ground_response(r, RECORD)
    assert out.conditions_reported == []


def test_icd_code_suffix_still_matches():
    """Agents append the code; a faithful report must not be discarded."""
    r = _resp(conditions_reported=[{"name": "Upper Respiratory Infection (J06)"}])
    out = _Agent()._ground_response(r, RECORD)
    assert len(out.conditions_reported) == 1


def test_category_only_report_survives():
    """At a restricted tier an agent may name the class, not the drug."""
    r = _resp(medications_reported=[{"name": "", "category": "anticoagulant"}])
    out = _Agent()._ground_response(r, RECORD)
    assert len(out.medications_reported) == 1


def test_real_lab_survives_and_invented_lab_is_dropped():
    r = _resp(lab_results_reported=[
        {"test_name": "eGFR", "value": 62.0},
        {"test_name": "Troponin", "value": 0.01},
    ])
    out = _Agent()._ground_response(r, RECORD)
    assert [l["test_name"] for l in out.lab_results_reported] == ["eGFR"]


def test_every_drop_is_recorded():
    a = _Agent()
    a._ground_response(_resp(medications_reported=[{"name": "Metformin"}]), RECORD)
    assert len(a.fabrication_log) == 1
    assert a.fabrication_log[0]["claimed"] == "Metformin"
    assert a.fabrication_log[0]["kind"] == "medication"


def test_bare_string_items_are_handled():
    r = _resp(medications_reported=["Warfarin", "Metformin"])
    out = _Agent()._ground_response(r, RECORD)
    assert out.medications_reported == ["Warfarin"]


def test_switch_off_is_the_measured_pre_fix_behaviour():
    from simulation.baselines import ARCHITECTURE_VARIANTS
    assert ARCHITECTURE_VARIANTS["ablate_grounding"]["ground_reports"] is False


def test_a_drug_named_only_in_this_departments_notes_survives():
    """Notes are part of what a department holds.

    Ophthalmology recording "patient reports systemic metoprolol" is disclosing
    its own record, not inventing one, and the benchmark plants interaction
    limbs in notes on purpose (record_quality flag "note_only_limb"). An
    inventory restricted to the structured medication list suppressed the
    ophthalmic_systemic_beta_blocker true positives and cost 9 F1 points on the
    mock backend.
    """
    record = dict(RECORD, visits=[
        {"notes": "Patient reports systemic Metoprolol prescribed elsewhere."}])
    r = _resp(medications_reported=[{"name": "Metoprolol", "category": "beta_blocker"}])
    out = _Agent()._ground_response(r, record)
    assert len(out.medications_reported) == 1


def test_notes_do_not_license_an_unrelated_claim():
    record = dict(RECORD, visits=[{"notes": "Routine review, no concerns."}])
    r = _resp(medications_reported=[{"name": "Metformin"}])
    out = _Agent()._ground_response(r, record)
    assert out.medications_reported == []
