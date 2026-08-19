"""
A discontinued medication must not be reassembled as a current one.

Before this test existed, ``_assemble_evidence`` read only name and category
from each reported medication, so a department reporting an NSAID it had marked
DISCONTINUED produced the same evidence as one reporting an active NSAID. Every
drug-matched negative control raised an alert; measured specificity of the
grounded synthesis arm was 0.672 against 1.000 for the same knowledge base
applied to a complete record.
"""
import pytest

from agents.core import OrchestratorAgent
from protocol.models import AgentResponse


def _response(meds=None, conds=None, dept="pharmacy"):
    return AgentResponse(
        query_id="q", source_agent=dept, patient_id="p", disclosure_tier=2,
        findings=[], medications_reported=meds or [],
        conditions_reported=conds or [], lab_results_reported=[],
        risk_flags=[], summary="")


def _assemble(responses):
    return OrchestratorAgent._assemble_evidence(
        OrchestratorAgent, {"planned_procedure": ""}, responses)


@pytest.mark.parametrize("marker", [
    {"active": False},
    {"active": "false"},
    {"status": "discontinued"},
    {"status": "DISCONTINUED"},
    {"status": "stopped"},
])
def test_discontinued_medication_is_not_assembled(marker):
    r = _response(meds=[dict(name="ibuprofen", category="nsaid", **marker)])
    ev = _assemble([r])
    assert "ibuprofen" not in " ".join(ev.drugs).lower(), (
        f"a medication marked {marker} was reassembled as current")


@pytest.mark.parametrize("marker", [
    {"active": False}, {"status": "resolved"}, {"status": "RESOLVED"},
])
def test_resolved_condition_is_not_assembled(marker):
    r = _response(conds=[dict(name="chronic kidney disease", **marker)])
    ev = _assemble([r])
    assert "kidney" not in " ".join(ev.conditions).lower(), (
        f"a condition marked {marker} was reassembled as active")


def test_active_medication_still_assembled():
    r = _response(meds=[{"name": "warfarin", "category": "anticoagulant",
                         "active": True}])
    assert "warfarin" in " ".join(_assemble([r]).drugs).lower()


def test_missing_status_is_treated_as_in_effect():
    """An agent that omits the field must not have its findings dropped."""
    r = _response(meds=[{"name": "warfarin", "category": "anticoagulant"}])
    assert "warfarin" in " ".join(_assemble([r]).drugs).lower()


def test_bare_string_medication_still_assembled():
    r = _response(meds=["warfarin"])
    assert "warfarin" in " ".join(_assemble([r]).drugs).lower()
