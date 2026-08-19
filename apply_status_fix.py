#!/usr/bin/env python3
"""
Carry medication and condition status across the department boundary.
====================================================================

The diagnostic showed the grounded synthesis arm false-alarming on a third of
the negative controls, while the centralized rule engine using the same
knowledge base on the same records scored precision 1.000. The difference is
that ``baselines._evidence_from_record`` filters on status --

    if m.active: ev.add_drug(...)
    if c.active: ev.add_condition(...)

-- and ``OrchestratorAgent._assemble_evidence`` has no equivalent, because the
department response schema carries no status field at all. A DISCONTINUED
medication or a RESOLVED condition crossed the boundary indistinguishable from
a current one, and the assembly step reinstated it as active. That is precisely
what discontinued_nsaid_on_warfarin, resolved_ckd_normal_function and
metformin_normal_renal were built to catch.

This adds the field to the response contract and honours it in assembly, and
writes a regression test that fails on the old behaviour.

Run once from the repository root:  python apply_status_fix.py
"""
import os
import sys
import re

ROOT = os.getcwd()
PKG = os.path.join(ROOT, "medagentnet")
if not os.path.isdir(PKG):
    PKG = ROOT  # already inside medagentnet/
    if not os.path.isdir(os.path.join(PKG, "agents")):
        sys.exit("run this from the repository root (the directory holding medagentnet/)")

changed = []


def edit(relpath, old, new, label, sentinel):
    p = os.path.join(PKG, relpath)
    s = open(p).read()
    if sentinel in s:
        print(f"  = {label}: already applied")
        return
    if old not in s:
        sys.exit(f"  ! {label}: anchor not found in {relpath}; patch not applied")
    open(p, "w").write(s.replace(old, new, 1))
    changed.append(relpath)
    print(f"  + {label}")


# ── 1. the response contract gains a status field ────────────────────────────
edit("llm/prompts.py",
     """   - medications_reported: medications relevant to the query (name, category, relevance)
   - conditions_reported: conditions relevant to the query""",
     """   - medications_reported: medications relevant to the query, each with
     name, category, relevance, and "active": true or false. Set active to
     false for anything marked DISCONTINUED.
   - conditions_reported: conditions relevant to the query, each with name and
     "active": true or false. Set active to false for anything marked
     RESOLVED.""",
     "response contract carries status",
     sentinel='Set active to')


# ── 2. assembly honours it ───────────────────────────────────────────────────
edit("agents/core.py",
     '''    def _assemble_evidence(self, context: dict, responses: list[AgentResponse]):''',
     '''    @staticmethod
    def _in_effect(item) -> bool:
        """Whether a reported item is current, per the responding department.

        The centralized comparator filters on ``m.active`` before evaluating
        any rule. The federated path could not, because the response schema
        carried no status, so a discontinued drug was reassembled as a current
        one and the drug-matched negative controls all raised alerts.

        A missing status counts as in effect: an agent that omits the field
        should not have its findings silently discarded.
        """
        if not isinstance(item, dict):
            return True
        v = item.get("active")
        if isinstance(v, bool):
            return v
        if isinstance(v, (int, float)):
            return bool(v)
        if isinstance(v, str) and v.strip().lower() in (
                "false", "no", "0", "discontinued", "resolved", "inactive"):
            return False
        s = str(item.get("status", "")).strip().lower()
        if s in ("discontinued", "resolved", "inactive", "stopped", "former",
                 "past", "ceased", "historical"):
            return False
        return True

    def _assemble_evidence(self, context: dict, responses: list[AgentResponse]):''',
     "_in_effect helper", sentinel="def _in_effect(")

edit("agents/core.py",
     """            for m in r.medications_reported:
                if isinstance(m, dict):
                    ev.add_drug(m.get("name", ""), m.get("category", ""), dept)
                    if not m.get("name") and m.get("category"):
                        ev.categories.add(str(m["category"]).lower())
                elif isinstance(m, str):
                    ev.add_drug(m, "", dept)
            for c in r.conditions_reported:
                if isinstance(c, dict):
                    ev.add_condition(c.get("name", ""), dept)
                elif isinstance(c, str):
                    ev.add_condition(c, dept)""",
     """            for m in r.medications_reported:
                if not self._in_effect(m):
                    continue
                if isinstance(m, dict):
                    ev.add_drug(m.get("name", ""), m.get("category", ""), dept)
                    if not m.get("name") and m.get("category"):
                        ev.categories.add(str(m["category"]).lower())
                elif isinstance(m, str):
                    ev.add_drug(m, "", dept)
            for c in r.conditions_reported:
                if not self._in_effect(c):
                    continue
                if isinstance(c, dict):
                    ev.add_condition(c.get("name", ""), dept)
                elif isinstance(c, str):
                    ev.add_condition(c, dept)""",
     "assembly skips items not in effect",
     sentinel="if not self._in_effect(m):")


# ── 3. regression test ───────────────────────────────────────────────────────
TEST = '''"""
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
'''

tp = os.path.join(PKG, "tests", "test_status_propagation.py")
os.makedirs(os.path.dirname(tp), exist_ok=True)
if os.path.exists(tp):
    print("  = regression test: already present")
else:
    open(tp, "w").write(TEST)
    changed.append("tests/test_status_propagation.py")
    print("  + regression test")

print("\nchanged:", ", ".join(changed) if changed else "nothing")
print("\nnext:")
print("  cd medagentnet && python -m pytest tests/ -q")
