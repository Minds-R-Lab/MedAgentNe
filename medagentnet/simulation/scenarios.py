"""
MedAgentNet - Query construction (revision R1)
==============================================

In R0 the scenario driver built each query context out of the ground-truth
label:

    context = {
        "planned_procedure": conflict["trigger_procedure"],   # from the label
        "reason":            conflict["description"],         # the answer
    }
    for med in patient.medications:                          # every department's
        context[f"current_med_{med.name}"] = med.name        # drugs, at any tier

and for patterns it additionally passed ``"expected": expected_diagnosis`` and a
pre-computed ``rising``/``declining`` trend string. That context was rendered
verbatim into every agent's prompt at every disclosure tier.

This module replaces that with a query that a clinician could actually have
issued. Three properties are enforced:

**P1 - No label content.** The context is built only from the patient's own
record and from a department-level procedure vocabulary. No field of
``known_conflicts`` / ``known_patterns`` is read.

**P2 - Schema invariance.** Positive, distractor, ambiguous and control cases
produce contexts with identical key sets. In R0 a positive case carried a
``reason`` key and a set of ``current_med_*`` keys that negatives lacked, so the
cohorts were separable from the prompt shape alone, before any clinical content
was read.

**P3 - No cross-department pre-aggregation.** The context never contains data
from a department other than the one initiating the encounter. Retrieving the
rest is the job the architecture is supposed to do.

``build_*_scenario`` returns a ``ScenarioSpec``; the runner turns it into a
query. ``LEGACY_MODE`` reproduces the R0 construction so the size of the R0 leak
can be quantified rather than merely asserted.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Optional

from protocol.models import PatientRecord


# Procedures and review reasons a department can plausibly initiate. These are
# properties of the department, not of the patient's diagnosis.
DEPARTMENT_ENCOUNTERS = {
    "cardiology": [
        ("anticoagulation_review", "Anticoagulation clinic review."),
        ("hypertension_review", "Blood pressure review."),
        ("arrhythmia_assessment", "Assessment of rhythm disturbance."),
        ("pre_operative_cardiac_clearance", "Cardiac clearance before surgery."),
    ],
    "dental": [
        ("tooth_extraction", "Extraction planned at this visit."),
        ("dental_implant", "Implant placement planned."),
        ("dental_hygiene_visit", "Scale and polish, no invasive work planned."),
        ("dental_pain_management", "Analgesia required for dental pain."),
    ],
    "general_practice": [
        ("annual_health_check", "Annual health check."),
        ("medication_review", "Structured medication review."),
        ("acute_infection_review", "Review of an acute infective episode."),
        ("hypertension_review", "Blood pressure review."),
    ],
    "endocrinology": [
        ("diabetes_review", "Diabetes clinic review."),
        ("thyroid_review", "Thyroid function review."),
        ("metabolic_assessment", "Metabolic assessment."),
    ],
    "ophthalmology": [
        ("glaucoma_management", "Intraocular pressure management."),
        ("retinal_screening", "Retinal screening appointment."),
        ("cataract_assessment", "Assessment for cataract surgery."),
    ],
    "nephrology": [
        ("renal_assessment", "Renal function assessment."),
        ("dialysis_planning", "Planning for renal replacement therapy."),
        ("emergency_evaluation", "Urgent assessment of deteriorating renal function."),
    ],
    "neurology": [
        ("seizure_review", "Seizure control review."),
        ("neuropathy_assessment", "Assessment of peripheral nerve symptoms."),
        ("headache_assessment", "Assessment of headache."),
    ],
    "rheumatology": [
        ("pain_management", "Review of analgesic strategy."),
        ("dmard_monitoring", "Disease-modifying therapy monitoring."),
        ("immunosuppression_monitoring", "Immunosuppression monitoring."),
    ],
    "pulmonology": [
        ("respiratory_assessment", "Respiratory function assessment."),
        ("inhaler_review", "Inhaler technique and therapy review."),
    ],
    "laboratory": [
        ("result_review", "Review of recently reported results."),
    ],
}

DEFAULT_ENCOUNTER = [("clinical_review", "Routine clinical review.")]

# One reason string, used for every cross-departmental pattern query regardless
# of cohort, so that positives and controls are indistinguishable by wording.
PATTERN_QUERY_REASON = (
    "Cross-departmental review requested during this encounter. Report any "
    "findings from your department that could contribute to a multi-system "
    "picture, including longitudinal trends in your own results."
)

CONFLICT_QUERY_REASON = (
    "Pre-treatment safety check for the encounter above. Report any medication, "
    "condition or result in your department that could interact with, "
    "contraindicate or complicate management."
)


@dataclass
class ScenarioSpec:
    """A query a clinician issues, plus the bookkeeping the scorer needs."""
    patient: PatientRecord
    requesting_department: str
    clinical_context: dict
    query_type: str
    is_emergency: bool
    scenario_name: str
    cohort: str
    # Evaluation-side only; never reaches an agent.
    ground_truth: dict = field(default_factory=dict)


def _choose_encounter(rng: random.Random, patient: PatientRecord,
                      preferred_department: Optional[str] = None):
    """Pick the department initiating the encounter, and what it is doing.

    The choice depends only on which departments the patient attends, never on
    what is wrong with them.
    """
    hint = getattr(patient, "encounter_hint", None) or {}
    dept = preferred_department or hint.get("department")
    if not dept or dept not in patient.departments:
        candidates = [d for d in patient.departments] or ["general_practice"]
        dept = rng.choice(candidates)

    options = DEPARTMENT_ENCOUNTERS.get(dept, DEFAULT_ENCOUNTER)
    if hint.get("procedure"):
        for proc, desc in options:
            if proc == hint["procedure"]:
                return dept, proc, desc
    proc, desc = rng.choice(options)
    return dept, proc, desc


def _base_context(procedure: str, description: str, reason: str) -> dict:
    """The invariant context schema shared by every cohort."""
    return {
        "planned_procedure": procedure,
        "relevant_categories": ["all"],
        "query_reason": f"{description} {reason}",
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Scenario builders
# ─────────────────────────────────────────────────────────────────────────────

def build_safety_scenario(patient: PatientRecord, rng: random.Random,
                          preferred_department: Optional[str] = None,
                          is_emergency: bool = False) -> ScenarioSpec:
    """A pre-treatment safety check. Used for conflict, distractor, ambiguous
    and control patients alike, with an identical context schema."""
    dept, proc, desc = _choose_encounter(rng, patient, preferred_department)
    context = _base_context(proc, desc, CONFLICT_QUERY_REASON)

    gt = {}
    if patient.known_conflicts:
        gt = {"kind": "conflict", "labels": list(patient.known_conflicts)}
        name = patient.known_conflicts[0]["conflict_name"]
    elif patient.negative_controls:
        gt = {"kind": "negative", "labels": list(patient.negative_controls)}
        name = patient.negative_controls[0]["control_name"]
    elif patient.ambiguous_cases:
        gt = {"kind": "ambiguous", "labels": list(patient.ambiguous_cases)}
        name = patient.ambiguous_cases[0]["case_name"]
    else:
        gt = {"kind": "clean", "labels": []}
        name = "clean_control"

    return ScenarioSpec(
        patient=patient,
        requesting_department=dept,
        clinical_context=context,
        query_type="MED_CONFLICT",
        is_emergency=is_emergency,
        scenario_name=name,
        cohort=getattr(patient, "cohort", ""),
        ground_truth=gt,
    )


def build_pattern_scenario(patient: PatientRecord, rng: random.Random,
                           preferred_department: Optional[str] = None) -> ScenarioSpec:
    """A cross-departmental pattern query, issued identically for pattern
    patients and for controls."""
    dept, proc, desc = _choose_encounter(rng, patient, preferred_department)
    context = _base_context(proc, desc, PATTERN_QUERY_REASON)

    if patient.known_patterns:
        gt = {"kind": "pattern", "labels": list(patient.known_patterns)}
        name = patient.known_patterns[0]["pattern_name"]
    else:
        gt = {"kind": "clean", "labels": []}
        name = "clean_control"

    return ScenarioSpec(
        patient=patient,
        requesting_department=dept,
        clinical_context=context,
        query_type="LONG_PATTERN",
        is_emergency=False,
        scenario_name=name,
        cohort=getattr(patient, "cohort", ""),
        ground_truth=gt,
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Legacy (R0) construction, retained to quantify the leak
# ─────────────────────────────────────────────────────────────────────────────

def build_legacy_conflict_context(patient: PatientRecord, conflict: dict) -> dict:
    """Exact reproduction of the R0 context, including the ground-truth
    description and the cross-department medication list."""
    context = {
        "planned_procedure": conflict.get("trigger_procedure", "evaluation"),
        "relevant_categories": ["all"],
        "reason": conflict.get("description", ""),
    }
    for med in patient.medications:
        context[f"current_med_{med.name.lower().replace(' ', '_')}"] = med.name
    return context


def build_legacy_pattern_context(patient: PatientRecord, pattern: dict) -> dict:
    """Exact reproduction of the R0 pattern context."""
    context = {
        "pattern_category": pattern.get("pattern_type", "diagnostic"),
        "departments_involved": pattern.get("departments", []),
        "expected": pattern.get("expected_diagnosis", ""),
        "query_reason": (
            "Cross-departmental pattern analysis requested. Looking for "
            "connections between findings across "
            f"{', '.join(pattern.get('departments', []))}. Check for trends, "
            "rising/declining values, and multi-system disease patterns."
        ),
    }
    by_code = {}
    for lab in patient.lab_results:
        by_code.setdefault(lab.test_code, []).append(lab)
    for code, labs in by_code.items():
        ordered = sorted(labs, key=lambda x: x.date)
        values = [l.value for l in ordered]
        if len(values) > 1:
            trend = "rising" if values[-1] > values[0] else "declining"
            context[f"lab_trend_{code.lower()}"] = (
                f"{ordered[0].test_name}: {' -> '.join(str(v) for v in values)} ({trend})"
            )
    for cond in patient.conditions:
        context[f"condition_{cond.department}_{cond.code}"] = (
            f"{cond.name} (severity: {cond.severity}, dept: {cond.department})"
        )
    for med in patient.medications:
        context[f"medication_{med.department}_{med.name.lower().replace(' ', '_')}"] = (
            f"{med.name} ({med.category}), prescribed by {med.department}"
        )
    return context


def context_leak_report(context: dict, patient: PatientRecord) -> dict:
    """Measure how much of a context is label-derived or cross-departmental.

    Reported for both constructions so the paper can state the size of the R0
    leak numerically instead of describing it.
    """
    from agents.core import DepartmentAgent
    allowed = set(DepartmentAgent.ALLOWED_CONTEXT_KEYS)

    label_terms = set()
    for c in patient.known_conflicts:
        label_terms.update(str(c.get("conflict_name", "")).lower().split("_"))
        for m in c.get("medications", []):
            label_terms.add(str(m).lower())
    for p in patient.known_patterns:
        label_terms.update(str(p.get("expected_diagnosis", "")).lower().split())
        label_terms.update(str(p.get("pattern_name", "")).lower().split("_"))
    label_terms = {t for t in label_terms if len(t) > 3}

    blob = " ".join(f"{k} {v}" for k, v in context.items()).lower()
    return {
        "n_keys": len(context),
        "n_disallowed_keys": sum(1 for k in context if k not in allowed),
        "label_terms_present": sorted(t for t in label_terms if t in blob),
        "n_label_terms_present": sum(1 for t in label_terms if t in blob),
        "carries_ground_truth_text": any(
            k in context for k in ("reason", "expected", "departments_involved")
        ),
    }
