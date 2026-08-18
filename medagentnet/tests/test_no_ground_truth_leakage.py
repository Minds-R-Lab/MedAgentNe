"""Regression tests for the R0 defect in which the ground truth reached the
agent prompts.

In R0 the scenario driver built each query context out of the label:

    context["reason"]   = conflict["description"]        # the answer, in words
    context["expected"] = pattern["expected_diagnosis"]  # the answer, verbatim
    context[f"current_med_{name}"] = name                # every department's drugs

and ``DepartmentAgent._build_user_prompt`` rendered the whole context into every
prompt at every disclosure tier. These tests fail if any of that returns.
"""
import random
import pytest

from data.generator_hard import HardCaseGenerator
from simulation import scenarios as scen
from simulation.runner_v2 import HardRunner
from agents.core import DepartmentAgent
from protocol.models import ClinicalQuery


@pytest.fixture(scope="module")
def cohort():
    gen = HardCaseGenerator(config_dir="config", seed=7)
    return gen.generate(40)


def test_context_contains_no_label_fields(cohort):
    """No query context may carry a field derived from the ground truth."""
    rng = random.Random(0)
    forbidden = {"reason", "expected", "departments_involved", "pattern_category"}
    for patient in cohort:
        for spec in (scen.build_safety_scenario(patient, rng),
                     scen.build_pattern_scenario(patient, rng)):
            assert not (set(spec.clinical_context) & forbidden), (
                f"{spec.scenario_name}: context carries {set(spec.clinical_context) & forbidden}")


def test_context_carries_no_cross_department_medications(cohort):
    """A context must not pre-aggregate other departments' medications.

    R0 copied every medication into the context with the comment
    "so agents can detect them", which removed the need for any exchange.
    """
    rng = random.Random(1)
    for patient in cohort:
        spec = scen.build_safety_scenario(patient, rng)
        blob = " ".join(f"{k} {v}" for k, v in spec.clinical_context.items()).lower()
        for med in patient.medications:
            assert med.name.lower() not in blob, (
                f"{med.name} leaked into the query context")


def test_context_schema_is_identical_across_cohorts(cohort):
    """Positives and negatives must not be separable by prompt shape.

    In R0 a positive context carried a `reason` key and a set of
    `current_med_*` keys that negatives lacked, so the cohorts were separable
    before any clinical content was read.
    """
    rng = random.Random(2)
    schemas = {}
    for patient in cohort:
        spec = scen.build_safety_scenario(patient, rng)
        schemas.setdefault(getattr(patient, "cohort", "?"), set()).add(
            frozenset(spec.clinical_context))
    all_schemas = set().union(*schemas.values())
    assert len(all_schemas) == 1, f"context key sets differ by cohort: {schemas}"


def test_expected_diagnosis_never_appears_in_a_prompt(cohort):
    """The rendered prompt must not contain the target diagnosis."""
    runner = HardRunner(seed=7, num_patients=0)
    rng = random.Random(3)
    for patient in cohort:
        if not patient.known_patterns:
            continue
        expected = patient.known_patterns[0]["expected_diagnosis"].lower()
        spec = scen.build_pattern_scenario(patient, rng)
        for dept in patient.departments:
            agent = runner.department_agents.get(dept)
            if agent is None:
                continue
            agent.load_patient_data(patient)
            q = ClinicalQuery(source_agent=spec.requesting_department,
                              target_agent=dept, patient_id=patient.patient_id,
                              query_type=spec.query_type,
                              clinical_context=spec.clinical_context,
                              disclosure_tier=3)
            prompt = agent._build_user_prompt(q, agent.patient_store[patient.patient_id], 3)
            # The record may legitimately contain a diagnosis name; the context
            # section may not.
            ctx_section = prompt.split("PATIENT DATA")[0].lower()
            assert expected not in ctx_section, (
                f"expected diagnosis '{expected}' present in the query context")


def test_whitelist_drops_unknown_context_fields():
    """The agent must discard any field a clinician would not supply.

    This is the structural guard: even if a future scenario builder
    reintroduced a label field, it would not reach the prompt.
    """
    runner = HardRunner(seed=7, num_patients=0)
    agent = next(iter(runner.department_agents.values()))
    dirty = {"planned_procedure": "tooth_extraction",
             "reason": "Patient on Warfarin needs dental extraction (bleeding risk)",
             "expected": "Type 2 Diabetes Mellitus",
             "current_med_warfarin": "Warfarin"}
    clean, dropped = agent.sanitize_context(dirty)
    assert set(clean) == {"planned_procedure"}
    assert set(dropped) == {"reason", "expected", "current_med_warfarin"}


def test_ground_truth_is_absent_from_the_department_slice(cohort):
    """A department's view of a patient must not include the label."""
    for patient in cohort:
        for dept in patient.departments:
            slice_ = patient.get_department_records(dept)
            for field in ("known_conflicts", "known_patterns",
                          "negative_controls", "ambiguous_cases", "cohort"):
                assert field not in slice_, f"{field} present in {dept} slice"
