"""Tests for the graded, ground-truth-matched scorer.

R0 counted a conflict as detected whenever any alert existed whose type was not
a sentinel value, and a pattern whenever any of about twenty generic words
appeared anywhere in any agent's free text. These tests pin the replacement.
"""
import pytest

from simulation.evaluation import (
    grade_scenario, evaluate_run, wilson, mcnemar,
    L_MISS, L_FLAGGED, L_LOCALISED, L_EXPLAINED,
)


def _scenario(alerts, kind="conflict", name="triple_whammy_aki",
              meds=("Lisinopril", "Ibuprofen", "Hydrochlorothiazide"),
              summaries=None):
    label = ({"conflict_name": name, "medications": list(meds), "departments": []}
             if kind == "conflict" else
             {"pattern_name": name, "expected_diagnosis": "Type 2 Diabetes Mellitus",
              "departments": []})
    return {
        "scenario_name": name, "cohort": kind, "elapsed_seconds": 0.1,
        "ground_truth": {"kind": kind, "labels": [label]},
        "record_quality_flags": [],
        "alerts": alerts, "num_alerts": len(alerts),
        "response_summaries": summaries or [],
    }


def _alert(desc, level="high_risk", atype="cross_department_interaction", meds=()):
    return {"alert_level": level, "alert_type": atype, "description": desc,
            "involved_medications": list(meds), "involved_conditions": [],
            "recommendation": ""}


def test_no_alert_is_a_miss():
    assert grade_scenario(_scenario([])).level == L_MISS


def test_vague_alert_does_not_count_as_detection():
    """The R0 criterion in one assertion: an unrelated alert must not score."""
    g = grade_scenario(_scenario([_alert("Clinical risk factor identified")]))
    assert g.level == L_FLAGGED
    assert g.level < L_LOCALISED


def test_naming_the_medications_reaches_localised():
    g = grade_scenario(_scenario([
        _alert("Lisinopril and Ibuprofen and Hydrochlorothiazide together",
               meds=("Lisinopril", "Ibuprofen", "Hydrochlorothiazide"))]))
    assert g.level >= L_LOCALISED


def test_naming_the_mechanism_reaches_explained():
    g = grade_scenario(_scenario([
        _alert("Lisinopril, Ibuprofen and Hydrochlorothiazide: nephrotoxic "
               "combination causing acute kidney injury")]))
    assert g.level == L_EXPLAINED
    assert "kidney" in " ".join(g.matched_mechanism) or g.matched_mechanism


def test_info_level_alert_is_not_actionable():
    g = grade_scenario(_scenario([
        _alert("Lisinopril Ibuprofen Hydrochlorothiazide", level="info")]))
    assert g.level == L_MISS


def test_sentinel_alert_types_are_ignored():
    for t in ("no_conflict", "parse_error", "llm_error", "format_error"):
        g = grade_scenario(_scenario([_alert("something", atype=t)]))
        assert g.level == L_MISS, t


def test_generic_keyword_in_a_summary_does_not_detect_a_pattern():
    """R0 passed a pattern if any agent said 'abnormal' or 'elevated'."""
    s = _scenario([], kind="pattern", name="undiagnosed_diabetes",
                  summaries=[{"summary": "Elevated result noted, abnormal kidney value",
                              "risk_flags": []}])
    assert grade_scenario(s).level == L_MISS


def test_any_alert_on_a_negative_is_a_false_positive():
    neg = {"scenario_name": "warfarin_paracetamol_safe", "cohort": "distractor",
           "elapsed_seconds": 0.1, "record_quality_flags": [],
           "ground_truth": {"kind": "negative", "labels": [
               {"control_name": "warfarin_paracetamol_safe", "mirrors": "x",
                "medications": ["Warfarin", "Paracetamol"]}]},
           "alerts": [_alert("Warfarin present, bleeding risk")], "num_alerts": 1,
           "response_summaries": []}
    g = grade_scenario(neg)
    assert g.alerted and g.level_name == "false_alarm"


def test_ambiguous_cases_are_excluded_from_precision_and_recall():
    amb = {"scenario_name": "borderline_hba1c_prediabetes", "cohort": "ambiguous",
           "elapsed_seconds": 0.1, "record_quality_flags": [],
           "ground_truth": {"kind": "ambiguous", "labels": [
               {"case_name": "borderline_hba1c_prediabetes", "adjudication": "either"}]},
           "alerts": [_alert("Rising HbA1c")], "num_alerts": 1,
           "response_summaries": []}
    ev = evaluate_run([amb])
    assert ev["classification"]["tp"] == 0
    assert ev["classification"]["fp"] == 0
    assert ev["ambiguous_alert_rate"]["n"] == 1


def test_loose_criterion_is_reported_alongside_the_strict_one():
    """The paper compares the two directly, so both must be computed."""
    ev = evaluate_run([_scenario([_alert("Clinical risk factor identified")])])
    assert ev["classification"]["recall"] == 0.0
    assert ev["legacy_loose_criterion"]["conflict_detection"]["rate"] == 1.0


def test_wilson_interval_brackets_the_point_estimate():
    p, lo, hi = wilson(1, 60)
    assert lo < p < hi and lo >= 0.0 and hi <= 1.0
    assert wilson(0, 20)[0] == 0.0 and wilson(0, 20)[2] > 0.0


def test_mcnemar_is_symmetric_and_bounded():
    a, b = mcnemar(10, 0), mcnemar(0, 10)
    assert a["statistic"] == b["statistic"]
    assert 0.0 <= a["p_value"] <= 1.0
    assert mcnemar(5, 5)["p_value"] > 0.5
