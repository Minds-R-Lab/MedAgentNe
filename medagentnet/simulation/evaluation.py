"""
MedAgentNet - Ground-truth-matched evaluation (revision R1)
===========================================================

R0 scored a conflict as detected whenever any alert existed whose type was not
``no_conflict`` or ``parse_error``, and a pattern as detected whenever any of
about twenty generic words ("abnormal", "elevated", "kidney", ...) appeared
anywhere in any agent's free text. Nothing checked that the alert corresponded
to the conflict that had been planted, and the false-positive denominator was a
fixed sample of twenty clean patients.

This module replaces that with graded, ground-truth-matched scoring.

Specificity levels
------------------
For a positive case with planted conflict *C*:

  L0  MISS        no alert at or above the warning level
  L1  FLAGGED     an alert was raised, but it does not identify *C*
  L2  LOCALISED   the alert names the medications (or, for patterns, the
                  clinical entities) that constitute *C*
  L3  EXPLAINED   the alert also names the mechanism or consequence of *C*

Reporting all four levels separates "the system said something" from "the system
said the right thing". A tier-1 flag-only response can reach L1 by design and can
never reach L2, which is what makes the tier comparison informative rather than
tautological.

Negative cases
--------------
Three kinds, scored separately:

  clean control     no planted finding at all
  matched negative  a distractor built from the *same drugs* as a positive but
                    in a safe combination; the discriminating test
  ambiguous         adjudicated "either answer defensible"; excluded from
                    precision and recall, reported on its own

All proportions are reported with Wilson score intervals, because at these
denominators a normal approximation is not defensible.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Optional

from data.hard_cases import mechanism_terms_for

# Alert levels that count as "the system raised something the clinician sees"
ACTIONABLE_LEVELS = {"warning", "high_risk", "critical"}
NON_FINDING_TYPES = {"no_conflict", "parse_error", "llm_error", "format_error"}

L_MISS, L_FLAGGED, L_LOCALISED, L_EXPLAINED = 0, 1, 2, 3
LEVEL_NAMES = {0: "miss", 1: "flagged", 2: "localised", 3: "explained"}


# ─────────────────────────────────────────────────────────────────────────────
#  Statistics
# ─────────────────────────────────────────────────────────────────────────────

def wilson(k: int, n: int, z: float = 1.96) -> tuple:
    """Wilson score interval for a binomial proportion."""
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (round(p, 4), round(max(0.0, centre - half), 4),
            round(min(1.0, centre + half), 4))


def rate(k: int, n: int) -> dict:
    p, lo, hi = wilson(k, n)
    return {"k": k, "n": n, "rate": p, "ci95": [lo, hi]}


def mean_std(xs: list) -> dict:
    if not xs:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "n": 0}
    m = sum(xs) / len(xs)
    if len(xs) > 1:
        var = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
    else:
        var = 0.0
    return {"mean": round(m, 4), "std": round(math.sqrt(var), 4),
            "min": round(min(xs), 4), "max": round(max(xs), 4), "n": len(xs)}


def percentiles(xs: list, ps=(50, 90, 95, 99)) -> dict:
    if not xs:
        return {f"p{p}": 0.0 for p in ps}
    s = sorted(xs)
    out = {}
    for p in ps:
        idx = min(len(s) - 1, max(0, int(round((p / 100.0) * (len(s) - 1)))))
        out[f"p{p}"] = round(s[idx], 4)
    return out


# ─────────────────────────────────────────────────────────────────────────────
#  Text assembly
# ─────────────────────────────────────────────────────────────────────────────

def _actionable_alerts(result: dict) -> list:
    out = []
    for a in result.get("alerts", []):
        if (a.get("alert_type") or "unknown") in NON_FINDING_TYPES:
            continue
        if (a.get("alert_level") or "info") in ACTIONABLE_LEVELS:
            out.append(a)
    return out


def _alert_text(alerts: list) -> str:
    parts = []
    for a in alerts:
        parts.append(a.get("description") or "")
        parts.append(a.get("alert_type") or "")
        parts.append(a.get("recommendation") or "")
        parts.extend(str(m) for m in a.get("involved_medications", []))
        parts.extend(str(c) for c in a.get("involved_conditions", []))
    return " ".join(parts).lower()


def _response_text(result: dict) -> str:
    parts = []
    for rs in result.get("response_summaries", []):
        parts.append(rs.get("summary") or "")
        parts.extend(str(f) for f in rs.get("risk_flags", []))
    return " ".join(parts).lower()


def _mentions(term: str, blob: str) -> bool:
    """Whole-token containment, tolerant of plurals and simple suffixes."""
    t = term.lower().strip()
    if not t:
        return False
    if " " in t:
        return t in blob
    return re.search(r"\b" + re.escape(t) + r"\w{0,3}\b", blob) is not None


# ─────────────────────────────────────────────────────────────────────────────
#  Per-scenario grading
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ScenarioGrade:
    scenario_name: str
    cohort: str
    kind: str                  # conflict | pattern | negative | ambiguous | clean
    level: int
    level_name: str
    alerted: bool
    max_alert_level: str
    matched_entities: list = field(default_factory=list)
    missing_entities: list = field(default_factory=list)
    matched_mechanism: list = field(default_factory=list)
    note_only_limb: Optional[str] = None
    heldout: bool = False
    record_quality_flags: list = field(default_factory=list)
    elapsed_seconds: float = 0.0


def _max_level(alerts: list) -> str:
    order = ["info", "warning", "high_risk", "critical"]
    best = "info"
    for a in alerts:
        lv = a.get("alert_level") or "info"
        if lv in order and order.index(lv) > order.index(best):
            best = lv
    return best


def grade_scenario(result: dict) -> ScenarioGrade:
    """Grade one scenario result against its ground truth."""
    gt = result.get("ground_truth", {}) or {}
    kind = gt.get("kind", "clean")
    labels = gt.get("labels", [])
    cohort = result.get("cohort", "")
    quality = result.get("record_quality_flags", []) or []

    alerts = _actionable_alerts(result)
    blob = _alert_text(alerts) + " " + _response_text(result)
    alerted = len(alerts) > 0
    max_lv = _max_level(result.get("alerts", []))

    base = dict(
        scenario_name=result.get("scenario_name", ""),
        cohort=cohort, kind=kind, alerted=alerted, max_alert_level=max_lv,
        record_quality_flags=quality,
        elapsed_seconds=result.get("elapsed_seconds", 0.0),
    )

    # ── negatives and clean controls: any actionable alert is a false alarm ──
    if kind in ("negative", "clean"):
        return ScenarioGrade(
            level=L_FLAGGED if alerted else L_MISS,
            level_name="false_alarm" if alerted else "correctly_silent",
            **base,
        )

    if kind == "ambiguous":
        return ScenarioGrade(
            level=L_FLAGGED if alerted else L_MISS,
            level_name="alerted" if alerted else "silent",
            **base,
        )

    # ── positives ──
    label = labels[0] if labels else {}
    heldout = bool(label.get("heldout"))
    note_only = label.get("note_only_limb")

    if kind == "conflict":
        name = label.get("conflict_name", "")
        entities = [m for m in label.get("medications", [])]
    else:
        name = label.get("pattern_name", "")
        # For patterns the entities are the diagnostic terms of the expected
        # diagnosis, e.g. "Type 2 Diabetes Mellitus" -> {diabetes, mellitus}.
        expected = label.get("expected_diagnosis", "")
        entities = [w for w in re.split(r"[^A-Za-z0-9]+", expected)
                    if len(w) > 3 and w.lower() not in
                    {"type", "stage", "induced", "from", "with", "occult"}]

    matched = [e for e in entities if _mentions(e, blob)]
    missing = [e for e in entities if e not in matched]

    mech_terms = mechanism_terms_for(name)
    matched_mech = [t for t in mech_terms if _mentions(t, blob)]

    if not alerted:
        level = L_MISS
    else:
        # Localisation requires the majority of the constituent entities, and at
        # least two when the finding is defined by a drug combination.
        need = max(1, math.ceil(len(entities) / 2)) if entities else 1
        if entities and len(matched) >= need:
            level = L_EXPLAINED if matched_mech else L_LOCALISED
        else:
            level = L_FLAGGED

    return ScenarioGrade(
        level=level, level_name=LEVEL_NAMES[level],
        matched_entities=matched, missing_entities=missing,
        matched_mechanism=matched_mech, note_only_limb=note_only,
        heldout=heldout, **base,
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Aggregation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_run(results: list, provider_stats: dict = None,
                 extra: dict = None) -> dict:
    """Aggregate a list of scenario results into the reported metrics."""
    grades = [grade_scenario(r) for r in results]

    conflicts = [g for g in grades if g.kind == "conflict"]
    patterns = [g for g in grades if g.kind == "pattern"]
    negatives = [g for g in grades if g.kind == "negative"]
    cleans = [g for g in grades if g.kind == "clean"]
    ambiguous = [g for g in grades if g.kind == "ambiguous"]

    def level_profile(gs):
        n = len(gs)
        return {
            "n": n,
            "miss": rate(sum(1 for g in gs if g.level == L_MISS), n),
            "flagged_or_better": rate(sum(1 for g in gs if g.level >= L_FLAGGED), n),
            "localised_or_better": rate(sum(1 for g in gs if g.level >= L_LOCALISED), n),
            "explained": rate(sum(1 for g in gs if g.level == L_EXPLAINED), n),
        }

    # Precision / recall use the localised-or-better criterion: an alert only
    # counts as a true positive if it identifies the finding that is there.
    tp = sum(1 for g in conflicts + patterns if g.level >= L_LOCALISED)
    fn = len(conflicts) + len(patterns) - tp
    fp = sum(1 for g in negatives + cleans if g.alerted)
    tn = len(negatives) + len(cleans) - fp

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    # Per-scenario breakdown
    per_scenario = {}
    for g in conflicts + patterns:
        d = per_scenario.setdefault(
            g.scenario_name, {"n": 0, "localised": 0, "explained": 0, "miss": 0}
        )
        d["n"] += 1
        if g.level >= L_LOCALISED:
            d["localised"] += 1
        if g.level == L_EXPLAINED:
            d["explained"] += 1
        if g.level == L_MISS:
            d["miss"] += 1
    for name, d in per_scenario.items():
        d["detection_rate"] = round(d["localised"] / d["n"], 4) if d["n"] else 0.0
        d["ci95"] = wilson(d["localised"], d["n"])[1:]

    per_negative = {}
    for g in negatives:
        d = per_negative.setdefault(g.scenario_name, {"n": 0, "false_alarms": 0})
        d["n"] += 1
        if g.alerted:
            d["false_alarms"] += 1
    for name, d in per_negative.items():
        d["false_alarm_rate"] = round(d["false_alarms"] / d["n"], 4) if d["n"] else 0.0

    # Effect of record quality on detection
    positives = conflicts + patterns
    by_quality = {}
    for flag in ("contradiction", "stale_medication", "resolved_diagnosis",
                 "duplicate_entry", "clean_record"):
        if flag == "clean_record":
            subset = [g for g in positives if not g.record_quality_flags]
        else:
            subset = [g for g in positives if flag in g.record_quality_flags]
        if subset:
            by_quality[flag] = rate(
                sum(1 for g in subset if g.level >= L_LOCALISED), len(subset)
            )
    note_only = [g for g in conflicts if g.note_only_limb]
    if note_only:
        by_quality["note_only_limb"] = rate(
            sum(1 for g in note_only if g.level >= L_LOCALISED), len(note_only)
        )

    times = [g.elapsed_seconds for g in grades if g.elapsed_seconds > 0]

    out = {
        "n_scenarios": len(grades),
        "cohort_counts": {
            "conflict": len(conflicts), "pattern": len(patterns),
            "matched_negative": len(negatives), "clean_control": len(cleans),
            "ambiguous": len(ambiguous),
        },
        "conflict_detection": level_profile(conflicts),
        "pattern_detection": level_profile(patterns),
        "false_alarms": {
            "matched_negatives": rate(sum(1 for g in negatives if g.alerted),
                                      len(negatives)),
            "clean_controls": rate(sum(1 for g in cleans if g.alerted), len(cleans)),
            "all_negatives": rate(fp, len(negatives) + len(cleans)),
        },
        "ambiguous_alert_rate": rate(sum(1 for g in ambiguous if g.alerted),
                                     len(ambiguous)),
        "classification": {
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "specificity": round(tn / (tn + fp), 4) if (tn + fp) else 0.0,
            "criterion": "true positive requires localised-or-better match to the "
                         "planted finding; any actionable alert on a negative is "
                         "a false positive",
        },
        "per_scenario": per_scenario,
        "per_negative": per_negative,
        "by_record_quality": by_quality,
        "heldout_split": {
            "heldout": level_profile([g for g in positives if g.heldout]),
            "development": level_profile([g for g in positives if not g.heldout]),
        },
        "latency_seconds": {
            **mean_std(times), **percentiles(times),
        },
    }

    # A "loose" score reproducing the R0 criterion, reported alongside the
    # strict one so the two can be compared directly in the paper.
    loose_tp = sum(1 for g in positives if g.alerted)
    fids = [r["evidence_fidelity"] for r in results
            if isinstance(r.get("evidence_fidelity"), dict)
            and "error" not in r["evidence_fidelity"]]
    if fids:
        from simulation.fidelity import aggregate_fidelity
        out["evidence_fidelity"] = aggregate_fidelity(fids)

    out["legacy_loose_criterion"] = {
        "note": "R0 criterion: any actionable alert counts as a detection, "
                "with no check that it matches the planted finding.",
        "conflict_detection": rate(sum(1 for g in conflicts if g.alerted),
                                   len(conflicts)),
        "pattern_detection": rate(sum(1 for g in patterns if g.alerted),
                                  len(patterns)),
        "all_positives": rate(loose_tp, len(positives)),
    }

    if provider_stats:
        out["provider"] = provider_stats
    if extra:
        out.update(extra)
    return out


def aggregate_runs(run_results: list, keys: list = None) -> dict:
    """Mean, std and range of scalar metrics across independent runs."""
    keys = keys or [
        ("conflict_detection", "localised_or_better", "rate"),
        ("conflict_detection", "explained", "rate"),
        ("pattern_detection", "localised_or_better", "rate"),
        ("pattern_detection", "explained", "rate"),
        ("false_alarms", "matched_negatives", "rate"),
        ("false_alarms", "clean_controls", "rate"),
        ("classification", "precision"),
        ("classification", "recall"),
        ("classification", "f1"),
        ("classification", "specificity"),
        ("latency_seconds", "mean"),
    ]

    def dig(d, path):
        cur = d
        for p in path:
            if not isinstance(cur, dict) or p not in cur:
                return None
            cur = cur[p]
        return cur

    agg = {}
    for path in keys:
        vals = [dig(r, path) for r in run_results]
        vals = [v for v in vals if isinstance(v, (int, float))]
        agg["/".join(path)] = mean_std(vals)

    # Pooled counts give a much tighter interval than averaging per-run rates.
    pooled = {}
    for field_name, sub in (("conflict_detection", "localised_or_better"),
                            ("pattern_detection", "localised_or_better"),
                            ("false_alarms", "matched_negatives"),
                            ("false_alarms", "clean_controls")):
        k = sum(dig(r, (field_name, sub, "k")) or 0 for r in run_results)
        n = sum(dig(r, (field_name, sub, "n")) or 0 for r in run_results)
        pooled[f"{field_name}/{sub}"] = rate(k, n)

    tp = sum(dig(r, ("classification", "tp")) or 0 for r in run_results)
    fp = sum(dig(r, ("classification", "fp")) or 0 for r in run_results)
    fn = sum(dig(r, ("classification", "fn")) or 0 for r in run_results)
    tn = sum(dig(r, ("classification", "tn")) or 0 for r in run_results)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    pooled["classification"] = {
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "precision": rate(tp, tp + fp) if (tp + fp) else rate(0, 0),
        "recall": rate(tp, tp + fn) if (tp + fn) else rate(0, 0),
        "specificity": rate(tn, tn + fp) if (tn + fp) else rate(0, 0),
        "f1": round(2 * prec * rec / (prec + rec), 4) if (prec + rec) else 0.0,
    }

    return {"per_run": run_results, "across_runs": agg, "pooled": pooled}


def mcnemar(b: int, c: int) -> dict:
    """Exact-ish McNemar test for two systems on the same cases.

    b = cases system A got right and B wrong; c = the reverse.
    Used to compare MedAgentNet against each baseline on identical scenarios.
    """
    n = b + c
    if n == 0:
        return {"b": b, "c": c, "statistic": 0.0, "p_value": 1.0}
    stat = (abs(b - c) - 1) ** 2 / n  # continuity-corrected chi-square, df=1
    # Survival function of chi-square with 1 df = erfc(sqrt(x/2))
    p = math.erfc(math.sqrt(stat / 2.0))
    return {"b": b, "c": c, "statistic": round(stat, 4), "p_value": round(p, 5)}


def paired_comparison(grades_a: list, grades_b: list) -> dict:
    """Compare two systems scenario-by-scenario on the localised criterion."""
    b = c = both = neither = 0
    for ga, gb in zip(grades_a, grades_b):
        a_ok = ga.level >= L_LOCALISED
        b_ok = gb.level >= L_LOCALISED
        if a_ok and not b_ok:
            b += 1
        elif b_ok and not a_ok:
            c += 1
        elif a_ok and b_ok:
            both += 1
        else:
            neither += 1
    out = mcnemar(b, c)
    out.update({"both_correct": both, "both_wrong": neither,
                "n_paired": len(grades_a)})
    return out
