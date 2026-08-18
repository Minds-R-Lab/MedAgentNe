"""
MedAgentNet - Hard-case cohort generator (revision R1)
======================================================

Extends ``PatientDataGenerator`` with everything the R0 benchmark was missing:

* drug-matched negative controls and ambiguous cases,
* a held-out scenario family,
* free-text clinical notes carrying part of the signal,
* record corruption (missing fields, contradictions, stale prescriptions,
  resolved diagnoses, brand/generic duplicates),
* seed-dependent template assignment, so that repeated seeds produce genuinely
  different case mixes rather than a fixed round-robin.

Every patient carries a ``cohort`` label and, for positives, a structured
ground-truth entry. The ground truth is consumed only by
``simulation/evaluation.py``; nothing here is written into a query context.
"""
from __future__ import annotations

import os
import json
import random
from datetime import datetime, timedelta
from typing import Optional

from protocol.models import (
    PatientRecord, Medication, Condition, LabResult, Visit
)
from data.generator import (
    PatientDataGenerator, CONFLICT_TEMPLATES, PATTERN_TEMPLATES,
)
from data.hard_cases import (
    TEMPLATE_REPAIRS, DISTRACTOR_TEMPLATES, AMBIGUOUS_TEMPLATES,
    HELDOUT_CONFLICT_TEMPLATES, HELDOUT_PATTERN_TEMPLATES,
    CLINICAL_NOTE_TEMPLATES, NOISE_OPERATORS, BRAND_NAMES,
)

# Cohort identifiers used throughout the evaluation
COHORT_CONFLICT = "conflict"
COHORT_PATTERN = "pattern"
COHORT_DISTRACTOR = "distractor"
COHORT_AMBIGUOUS = "ambiguous"
COHORT_CONTROL = "control"

DEFAULT_COHORT_MIX = {
    COHORT_CONFLICT: 0.30,
    COHORT_PATTERN: 0.20,
    COHORT_DISTRACTOR: 0.25,
    COHORT_AMBIGUOUS: 0.05,
    COHORT_CONTROL: 0.20,
}


def _apply_repairs(template: dict) -> dict:
    """Apply the R1 repairs to an R0 conflict template."""
    repair = TEMPLATE_REPAIRS.get(template["name"])
    if not repair:
        return template
    t = json.loads(json.dumps(template))
    if "medication_overrides" in repair:
        t["medications"] = repair["medication_overrides"]
    if "description_override" in repair:
        t["description"] = repair["description_override"]
    if "extra_findings" in repair:
        t.setdefault("findings", {})
        for dept, payload in repair["extra_findings"].items():
            dest = t["findings"].setdefault(dept, {})
            for key, items in payload.items():
                dest.setdefault(key, []).extend(items)
        for dept in repair["extra_findings"]:
            if dept not in t["departments"]:
                t["departments"].append(dept)
    return t


REPAIRED_CONFLICT_TEMPLATES = [_apply_repairs(t) for t in CONFLICT_TEMPLATES]


class HardCaseGenerator(PatientDataGenerator):
    """Generates the R1 evaluation cohorts."""

    def __init__(self, config_dir: str = "config",
                 seed: Optional[int] = None,
                 cohort_mix: Optional[dict] = None,
                 noise_rates: Optional[dict] = None,
                 use_heldout: bool = False,
                 enable_notes: bool = True):
        super().__init__(config_dir=config_dir)
        if seed is not None:
            self.rng = random.Random(seed)
        self.seed = seed if seed is not None else self.settings["random_seed"]
        self.cohort_mix = dict(cohort_mix or DEFAULT_COHORT_MIX)
        self.noise_rates = {
            k: v["default_rate"] for k, v in NOISE_OPERATORS.items()
        }
        if noise_rates is not None:
            self.noise_rates.update(noise_rates)
        self.use_heldout = use_heldout
        self.enable_notes = enable_notes

        self.conflict_pool = (
            HELDOUT_CONFLICT_TEMPLATES if use_heldout else REPAIRED_CONFLICT_TEMPLATES
        )
        self.pattern_pool = (
            HELDOUT_PATTERN_TEMPLATES if use_heldout else PATTERN_TEMPLATES
        )

    # ── helpers ──────────────────────────────────────────────────────────

    def _note(self, kind: str, **fmt) -> str:
        pool = CLINICAL_NOTE_TEMPLATES.get(kind, CLINICAL_NOTE_TEMPLATES["routine"])
        return self.rng.choice(pool).format(**fmt)

    def _attach_note(self, patient: PatientRecord, dept: str, text: str):
        """Attach a note to an existing visit in `dept`, or create one."""
        candidates = [v for v in patient.visits if v.department == dept]
        if candidates:
            visit = self.rng.choice(candidates)
            visit.notes = (visit.notes + " " + text).strip() if visit.notes else text
        else:
            patient.visits.append(Visit(
                department=dept,
                date=self._random_date(1, 12),
                reason=f"{self.dept_config.get(dept, {}).get('name', dept)} review",
                physician=self._random_physician(dept),
                notes=text,
            ))

    def _resolve_dept(self, dept: str) -> str:
        """Map a template's department onto one that exists in this federation.

        The agent-count sweep runs with a reduced set of departments; a scenario
        that names a department which is not deployed is reassigned to general
        practice rather than dropped, so the clinical content is preserved and
        only the number of hops changes.
        """
        if dept in self.dept_config:
            return dept
        for fallback in ("general_practice", "laboratory"):
            if fallback in self.dept_config:
                return fallback
        return next(iter(self.dept_config))

    def _random_physician(self, dept: str) -> str:
        return super()._random_physician(self._resolve_dept(dept))

    def _ensure_departments(self, patient: PatientRecord, depts):
        for d in depts:
            d = self._resolve_dept(d)
            if d not in patient.departments:
                patient.departments.append(d)
        patient.departments.sort()

    def _add_medication(self, patient: PatientRecord, med_info: dict,
                        active: bool = True):
        existing = [m for m in patient.medications if m.name == med_info["name"]]
        if existing:
            return existing[0]
        stopped = med_info.get("stopped_months_ago")
        med = Medication(
            name=med_info["name"],
            category=med_info["category"],
            dose=med_info.get("dose", "10mg"),
            frequency=med_info.get("frequency", "daily"),
            prescribed_by=self._random_physician(med_info["dept"]),
            prescribed_date=self._random_date(
                (stopped or 1) + 1, (stopped or 1) + 18
            ),
            department=self._resolve_dept(med_info["dept"]),
            active=med_info.get("active", active),
        )
        patient.medications.append(med)
        if not med.active and self.enable_notes:
            self._attach_note(
                patient, self._resolve_dept(med_info["dept"]),
                self._note("stopped_med", drug=med.name,
                           months=stopped or self.rng.randint(2, 10)),
            )
        return med

    def _add_findings(self, patient: PatientRecord, findings: dict):
        """Plant conditions and longitudinal labs from a findings block."""
        for dept, payload in findings.items():
            dept = self._resolve_dept(dept)
            self._ensure_departments(patient, [dept])
            for cond in payload.get("conditions", []):
                if any(c.code == cond["code"] for c in patient.conditions):
                    continue
                patient.conditions.append(Condition(
                    code=cond["code"],
                    name=cond["name"],
                    severity=cond["severity"],
                    diagnosed_date=self._random_date(3, 24),
                    diagnosed_by=self._random_physician(dept),
                    department=dept,
                    active=cond.get("active", True),
                ))
            for lab in payload.get("lab_results", []):
                # A planted longitudinal series replaces any baseline draw of the
                # same assay. Otherwise the baseline value (drawn independently
                # and dated more recently) lands at the end of the series and
                # inverts the trend.
                patient.lab_results = [
                    l for l in patient.lab_results
                    if l.test_code != lab["test_code"]
                ]
                n = len(lab["values"])
                gap = lab.get("months_apart", 4)
                base = datetime.now() - timedelta(days=n * gap * 30)
                nr = tuple(lab.get("normal_range", (0, 0)))
                for i, val in enumerate(lab["values"]):
                    # jitter the sampling interval so series are not perfectly regular
                    jitter = self.rng.randint(-9, 9)
                    d = base + timedelta(days=i * gap * 30 + jitter)
                    patient.lab_results.append(LabResult(
                        test_code=lab["test_code"],
                        test_name=lab["test_name"],
                        value=val,
                        unit=lab.get("unit", ""),
                        normal_range=nr,
                        date=d.strftime("%Y-%m-%d"),
                        is_abnormal=(val < nr[0] or val > nr[1]) if nr != (0, 0) else False,
                        department="laboratory" if dept == "laboratory" else dept,
                    ))

    # ── baseline patient with realistic noise ────────────────────────────

    def _generate_base_patient(self, index: int) -> PatientRecord:
        patient = super()._generate_base_patient(index)

        # R0 drew every baseline lab strictly inside the normal range and hard-coded
        # is_abnormal=False, which made the flag a perfect indicator of a planted
        # pattern. Allow benign out-of-range values so the flag carries no label.
        for lab in patient.lab_results:
            nr = lab.normal_range
            if nr and nr != (0, 0) and self.rng.random() < 0.18:
                span = nr[1] - nr[0]
                if self.rng.random() < 0.5:
                    lab.value = round(nr[0] - span * self.rng.uniform(0.02, 0.12), 2)
                else:
                    lab.value = round(nr[1] + span * self.rng.uniform(0.02, 0.12), 2)
                lab.is_abnormal = True

        if self.enable_notes:
            for dept in patient.departments:
                if self.rng.random() < 0.7:
                    kind = self.rng.choices(
                        ["routine", "hedged", "negation"], weights=[0.6, 0.2, 0.2]
                    )[0]
                    self._attach_note(patient, dept, self._note(kind))
        return patient

    # ── cohort constructors ──────────────────────────────────────────────

    def _make_conflict_patient(self, index: int) -> PatientRecord:
        p = self._generate_base_patient(index)
        template = self.rng.choice(self.conflict_pool)
        self._ensure_departments(p, template["departments"])

        meds = list(template["medications"])
        note_only_drug = None
        if self.enable_notes and self.rng.random() < self.noise_rates["note_only"] \
                and len(meds) > 1:
            note_only_drug = meds[-1]
            meds = meds[:-1]

        for m in meds:
            self._add_medication(p, m)

        if note_only_drug:
            self._add_medication  # not called: documented in free text only
            self._attach_note(
                p, note_only_drug["dept"],
                self._note("med_mention", drug=note_only_drug["name"]),
            )

        if template.get("findings"):
            self._add_findings(p, template["findings"])

        p.known_conflicts.append({
            "conflict_name": template["name"],
            "medications": [m["name"] for m in template["medications"]],
            "departments": template["departments"],
            "alert_level": template.get("alert_level", "high_risk"),
            "note_only_limb": note_only_drug["name"] if note_only_drug else None,
            "heldout": self.use_heldout,
        })
        # The encounter that brings the patient in is part of the clinical
        # situation, not part of the label: a warfarin/extraction conflict only
        # exists if an extraction is actually planned. Each matched negative
        # control carries the same encounter type as the positive it mirrors, so
        # the encounter carries no discriminative information.
        p.encounter_hint = {
            "department": template.get("encounter_department",
                                       template.get("trigger_department", "")),
            "procedure": template.get("encounter_procedure",
                                      template.get("trigger_procedure", "")),
        }
        p.cohort = COHORT_CONFLICT
        return p

    def _make_pattern_patient(self, index: int) -> PatientRecord:
        p = self._generate_base_patient(index)
        template = self.rng.choice(self.pattern_pool)
        self._ensure_departments(p, template["departments"])
        self._add_findings(p, template["findings"])
        p.known_patterns.append({
            "pattern_name": template["name"],
            "departments": template["departments"],
            "pattern_type": template.get("pattern_type", "diagnostic_connection"),
            "expected_diagnosis": template.get("expected_diagnosis", ""),
            "heldout": self.use_heldout,
        })
        p.cohort = COHORT_PATTERN
        return p

    def _make_distractor_patient(self, index: int) -> PatientRecord:
        p = self._generate_base_patient(index)
        template = self.rng.choice(DISTRACTOR_TEMPLATES)
        self._ensure_departments(p, template["departments"])
        for m in template["medications"]:
            self._add_medication(p, m, active=m.get("active", True))
        if template.get("findings"):
            self._add_findings(p, template["findings"])
        p.negative_controls.append({
            "control_name": template["name"],
            "mirrors": template["mirrors"],
            "medications": [m["name"] for m in template["medications"]],
        })
        p.encounter_hint = {
            "department": template["encounter_department"],
            "procedure": template["encounter_procedure"],
        }
        p.cohort = COHORT_DISTRACTOR
        return p

    def _make_ambiguous_patient(self, index: int) -> PatientRecord:
        p = self._generate_base_patient(index)
        template = self.rng.choice(AMBIGUOUS_TEMPLATES)
        self._ensure_departments(p, template["departments"])
        for m in template["medications"]:
            self._add_medication(p, m)
        if template.get("findings"):
            self._add_findings(p, template["findings"])
        p.ambiguous_cases.append({
            "case_name": template["name"],
            "adjudication": template["adjudication"],
        })
        p.encounter_hint = {
            "department": template["encounter_department"],
            "procedure": template["encounter_procedure"],
        }
        p.cohort = COHORT_AMBIGUOUS
        return p

    def _encounter_pool(self) -> list[dict]:
        """Every (department, procedure) pair used by a positive or a matched
        negative. Controls draw from the same pool, so the encounter is
        distributed identically across cohorts and carries no label."""
        if getattr(self, "_pool", None):
            return self._pool
        pool = []
        for t in self.conflict_pool:
            pool.append({
                "department": t.get("encounter_department",
                                    t.get("trigger_department", "general_practice")),
                "procedure": t.get("encounter_procedure",
                                   t.get("trigger_procedure", "clinical_review")),
            })
        for t in DISTRACTOR_TEMPLATES:
            pool.append({"department": t["encounter_department"],
                         "procedure": t["encounter_procedure"]})
        self._pool = pool
        return pool

    def _make_control_patient(self, index: int) -> PatientRecord:
        p = self._generate_base_patient(index)
        hint = self.rng.choice(self._encounter_pool())
        self._ensure_departments(p, [hint["department"]])
        p.encounter_hint = dict(hint)
        p.cohort = COHORT_CONTROL
        return p

    # ── corruption operators ─────────────────────────────────────────────

    def _corrupt(self, p: PatientRecord):
        r = self.noise_rates

        # (a) missing fields
        for m in p.medications:
            if self.rng.random() < r["missing_fields"]:
                field = self.rng.choice(["dose", "frequency", "prescribed_date"])
                setattr(m, field, "")
        for l in p.lab_results:
            if self.rng.random() < r["missing_fields"] * 0.5:
                l.normal_range = (0, 0)

        # (b) contradictory documentation across departments
        if p.medications and self.rng.random() < r["contradiction"]:
            src = self.rng.choice(p.medications)
            other_depts = [d for d in p.departments if d != src.department]
            if other_depts:
                dupe = Medication(
                    name=src.name,
                    category=src.category,
                    dose=src.dose.replace("5", "7") if src.dose else "",
                    frequency=src.frequency,
                    prescribed_by=self._random_physician(self.rng.choice(other_depts)),
                    prescribed_date=self._random_date(1, 14),
                    department=self.rng.choice(other_depts),
                    active=not src.active,
                )
                p.medications.append(dupe)
                p.record_quality_flags.append("contradiction")

        # (c) stale prescription that must not be treated as active
        if self.rng.random() < r["stale_medication"]:
            stale = self.rng.choice([
                {"name": "Ibuprofen", "category": "nsaid", "dose": "400mg"},
                {"name": "Warfarin", "category": "anticoagulant", "dose": "5mg"},
                {"name": "Prednisolone", "category": "corticosteroid", "dose": "10mg"},
            ])
            dept = self.rng.choice(p.departments)
            months = self.rng.randint(6, 30)
            if not any(m.name == stale["name"] for m in p.medications):
                self._add_medication(p, {**stale, "dept": dept,
                                         "stopped_months_ago": months,
                                         "active": False})
                p.record_quality_flags.append("stale_medication")

        # (d) resolved diagnosis
        if self.rng.random() < r["resolved_diagnosis"]:
            dept = self.rng.choice(p.departments)
            resolved = self.rng.choice([
                {"code": "N17", "name": "Acute Kidney Injury (resolved)"},
                {"code": "J45", "name": "Childhood Asthma (resolved)"},
                {"code": "K27", "name": "Peptic Ulcer (healed)"},
            ])
            if not any(c.code == resolved["code"] for c in p.conditions):
                p.conditions.append(Condition(
                    code=resolved["code"], name=resolved["name"],
                    severity="mild", diagnosed_date=self._random_date(24, 60),
                    diagnosed_by=self._random_physician(dept),
                    department=dept, active=False,
                ))
                p.record_quality_flags.append("resolved_diagnosis")

        # (e) brand/generic duplicate
        if p.medications and self.rng.random() < r["duplicate_entry"]:
            src = self.rng.choice(p.medications)
            brand = BRAND_NAMES.get(src.name)
            if brand and not any(m.name == brand for m in p.medications):
                p.medications.append(Medication(
                    name=brand, category=src.category, dose=src.dose,
                    frequency=src.frequency,
                    prescribed_by=src.prescribed_by,
                    prescribed_date=src.prescribed_date,
                    department=src.department, active=src.active,
                ))
                p.record_quality_flags.append("duplicate_entry")

    # ── negative-control hygiene ─────────────────────────────────────────

    def _record_evidence(self, p: PatientRecord, procedure: str = ""):
        """Full-record evidence view, used only for label auditing."""
        from protocol.interactions import ClinicalEvidence
        ev = ClinicalEvidence()
        ev.procedure = (procedure or "").lower()
        for m in p.medications:
            if m.active:
                ev.add_drug(m.name, m.category, m.department)
        for c in p.conditions:
            if c.active:
                ev.add_condition(c.name, c.department)
        for l in p.lab_results:
            ev.add_lab(l.test_name, l.value, l.date, l.department)
        return ev

    def _purge_incidental_findings(self, p: PatientRecord, max_rounds: int = 6):
        """Make a negative control genuinely negative.

        Baseline prescribing draws from each department's formulary, so a
        patient intended as a control can acquire a real interaction by chance
        (for example warfarin from cardiology and an NSAID from rheumatology).
        R0 counted any alert on such a patient as a false positive even when the
        alert was clinically correct, which made the reported false-positive
        rate uninterpretable. Here the record is audited against the same
        knowledge base used for scoring, and incidental findings are removed by
        withdrawing a baseline (non-template) medication.
        """
        from protocol.interactions import evaluate_rules, evaluate_patterns

        protected = set()
        for entry in p.negative_controls:
            protected.update(entry.get("medications", []))
        for entry in p.ambiguous_cases:
            protected.update(entry.get("medications", []))

        procedure = (p.encounter_hint or {}).get("procedure", "")
        for _ in range(max_rounds):
            ev = self._record_evidence(p, procedure)
            hits = evaluate_rules(ev) + evaluate_patterns(ev)
            if not hits:
                return True
            removed = False
            for hit in hits:
                for med in list(p.medications):
                    if not med.active:
                        continue
                    if med.name in protected:
                        continue
                    if any(e.lower() in med.name.lower() or
                           med.name.lower() in e.lower()
                           for e in hit["entities"]):
                        p.medications.remove(med)
                        removed = True
                        break
                if removed:
                    break
            if not removed:
                # The finding rests on protected or non-medication evidence
                # (a planted lab trend, say). Record it rather than force it
                # away, so the case can be excluded from the negative set.
                p.record_quality_flags.append("residual_finding:" + hits[0]["id"])
                return False
        return False

    # ── public API ───────────────────────────────────────────────────────

    def generate(self, num_patients: Optional[int] = None) -> list[PatientRecord]:
        n = num_patients or self.settings["num_patients"]

        builders = {
            COHORT_CONFLICT: self._make_conflict_patient,
            COHORT_PATTERN: self._make_pattern_patient,
            COHORT_DISTRACTOR: self._make_distractor_patient,
            COHORT_AMBIGUOUS: self._make_ambiguous_patient,
            COHORT_CONTROL: self._make_control_patient,
        }

        # deterministic cohort counts, then shuffled assignment order
        counts = {}
        assigned = 0
        keys = list(self.cohort_mix.keys())
        for k in keys[:-1]:
            counts[k] = int(round(n * self.cohort_mix[k]))
            assigned += counts[k]
        counts[keys[-1]] = max(0, n - assigned)

        plan = []
        for cohort, c in counts.items():
            plan.extend([cohort] * c)
        self.rng.shuffle(plan)

        patients = []
        self.excluded_negatives = 0
        for i, cohort in enumerate(plan):
            p = builders[cohort](i)
            self._corrupt(p)
            if cohort in (COHORT_DISTRACTOR, COHORT_CONTROL):
                clean = self._purge_incidental_findings(p)
                if not clean:
                    # Cannot be made a valid negative; drop it from the negative
                    # set rather than score an arguably-correct alert as an error.
                    self.excluded_negatives += 1
                    continue
            patients.append(p)

        self.rng.shuffle(patients)
        return patients

    def save_patients(self, patients, output_dir: str = "data/patients_hard"):
        os.makedirs(output_dir, exist_ok=True)
        for p in patients:
            with open(os.path.join(output_dir, f"{p.patient_id}.json"), "w") as f:
                json.dump(p.to_dict(), f, indent=2, default=str)
        index = {
            "seed": self.seed,
            "heldout": self.use_heldout,
            "total_patients": len(patients),
            "cohort_counts": {
                c: sum(1 for p in patients if getattr(p, "cohort", "") == c)
                for c in DEFAULT_COHORT_MIX
            },
            "total_conflicts": sum(len(p.known_conflicts) for p in patients),
            "total_patterns": sum(len(p.known_patterns) for p in patients),
            "total_distractors": sum(len(p.negative_controls) for p in patients),
            "total_ambiguous": sum(len(p.ambiguous_cases) for p in patients),
            "noise_rates": self.noise_rates,
        }
        with open(os.path.join(output_dir, "_index.json"), "w") as f:
            json.dump(index, f, indent=2)
        return index
