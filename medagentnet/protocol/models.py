"""
MedAgentNet - Core Data Models
All data structures used throughout the system.
"""
from __future__ import annotations
import uuid
import json
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from enum import Enum, IntEnum
from typing import Optional


class DisclosureTier(IntEnum):
    """Privacy disclosure levels."""
    FLAG_ONLY = 1       # "A relevant concern exists"
    CLINICAL_SUMMARY = 2  # Medication name, dose, date
    FULL_CONTEXT = 3    # Complete history and notes


class QueryPriority(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class Severity(Enum):
    MILD = "mild"
    MODERATE = "moderate"
    CHRONIC = "chronic"
    SERIOUS = "serious"
    CRITICAL = "critical"


class AlertLevel(Enum):
    INFO = "info"
    WARNING = "warning"
    HIGH_RISK = "high_risk"
    CRITICAL = "critical"


# ── Patient Data Models ──────────────────────────────────────


@dataclass
class Medication:
    name: str
    category: str
    dose: str = ""
    frequency: str = ""
    prescribed_by: str = ""
    prescribed_date: str = ""
    department: str = ""
    interactions: list[str] = field(default_factory=list)
    risk_flags: list[str] = field(default_factory=list)
    active: bool = True

    def to_dict(self):
        return asdict(self)


@dataclass
class Condition:
    code: str
    name: str
    severity: str
    diagnosed_date: str = ""
    diagnosed_by: str = ""
    department: str = ""
    active: bool = True
    notes: str = ""

    def to_dict(self):
        return asdict(self)


@dataclass
class LabResult:
    test_code: str
    test_name: str
    value: float
    unit: str = ""
    normal_range: tuple = (0, 0)
    date: str = ""
    is_abnormal: bool = False
    department: str = "laboratory"

    def to_dict(self):
        d = asdict(self)
        d["normal_range"] = list(self.normal_range)
        return d


@dataclass
class Visit:
    department: str
    date: str
    reason: str
    physician: str = ""
    notes: str = ""
    procedures: list[str] = field(default_factory=list)
    medications_prescribed: list[str] = field(default_factory=list)

    def to_dict(self):
        return asdict(self)


@dataclass
class PatientRecord:
    patient_id: str
    name: str
    age: int
    gender: str
    departments: list[str] = field(default_factory=list)
    medications: list[Medication] = field(default_factory=list)
    conditions: list[Condition] = field(default_factory=list)
    lab_results: list[LabResult] = field(default_factory=list)
    visits: list[Visit] = field(default_factory=list)
    allergies: list[str] = field(default_factory=list)
    # Ground truth for evaluation — NEVER transmitted to an agent.
    known_conflicts: list[dict] = field(default_factory=list)
    known_patterns: list[dict] = field(default_factory=list)
    # R1 additions: cohort membership and hard-case bookkeeping
    cohort: str = ""
    negative_controls: list[dict] = field(default_factory=list)
    ambiguous_cases: list[dict] = field(default_factory=list)
    encounter_hint: dict = field(default_factory=dict)
    record_quality_flags: list[str] = field(default_factory=list)

    def to_dict(self):
        return {
            "patient_id": self.patient_id,
            "name": self.name,
            "age": self.age,
            "gender": self.gender,
            "departments": self.departments,
            "medications": [m.to_dict() for m in self.medications],
            "conditions": [c.to_dict() for c in self.conditions],
            "lab_results": [l.to_dict() for l in self.lab_results],
            "visits": [v.to_dict() for v in self.visits],
            "allergies": self.allergies,
            "cohort": self.cohort,
            "record_quality_flags": self.record_quality_flags,
            "known_conflicts": self.known_conflicts,
            "known_patterns": self.known_patterns,
            "negative_controls": self.negative_controls,
            "ambiguous_cases": self.ambiguous_cases,
        }

    def get_department_records(self, department: str) -> dict:
        """Return only records belonging to a specific department.

        This is the ONLY channel through which patient data reaches an agent.
        Ground-truth fields (known_conflicts, known_patterns,
        negative_controls, ambiguous_cases, cohort) are deliberately absent.
        """
        return {
            "patient_id": self.patient_id,
            "age": self.age,
            "gender": self.gender,
            "medications": [m.to_dict() for m in self.medications if m.department == department],
            "conditions": [c.to_dict() for c in self.conditions if c.department == department],
            "lab_results": [l.to_dict() for l in self.lab_results if l.department == department],
            "visits": [v.to_dict() for v in self.visits if v.department == department],
            "allergies": self.allergies,
        }

    def phi_inventory(self, department: str = None) -> dict:
        """Enumerate the distinct PHI items held for this patient.

        Used by the privacy-leakage metric as the denominator: the set of
        identifiable clinical facts an agent could in principle disclose.
        """
        def _sel(items):
            return [i for i in items
                    if department is None or getattr(i, "department", "") == department]
        return {
            "medications": sorted({m.name for m in _sel(self.medications)}),
            "conditions": sorted({c.name for c in _sel(self.conditions)}),
            "lab_tests": sorted({l.test_name for l in _sel(self.lab_results)}),
            "dose_values": sorted({m.dose for m in _sel(self.medications) if m.dose}),
            "dates": sorted({m.prescribed_date for m in _sel(self.medications)
                             if m.prescribed_date}),
            "note_text": [v.notes for v in _sel(self.visits) if v.notes],
        }

    def to_json(self):
        return json.dumps(self.to_dict(), indent=2, default=str)


# ── Protocol Models ──────────────────────────────────────────


@dataclass
class ClinicalQuery:
    query_id: str = field(default_factory=lambda: f"QRY-{uuid.uuid4().hex[:8]}")
    source_agent: str = ""
    target_agent: str = ""
    patient_id: str = ""
    query_type: str = ""
    clinical_context: dict = field(default_factory=dict)
    disclosure_tier: int = 2
    consent_token: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    priority: str = "medium"

    def to_dict(self):
        return asdict(self)

    def to_json(self):
        return json.dumps(self.to_dict(), indent=2)


@dataclass
class AgentResponse:
    response_id: str = field(default_factory=lambda: f"RSP-{uuid.uuid4().hex[:8]}")
    query_id: str = ""
    source_agent: str = ""
    patient_id: str = ""
    disclosure_tier: int = 2
    findings: list[dict] = field(default_factory=list)
    medications_reported: list[dict] = field(default_factory=list)
    conditions_reported: list[dict] = field(default_factory=list)
    lab_results_reported: list[dict] = field(default_factory=list)
    risk_flags: list[str] = field(default_factory=list)
    summary: str = ""
    raw_llm_response: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self):
        return asdict(self)


@dataclass
class ConflictAlert:
    alert_id: str = field(default_factory=lambda: f"ALT-{uuid.uuid4().hex[:8]}")
    patient_id: str = ""
    alert_level: str = "warning"
    alert_type: str = ""  # "medication_conflict", "pattern_detected", "risk_flag"
    description: str = ""
    involved_departments: list[str] = field(default_factory=list)
    involved_medications: list[str] = field(default_factory=list)
    involved_conditions: list[str] = field(default_factory=list)
    recommendation: str = ""
    source_responses: list[str] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self):
        return asdict(self)


@dataclass
class AuditEntry:
    entry_id: str = field(default_factory=lambda: f"AUD-{uuid.uuid4().hex[:8]}")
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    event_type: str = ""  # "query_sent", "response_received", "consent_check", "alert_raised"
    source_agent: str = ""
    target_agent: str = ""
    patient_id: str = ""
    query_type: str = ""
    disclosure_tier: int = 2
    consent_granted: bool = True
    data_fields_shared: list[str] = field(default_factory=list)
    outcome: str = ""

    def to_dict(self):
        return asdict(self)
