"""
MedAgentNet - Synthetic Patient Data Generator
Generates realistic patient records with intentionally planted medication
conflicts and cross-departmental diagnostic patterns for testing.
"""
import os
import json
import random
import hashlib
from datetime import datetime, timedelta
from typing import Optional

import yaml

from protocol.models import (
    PatientRecord, Medication, Condition, LabResult, Visit
)


# ── Name pools ───────────────────────────────────────────────

FIRST_NAMES_M = [
    "Ahmed", "Mohammed", "Ali", "Omar", "Khalid", "Hassan", "Ibrahim",
    "James", "Robert", "Michael", "David", "John", "William", "Richard",
    "Carlos", "Yusuf", "Wei", "Raj", "Samuel", "Daniel", "Thomas", "Andrew",
    "Faisal", "Nasser", "Tariq", "Samir", "Karim", "Bilal", "Hamad", "Rashid",
]
FIRST_NAMES_F = [
    "Fatima", "Aisha", "Maryam", "Noor", "Sara", "Layla", "Hana",
    "Mary", "Patricia", "Jennifer", "Linda", "Elizabeth", "Susan", "Karen",
    "Ana", "Mei", "Priya", "Grace", "Emma", "Sophie", "Amina", "Zainab",
    "Noura", "Dana", "Reem", "Lina", "Huda", "Salma", "Rana", "Dina",
]
LAST_NAMES = [
    "Al-Rashid", "Al-Farsi", "Khan", "Ali", "Hassan", "Johnson", "Smith",
    "Williams", "Brown", "Chen", "Kumar", "Garcia", "Martinez", "Wilson",
    "Al-Sayed", "Al-Thani", "Mahmoud", "Ibrahim", "Patel", "Kim", "Tanaka",
    "Al-Dosari", "Al-Marri", "Al-Sulaiti", "Nasser", "Jaber", "Hamad",
]

# ── Conflict Templates ───────────────────────────────────────

CONFLICT_TEMPLATES = [
    {
        "name": "warfarin_dental_extraction",
        "description": "Patient on Warfarin needs dental extraction (bleeding risk)",
        "departments": ["cardiology", "dental"],
        "medications": [
            {"name": "Warfarin", "dept": "cardiology", "category": "anticoagulant",
             "dose": "5mg", "frequency": "daily"},
        ],
        "trigger_procedure": "tooth_extraction",
        "trigger_department": "dental",
        "alert_level": "high_risk",
        "risk": "Severe bleeding risk during dental extraction for patient on anticoagulant therapy",
    },
    {
        "name": "triple_whammy_aki",
        "description": "ACE inhibitor + NSAID + Diuretic = Acute Kidney Injury risk",
        "departments": ["cardiology", "rheumatology", "general_practice"],
        "medications": [
            {"name": "Lisinopril", "dept": "cardiology", "category": "ace_inhibitor",
             "dose": "10mg", "frequency": "daily"},
            {"name": "Ibuprofen", "dept": "rheumatology", "category": "nsaid",
             "dose": "400mg", "frequency": "three times daily"},
            {"name": "Hydrochlorothiazide", "dept": "general_practice", "category": "diuretic",
             "dose": "25mg", "frequency": "daily"},
        ],
        "trigger_procedure": "emergency_evaluation",
        "trigger_department": "nephrology",
        "alert_level": "critical",
        "risk": "Triple whammy nephrotoxic combination causing acute kidney injury",
    },
    {
        "name": "metformin_renal_failure",
        "description": "Metformin with declining renal function (lactic acidosis risk)",
        "departments": ["endocrinology", "nephrology"],
        "medications": [
            {"name": "Metformin", "dept": "endocrinology", "category": "biguanide",
             "dose": "1000mg", "frequency": "twice daily"},
        ],
        "trigger_procedure": "renal_assessment",
        "trigger_department": "nephrology",
        "alert_level": "critical",
        "risk": "Metformin contraindicated with eGFR < 30, lactic acidosis risk",
    },
    {
        "name": "beta_blocker_respiratory",
        "description": "Non-selective beta-blocker with asthma/COPD (bronchospasm risk)",
        "departments": ["cardiology", "pulmonology"],
        "medications": [
            {"name": "Metoprolol", "dept": "cardiology", "category": "beta_blocker",
             "dose": "100mg", "frequency": "daily"},
        ],
        "trigger_procedure": "respiratory_assessment",
        "trigger_department": "pulmonology",
        "alert_level": "high_risk",
        "risk": "Beta-blocker may worsen bronchospasm in patients with reactive airway disease",
    },
    {
        "name": "anticoagulant_nsaid_bleeding",
        "description": "Warfarin + NSAID from different departments (GI bleeding risk)",
        "departments": ["cardiology", "rheumatology"],
        "medications": [
            {"name": "Warfarin", "dept": "cardiology", "category": "anticoagulant",
             "dose": "5mg", "frequency": "daily"},
            {"name": "Ibuprofen", "dept": "rheumatology", "category": "nsaid",
             "dose": "600mg", "frequency": "three times daily"},
        ],
        "trigger_procedure": "pain_management",
        "trigger_department": "rheumatology",
        "alert_level": "critical",
        "risk": "Combined anticoagulant and NSAID therapy - high GI bleeding risk",
    },
    {
        "name": "ophthalmic_systemic_beta_blocker",
        "description": "Timolol eye drops + systemic beta-blocker (additive bradycardia)",
        "departments": ["ophthalmology", "cardiology"],
        "medications": [
            {"name": "Timolol Eye Drops", "dept": "ophthalmology", "category": "beta_blocker_ophthalmic",
             "dose": "0.5%", "frequency": "twice daily"},
            {"name": "Metoprolol", "dept": "cardiology", "category": "beta_blocker",
             "dose": "50mg", "frequency": "twice daily"},
        ],
        "trigger_procedure": "glaucoma_management",
        "trigger_department": "ophthalmology",
        "alert_level": "high_risk",
        "risk": "Additive beta-blockade from ophthalmic and systemic routes - bradycardia risk",
    },
    {
        "name": "methotrexate_nsaid_renal",
        "description": "Methotrexate + NSAID (increased methotrexate toxicity)",
        "departments": ["rheumatology", "dental"],
        "medications": [
            {"name": "Methotrexate", "dept": "rheumatology", "category": "dmard",
             "dose": "15mg", "frequency": "weekly"},
            {"name": "Ibuprofen", "dept": "dental", "category": "nsaid",
             "dose": "400mg", "frequency": "as needed"},
        ],
        "trigger_procedure": "dental_pain_management",
        "trigger_department": "dental",
        "alert_level": "high_risk",
        "risk": "NSAIDs reduce methotrexate clearance - increased toxicity risk",
    },
    {
        "name": "carbamazepine_warfarin",
        "description": "Carbamazepine reduces Warfarin effectiveness",
        "departments": ["neurology", "cardiology"],
        "medications": [
            {"name": "Carbamazepine", "dept": "neurology", "category": "anticonvulsant",
             "dose": "200mg", "frequency": "twice daily"},
            {"name": "Warfarin", "dept": "cardiology", "category": "anticoagulant",
             "dose": "5mg", "frequency": "daily"},
        ],
        "trigger_procedure": "anticoagulation_review",
        "trigger_department": "cardiology",
        "alert_level": "high_risk",
        "risk": "Carbamazepine induces warfarin metabolism - subtherapeutic INR and stroke risk",
    },
]

PATTERN_TEMPLATES = [
    {
        "name": "undiagnosed_diabetes",
        "description": "Rising glucose + retinal changes + neuropathy = undiagnosed diabetes",
        "departments": ["general_practice", "ophthalmology", "neurology", "laboratory"],
        "findings": {
            "general_practice": {
                "conditions": [{"code": "R73", "name": "Elevated Fasting Glucose", "severity": "moderate"}],
            },
            "ophthalmology": {
                "conditions": [{"code": "H36", "name": "Diabetic Retinopathy", "severity": "moderate"}],
            },
            "neurology": {
                "conditions": [{"code": "G62", "name": "Peripheral Neuropathy", "severity": "moderate"}],
            },
            "laboratory": {
                "lab_results": [
                    {"test_code": "HBA1C", "test_name": "HbA1c", "values": [5.5, 5.8, 6.1, 6.3],
                     "normal_range": [4.0, 5.6], "months_apart": 4},
                    {"test_code": "FBG", "test_name": "Fasting Glucose", "values": [5.2, 5.6, 6.0, 6.5],
                     "normal_range": [3.9, 5.5], "months_apart": 4},
                ],
            },
        },
        "pattern_type": "diagnostic_connection",
        "expected_diagnosis": "Type 2 Diabetes Mellitus",
    },
    {
        "name": "ckd_progression",
        "description": "Declining eGFR + hypertension + proteinuria = CKD progression",
        "departments": ["nephrology", "cardiology", "laboratory"],
        "findings": {
            "nephrology": {
                "conditions": [{"code": "N18", "name": "Chronic Kidney Disease Stage 2", "severity": "moderate"}],
            },
            "cardiology": {
                "conditions": [{"code": "I10", "name": "Essential Hypertension", "severity": "chronic"}],
            },
            "laboratory": {
                "lab_results": [
                    {"test_code": "EGFR", "test_name": "eGFR", "values": [75, 68, 58, 48],
                     "normal_range": [90, 120], "months_apart": 6},
                    {"test_code": "CREAT", "test_name": "Creatinine", "values": [1.1, 1.3, 1.5, 1.8],
                     "normal_range": [0.6, 1.2], "months_apart": 6},
                ],
            },
        },
        "pattern_type": "disease_progression",
        "expected_diagnosis": "CKD Stage 3b Progression",
    },
    {
        "name": "thyroid_cardiac_connection",
        "description": "New atrial fibrillation + weight changes + thyroid symptoms = hyperthyroidism",
        "departments": ["cardiology", "endocrinology", "general_practice", "laboratory"],
        "findings": {
            "cardiology": {
                "conditions": [{"code": "I48", "name": "New-onset Atrial Fibrillation", "severity": "serious"}],
            },
            "general_practice": {
                "conditions": [{"code": "R63", "name": "Unexplained Weight Loss", "severity": "moderate"}],
            },
            "endocrinology": {
                "conditions": [{"code": "E05", "name": "Hyperthyroidism", "severity": "moderate"}],
            },
            "laboratory": {
                "lab_results": [
                    {"test_code": "TSH", "test_name": "TSH", "values": [0.3, 0.15, 0.05],
                     "normal_range": [0.4, 4.0], "months_apart": 3},
                ],
            },
        },
        "pattern_type": "diagnostic_connection",
        "expected_diagnosis": "Hyperthyroid-induced Atrial Fibrillation",
    },
]


class PatientDataGenerator:
    """Generates synthetic patient records with planted conflicts and patterns."""

    def __init__(self, config_dir: str = "config"):
        self.config_dir = config_dir
        dept_path = os.path.join(config_dir, "departments.yaml")
        settings_path = os.path.join(config_dir, "settings.yaml")

        with open(dept_path) as f:
            self.dept_config = yaml.safe_load(f)["departments"]
        with open(settings_path) as f:
            self.settings = yaml.safe_load(f)["simulation"]

        self.departments = list(self.dept_config.keys())
        self.rng = random.Random(self.settings["random_seed"])

    def _make_id(self, name: str) -> str:
        return f"PAT-{hashlib.sha256(name.encode()).hexdigest()[:8]}"

    def _random_date(self, months_ago_min: int = 1, months_ago_max: int = 24) -> str:
        days = self.rng.randint(months_ago_min * 30, months_ago_max * 30)
        dt = datetime.now() - timedelta(days=days)
        return dt.strftime("%Y-%m-%d")

    def _random_physician(self, dept: str) -> str:
        prefix = "Dr."
        dept_name = self.dept_config[dept]["name"]
        name = self.rng.choice(LAST_NAMES)
        return f"{prefix} {name} ({dept_name})"

    def _generate_base_patient(self, index: int) -> PatientRecord:
        """Generate a patient with random departments and baseline data."""
        gender = self.rng.choice(["M", "F"])
        if gender == "M":
            first = self.rng.choice(FIRST_NAMES_M)
        else:
            first = self.rng.choice(FIRST_NAMES_F)
        last = self.rng.choice(LAST_NAMES)
        name = f"{first} {last}"
        age = self.rng.randint(25, 85)

        # Assign departments
        n_depts = self.rng.randint(
            self.settings["min_departments_per_patient"],
            self.settings["max_departments_per_patient"]
        )
        # Always include GP and Lab
        depts = {"general_practice", "laboratory"}
        available = [d for d in self.departments if d not in depts]
        extra = self.rng.sample(available, min(n_depts - 2, len(available)))
        depts.update(extra)

        patient = PatientRecord(
            patient_id=self._make_id(f"{name}-{index}"),
            name=name,
            age=age,
            gender=gender,
            departments=sorted(list(depts)),
            allergies=self.rng.sample(
                ["Penicillin", "Sulfa", "Latex", "Shellfish", "Peanuts", "None"],
                k=self.rng.randint(0, 2)
            ),
        )

        # Add baseline conditions and medications for each department
        for dept in patient.departments:
            dept_data = self.dept_config.get(dept, {})

            # Add conditions
            conditions = dept_data.get("common_conditions", [])
            if conditions:
                n_cond = self.rng.randint(0, min(2, len(conditions)))
                for cond in self.rng.sample(conditions, n_cond):
                    patient.conditions.append(Condition(
                        code=cond["code"],
                        name=cond["name"],
                        severity=cond["severity"],
                        diagnosed_date=self._random_date(3, 36),
                        diagnosed_by=self._random_physician(dept),
                        department=dept,
                    ))

            # Add medications
            medications = dept_data.get("common_medications", [])
            if medications:
                n_med = self.rng.randint(0, min(2, len(medications)))
                for med_template in self.rng.sample(medications, n_med):
                    doses = {"anticoagulant": "5mg", "beta_blocker": "50mg",
                             "ace_inhibitor": "10mg", "nsaid": "400mg",
                             "biguanide": "500mg", "diuretic": "25mg"}
                    patient.medications.append(Medication(
                        name=med_template["name"],
                        category=med_template["category"],
                        dose=doses.get(med_template["category"], "10mg"),
                        frequency="daily",
                        prescribed_by=self._random_physician(dept),
                        prescribed_date=self._random_date(1, 12),
                        department=dept,
                        interactions=med_template.get("interactions", []),
                        risk_flags=med_template.get("risk_flags", []),
                    ))

            # Add visits
            n_visits = self.rng.randint(1, 4)
            for _ in range(n_visits):
                patient.visits.append(Visit(
                    department=dept,
                    date=self._random_date(1, 18),
                    reason=f"Routine {dept_data.get('name', dept)} visit",
                    physician=self._random_physician(dept),
                ))

        # Add baseline lab results
        if "laboratory" in patient.departments:
            lab_tests = self.dept_config.get("laboratory", {}).get("lab_tests", [])
            for test in lab_tests:
                nr = test.get("normal_range")
                if nr:
                    value = round(self.rng.uniform(nr[0], nr[1]), 2)
                    patient.lab_results.append(LabResult(
                        test_code=test["code"],
                        test_name=test["name"],
                        value=value,
                        normal_range=tuple(nr),
                        date=self._random_date(1, 6),
                        is_abnormal=False,
                    ))

        return patient

    def _plant_conflict(self, patient: PatientRecord, template: dict) -> bool:
        """Plant a known medication conflict into a patient's records."""
        required_depts = set(template["departments"])
        if not required_depts.issubset(set(patient.departments)):
            # Add missing departments
            for dept in required_depts:
                if dept not in patient.departments:
                    patient.departments.append(dept)
                    patient.departments.sort()

        # Add the conflict medications
        for med_info in template["medications"]:
            # Check if medication already exists
            existing = [m for m in patient.medications if m.name == med_info["name"]]
            if not existing:
                patient.medications.append(Medication(
                    name=med_info["name"],
                    category=med_info["category"],
                    dose=med_info.get("dose", "10mg"),
                    frequency=med_info.get("frequency", "daily"),
                    prescribed_by=self._random_physician(med_info["dept"]),
                    prescribed_date=self._random_date(1, 12),
                    department=med_info["dept"],
                    interactions=[],
                    risk_flags=[],
                ))

        patient.known_conflicts.append({
            "conflict_name": template["name"],
            "description": template["description"],
            "departments": template["departments"],
            "medications": [m["name"] for m in template["medications"]],
            "alert_level": template["alert_level"],
            "risk": template["risk"],
            "trigger_department": template["trigger_department"],
            "trigger_procedure": template["trigger_procedure"],
        })
        return True

    def _plant_pattern(self, patient: PatientRecord, template: dict) -> bool:
        """Plant a cross-departmental diagnostic pattern into a patient."""
        required_depts = set(template["departments"])
        for dept in required_depts:
            if dept not in patient.departments:
                patient.departments.append(dept)
        patient.departments.sort()

        for dept, findings in template["findings"].items():
            # Add conditions
            for cond in findings.get("conditions", []):
                existing = [c for c in patient.conditions if c.code == cond["code"]]
                if not existing:
                    patient.conditions.append(Condition(
                        code=cond["code"],
                        name=cond["name"],
                        severity=cond["severity"],
                        diagnosed_date=self._random_date(3, 18),
                        diagnosed_by=self._random_physician(dept),
                        department=dept,
                    ))

            # Add longitudinal lab results
            for lab in findings.get("lab_results", []):
                base_date = datetime.now() - timedelta(
                    days=len(lab["values"]) * lab["months_apart"] * 30
                )
                for i, val in enumerate(lab["values"]):
                    result_date = base_date + timedelta(days=i * lab["months_apart"] * 30)
                    nr = tuple(lab.get("normal_range", (0, 0)))
                    patient.lab_results.append(LabResult(
                        test_code=lab["test_code"],
                        test_name=lab["test_name"],
                        value=val,
                        normal_range=nr,
                        date=result_date.strftime("%Y-%m-%d"),
                        is_abnormal=val < nr[0] or val > nr[1] if nr != (0, 0) else False,
                    ))

        patient.known_patterns.append({
            "pattern_name": template["name"],
            "description": template["description"],
            "departments": template["departments"],
            "pattern_type": template["pattern_type"],
            "expected_diagnosis": template["expected_diagnosis"],
        })
        return True

    def generate(self, num_patients: Optional[int] = None) -> list[PatientRecord]:
        """Generate the full patient dataset."""
        n = num_patients or self.settings["num_patients"]
        conflict_rate = self.settings["conflict_rate"]
        pattern_rate = self.settings["pattern_rate"]

        patients = []
        n_conflicts = int(n * conflict_rate)
        n_patterns = int(n * pattern_rate)

        for i in range(n):
            patient = self._generate_base_patient(i)

            # Plant conflicts in the first batch
            if i < n_conflicts:
                template = CONFLICT_TEMPLATES[i % len(CONFLICT_TEMPLATES)]
                self._plant_conflict(patient, template)

            # Plant patterns in the next batch (some overlap is fine)
            if n_conflicts <= i < n_conflicts + n_patterns:
                template = PATTERN_TEMPLATES[i % len(PATTERN_TEMPLATES)]
                self._plant_pattern(patient, template)

            # Some patients get both
            if i < min(5, n_conflicts):
                pt_template = PATTERN_TEMPLATES[i % len(PATTERN_TEMPLATES)]
                self._plant_pattern(patient, pt_template)

            patients.append(patient)

        self.rng.shuffle(patients)
        return patients

    def save_patients(self, patients: list[PatientRecord], output_dir: str = "data/patients"):
        """Save patient records to JSON files."""
        os.makedirs(output_dir, exist_ok=True)

        # Save individual records
        for p in patients:
            path = os.path.join(output_dir, f"{p.patient_id}.json")
            with open(path, "w") as f:
                json.dump(p.to_dict(), f, indent=2, default=str)

        # Save summary index
        index = {
            "total_patients": len(patients),
            "patients_with_conflicts": sum(1 for p in patients if p.known_conflicts),
            "patients_with_patterns": sum(1 for p in patients if p.known_patterns),
            "total_conflicts": sum(len(p.known_conflicts) for p in patients),
            "total_patterns": sum(len(p.known_patterns) for p in patients),
            "departments_used": sorted(list(set(
                d for p in patients for d in p.departments
            ))),
            "patient_ids": [p.patient_id for p in patients],
        }
        with open(os.path.join(output_dir, "_index.json"), "w") as f:
            json.dump(index, f, indent=2)

        return index
