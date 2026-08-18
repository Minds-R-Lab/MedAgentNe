"""
MedAgentNet - Hard-case case library (revision R1)
==================================================

This module supplies the evaluation material that the R0 benchmark lacked:

1. ``DISTRACTOR_TEMPLATES`` - drug-matched *negative* controls. Each distractor
   reuses the medications of a positive conflict template but arranges them in a
   combination that is NOT clinically actionable. A detector that keys on drug
   names alone will fire on these; a detector that reasons about the combination
   will not. These give the false-positive rate a meaningful denominator.

2. ``AMBIGUOUS_TEMPLATES`` - borderline cases with adjudicated "acceptable
   either way" ground truth, scored separately from the strict positives.

3. ``HELDOUT_CONFLICT_TEMPLATES`` / ``HELDOUT_PATTERN_TEMPLATES`` - a second
   scenario family that was written after the prompts, the routing rules and the
   rule-based provider were frozen. Nothing in ``llm/prompts.py`` or the
   Mock provider's keyword table refers to these drugs or diseases, so they
   measure generalisation rather than recall of the development set.

4. ``CLINICAL_NOTE_TEMPLATES`` - naturalistic free-text notes, including
   abbreviations, negations and hedging, so that part of the clinical signal
   sits outside the structured fields.

5. Corruption operators (``NOISE_OPERATORS``) that produce incomplete records,
   contradictory cross-departmental documentation, stale/discontinued
   medications and resolved diagnoses.

Design rule followed throughout: every template records *what is in the data*.
No template stores the answer in a field that is ever transmitted to an agent.
The evaluation-facing label lives in ``known_conflicts`` / ``known_patterns``
and is consumed only by ``simulation/evaluation.py``.
"""

# ─────────────────────────────────────────────────────────────────────────────
#  1. Repairs to the R0 positive templates
# ─────────────────────────────────────────────────────────────────────────────
# Two R0 templates asserted a conflict whose second limb was never instantiated
# in the record: `metformin_renal_failure` planted Metformin but no renal
# impairment, and `beta_blocker_respiratory` planted Metoprolol (a
# cardioselective agent) but no airway disease. Both are repaired here.

TEMPLATE_REPAIRS = {
    "metformin_renal_failure": {
        # eGFR must actually be low for the contraindication to exist.
        "extra_findings": {
            "nephrology": {
                "conditions": [
                    {"code": "N18.4", "name": "Chronic Kidney Disease Stage 4",
                     "severity": "serious"},
                ],
            },
            "laboratory": {
                "lab_results": [
                    {"test_code": "EGFR", "test_name": "eGFR",
                     "values": [42, 35, 28], "unit": "mL/min/1.73m2",
                     "normal_range": [90, 120], "months_apart": 4},
                ],
            },
        },
    },
    "beta_blocker_respiratory": {
        # Use a genuinely non-selective agent and plant the airway disease.
        "medication_overrides": [
            {"name": "Propranolol", "dept": "cardiology", "category": "beta_blocker",
             "dose": "40mg", "frequency": "twice daily"},
        ],
        "extra_findings": {
            "pulmonology": {
                "conditions": [
                    {"code": "J45", "name": "Persistent Asthma", "severity": "moderate"},
                ],
            },
        },
        "description_override":
            "Non-selective beta-blocker prescribed to a patient with documented asthma",
    },
}


# ─────────────────────────────────────────────────────────────────────────────
#  2. Drug-matched negative controls (hard negatives)
# ─────────────────────────────────────────────────────────────────────────────
# `mirrors` names the positive template each distractor is matched against, so
# the results table can report performance on matched pairs.

DISTRACTOR_TEMPLATES = [
    {
        "name": "warfarin_paracetamol_safe",
        "mirrors": "anticoagulant_nsaid_bleeding",
        "rationale": "Warfarin with paracetamol is the recommended analgesic "
                     "choice; no NSAID is present, so there is no added GI "
                     "bleeding risk beyond baseline anticoagulation.",
        "departments": ["cardiology", "rheumatology"],
        "medications": [
            {"name": "Warfarin", "dept": "cardiology", "category": "anticoagulant",
             "dose": "3mg", "frequency": "daily"},
            {"name": "Paracetamol", "dept": "rheumatology", "category": "analgesic",
             "dose": "500mg", "frequency": "as needed"},
        ],
        "encounter_department": "rheumatology",
        "encounter_procedure": "pain_management",
    },
    {
        "name": "ace_diuretic_no_nsaid",
        "mirrors": "triple_whammy_aki",
        "rationale": "ACE inhibitor plus thiazide is a standard, guideline-"
                     "recommended antihypertensive pairing. The third limb of "
                     "the triple whammy (an NSAID) is absent.",
        "departments": ["cardiology", "general_practice"],
        "medications": [
            {"name": "Lisinopril", "dept": "cardiology", "category": "ace_inhibitor",
             "dose": "10mg", "frequency": "daily"},
            {"name": "Hydrochlorothiazide", "dept": "general_practice",
             "category": "diuretic", "dose": "25mg", "frequency": "daily"},
        ],
        "encounter_department": "general_practice",
        "encounter_procedure": "hypertension_review",
    },
    {
        "name": "metformin_normal_renal",
        "mirrors": "metformin_renal_failure",
        "rationale": "Metformin at standard dose with a normal, stable eGFR. "
                     "No contraindication exists.",
        "departments": ["endocrinology", "laboratory"],
        "medications": [
            {"name": "Metformin", "dept": "endocrinology", "category": "biguanide",
             "dose": "1000mg", "frequency": "twice daily"},
        ],
        "findings": {
            "laboratory": {
                "lab_results": [
                    {"test_code": "EGFR", "test_name": "eGFR",
                     "values": [98, 96, 99], "unit": "mL/min/1.73m2",
                     "normal_range": [90, 120], "months_apart": 6},
                ],
            },
        },
        "encounter_department": "endocrinology",
        "encounter_procedure": "diabetes_review",
    },
    {
        "name": "cardioselective_bb_no_airway_disease",
        "mirrors": "beta_blocker_respiratory",
        "rationale": "Metoprolol is cardioselective and the patient has no "
                     "documented reactive airway disease.",
        "departments": ["cardiology"],
        "medications": [
            {"name": "Metoprolol", "dept": "cardiology", "category": "beta_blocker",
             "dose": "50mg", "frequency": "daily"},
        ],
        "encounter_department": "cardiology",
        "encounter_procedure": "hypertension_review",
    },
    {
        "name": "timolol_alone",
        "mirrors": "ophthalmic_systemic_beta_blocker",
        "rationale": "Topical timolol without any systemic beta-blocker. "
                     "Systemic absorption alone does not warrant an alert.",
        "departments": ["ophthalmology"],
        "medications": [
            {"name": "Timolol Eye Drops", "dept": "ophthalmology",
             "category": "beta_blocker_ophthalmic", "dose": "0.5%",
             "frequency": "twice daily"},
        ],
        "encounter_department": "ophthalmology",
        "encounter_procedure": "glaucoma_management",
    },
    {
        "name": "methotrexate_with_folate",
        "mirrors": "methotrexate_nsaid_renal",
        "rationale": "Weekly methotrexate with folic acid cover and no NSAID. "
                     "This is the standard regimen.",
        "departments": ["rheumatology"],
        "medications": [
            {"name": "Methotrexate", "dept": "rheumatology", "category": "dmard",
             "dose": "15mg", "frequency": "weekly"},
            {"name": "Folic Acid", "dept": "rheumatology", "category": "supplement",
             "dose": "5mg", "frequency": "weekly"},
        ],
        "encounter_department": "rheumatology",
        "encounter_procedure": "dmard_monitoring",
    },
    {
        "name": "carbamazepine_alone",
        "mirrors": "carbamazepine_warfarin",
        "rationale": "Carbamazepine monotherapy for epilepsy with no "
                     "concurrent anticoagulant to interact with.",
        "departments": ["neurology"],
        "medications": [
            {"name": "Carbamazepine", "dept": "neurology", "category": "anticonvulsant",
             "dose": "200mg", "frequency": "twice daily"},
        ],
        "encounter_department": "neurology",
        "encounter_procedure": "seizure_review",
    },
    {
        "name": "warfarin_dental_checkup_no_procedure",
        "mirrors": "warfarin_dental_extraction",
        "rationale": "Anticoagulated patient attending for a routine scale and "
                     "polish. No invasive procedure is planned, so no bleeding "
                     "risk assessment is triggered.",
        "departments": ["cardiology", "dental"],
        "medications": [
            {"name": "Warfarin", "dept": "cardiology", "category": "anticoagulant",
             "dose": "5mg", "frequency": "daily"},
        ],
        "encounter_department": "dental",
        "encounter_procedure": "dental_hygiene_visit",
    },
    {
        "name": "discontinued_nsaid_on_warfarin",
        "mirrors": "anticoagulant_nsaid_bleeding",
        "rationale": "The NSAID was stopped four months ago and is recorded as "
                     "inactive. Alerting on an inactive prescription is a "
                     "false positive.",
        "departments": ["cardiology", "rheumatology"],
        "medications": [
            {"name": "Warfarin", "dept": "cardiology", "category": "anticoagulant",
             "dose": "5mg", "frequency": "daily"},
            {"name": "Ibuprofen", "dept": "rheumatology", "category": "nsaid",
             "dose": "400mg", "frequency": "three times daily", "active": False,
             "stopped_months_ago": 4},
        ],
        "encounter_department": "rheumatology",
        "encounter_procedure": "pain_management",
    },
    {
        "name": "resolved_ckd_normal_function",
        "mirrors": "ckd_progression",
        "rationale": "A historical AKI episode that fully resolved. eGFR "
                     "recovered and has been stable for a year.",
        "departments": ["nephrology", "laboratory"],
        "medications": [],
        "findings": {
            "nephrology": {
                "conditions": [
                    {"code": "N17", "name": "Acute Kidney Injury (resolved)",
                     "severity": "mild", "active": False},
                ],
            },
            "laboratory": {
                "lab_results": [
                    {"test_code": "EGFR", "test_name": "eGFR",
                     "values": [62, 84, 95, 97], "unit": "mL/min/1.73m2",
                     "normal_range": [90, 120], "months_apart": 4},
                ],
            },
        },
        "encounter_department": "nephrology",
        "encounter_procedure": "renal_assessment",
    },
]


# ─────────────────────────────────────────────────────────────────────────────
#  3. Ambiguous cases (adjudicated as "either answer defensible")
# ─────────────────────────────────────────────────────────────────────────────

AMBIGUOUS_TEMPLATES = [
    {
        "name": "warfarin_low_dose_aspirin_indicated",
        "adjudication": "either",
        "rationale": "Warfarin plus low-dose aspirin raises bleeding risk but "
                     "is deliberately co-prescribed after mechanical valve "
                     "replacement. Flagging is defensible; suppressing on the "
                     "basis of the valve indication is also defensible.",
        "departments": ["cardiology", "general_practice"],
        "medications": [
            {"name": "Warfarin", "dept": "cardiology", "category": "anticoagulant",
             "dose": "6mg", "frequency": "daily"},
            {"name": "Aspirin", "dept": "general_practice", "category": "antiplatelet",
             "dose": "75mg", "frequency": "daily"},
        ],
        "findings": {
            "cardiology": {
                "conditions": [
                    {"code": "Z95.2", "name": "Mechanical Mitral Valve Prosthesis",
                     "severity": "chronic"},
                ],
            },
        },
        "encounter_department": "general_practice",
        "encounter_procedure": "medication_review",
    },
    {
        "name": "borderline_hba1c_prediabetes",
        "adjudication": "either",
        "rationale": "HbA1c 6.0-6.2% with no end-organ findings meets "
                     "pre-diabetes criteria but not diabetes. Raising a "
                     "pattern alert is defensible; so is watchful waiting.",
        "departments": ["general_practice", "laboratory"],
        "medications": [],
        "findings": {
            "laboratory": {
                "lab_results": [
                    {"test_code": "HBA1C", "test_name": "HbA1c",
                     "values": [5.9, 6.0, 6.1], "unit": "%",
                     "normal_range": [4.0, 5.6], "months_apart": 6},
                ],
            },
        },
        "encounter_department": "general_practice",
        "encounter_procedure": "annual_health_check",
    },
    {
        "name": "isolated_egfr_dip",
        "adjudication": "either",
        "rationale": "A single eGFR reading below baseline that recovers on "
                     "the next draw. Could be a transient pre-renal dip or the "
                     "start of a decline.",
        "departments": ["nephrology", "laboratory"],
        "medications": [],
        "findings": {
            "laboratory": {
                "lab_results": [
                    {"test_code": "EGFR", "test_name": "eGFR",
                     "values": [88, 71, 86], "unit": "mL/min/1.73m2",
                     "normal_range": [90, 120], "months_apart": 5},
                ],
            },
        },
        "encounter_department": "nephrology",
        "encounter_procedure": "renal_assessment",
    },
]


# ─────────────────────────────────────────────────────────────────────────────
#  4. Held-out scenario family (written after prompts/rules were frozen)
# ─────────────────────────────────────────────────────────────────────────────

HELDOUT_CONFLICT_TEMPLATES = [
    {
        "name": "ssri_nsaid_gi_bleed",
        "description": "SSRI with chronic NSAID use raises upper GI bleeding risk",
        "departments": ["neurology", "rheumatology"],
        "medications": [
            {"name": "Sertraline", "dept": "neurology", "category": "ssri",
             "dose": "50mg", "frequency": "daily"},
            {"name": "Naproxen", "dept": "rheumatology", "category": "nsaid",
             "dose": "500mg", "frequency": "twice daily"},
        ],
        "encounter_department": "rheumatology",
        "encounter_procedure": "pain_management",
        "alert_level": "high_risk",
        "mechanism_terms": ["gastrointestinal", "gi", "bleed", "haemorrhage",
                            "hemorrhage", "ulcer", "platelet"],
        "risk": "Serotonergic platelet inhibition plus NSAID mucosal injury",
    },
    {
        "name": "statin_macrolide_rhabdo",
        "description": "Simvastatin with clarithromycin raises rhabdomyolysis risk",
        "departments": ["cardiology", "general_practice"],
        "medications": [
            {"name": "Simvastatin", "dept": "cardiology", "category": "statin",
             "dose": "40mg", "frequency": "daily"},
            {"name": "Clarithromycin", "dept": "general_practice",
             "category": "macrolide_antibiotic", "dose": "500mg",
             "frequency": "twice daily"},
        ],
        "encounter_department": "general_practice",
        "encounter_procedure": "acute_infection_review",
        "alert_level": "critical",
        "mechanism_terms": ["rhabdomyolysis", "myopathy", "muscle", "creatine kinase",
                            "ck", "cyp3a4", "statin"],
        "risk": "CYP3A4 inhibition raises simvastatin exposure",
    },
    {
        "name": "potassium_sparing_ace_hyperkalaemia",
        "description": "Spironolactone with an ACE inhibitor risks hyperkalaemia",
        "departments": ["cardiology", "nephrology"],
        "medications": [
            {"name": "Spironolactone", "dept": "cardiology",
             "category": "potassium_sparing_diuretic", "dose": "25mg",
             "frequency": "daily"},
            {"name": "Ramipril", "dept": "nephrology", "category": "ace_inhibitor",
             "dose": "5mg", "frequency": "daily"},
        ],
        "findings": {
            "laboratory": {
                "lab_results": [
                    {"test_code": "K", "test_name": "Serum Potassium",
                     "values": [4.9, 5.3, 5.7], "unit": "mmol/L",
                     "normal_range": [3.5, 5.1], "months_apart": 3},
                ],
            },
        },
        "encounter_department": "nephrology",
        "encounter_procedure": "renal_assessment",
        "alert_level": "critical",
        "mechanism_terms": ["hyperkalaemia", "hyperkalemia", "potassium", "k+",
                            "arrhythmia", "cardiac arrest"],
        "risk": "Additive potassium retention",
    },
    {
        "name": "levothyroxine_calcium_absorption",
        "description": "Calcium carbonate taken with levothyroxine impairs absorption",
        "departments": ["endocrinology", "general_practice"],
        "medications": [
            {"name": "Levothyroxine", "dept": "endocrinology",
             "category": "thyroid_hormone", "dose": "100mcg", "frequency": "daily"},
            {"name": "Calcium Carbonate", "dept": "general_practice",
             "category": "mineral_supplement", "dose": "1500mg", "frequency": "daily"},
        ],
        "findings": {
            "laboratory": {
                "lab_results": [
                    {"test_code": "TSH", "test_name": "TSH",
                     "values": [2.1, 4.8, 7.6], "unit": "mIU/L",
                     "normal_range": [0.4, 4.0], "months_apart": 4},
                ],
            },
        },
        "encounter_department": "endocrinology",
        "encounter_procedure": "thyroid_review",
        "alert_level": "high_risk",
        "mechanism_terms": ["absorption", "chelat", "tsh", "hypothyroid",
                            "separate", "binding"],
        "risk": "Chelation in the gut lowers levothyroxine bioavailability",
    },
    {
        "name": "allopurinol_azathioprine_marrow",
        "description": "Allopurinol with azathioprine risks bone-marrow suppression",
        "departments": ["rheumatology", "nephrology"],
        "medications": [
            {"name": "Allopurinol", "dept": "nephrology",
             "category": "xanthine_oxidase_inhibitor", "dose": "300mg",
             "frequency": "daily"},
            {"name": "Azathioprine", "dept": "rheumatology",
             "category": "immunosuppressant", "dose": "100mg", "frequency": "daily"},
        ],
        "findings": {
            "laboratory": {
                "lab_results": [
                    {"test_code": "WBC", "test_name": "White Cell Count",
                     "values": [6.1, 4.2, 2.4], "unit": "x10^9/L",
                     "normal_range": [4.0, 11.0], "months_apart": 2},
                ],
            },
        },
        "encounter_department": "rheumatology",
        "encounter_procedure": "immunosuppression_monitoring",
        "alert_level": "critical",
        "mechanism_terms": ["marrow", "myelosuppression", "neutropenia",
                            "white cell", "wbc", "xanthine oxidase", "pancytopenia"],
        "risk": "Blocked azathioprine catabolism causes accumulation",
    },
    {
        "name": "digoxin_amiodarone_toxicity",
        "description": "Amiodarone raises digoxin levels toward toxicity",
        "departments": ["cardiology", "general_practice"],
        "medications": [
            {"name": "Digoxin", "dept": "cardiology", "category": "cardiac_glycoside",
             "dose": "250mcg", "frequency": "daily"},
            {"name": "Amiodarone", "dept": "cardiology", "category": "antiarrhythmic",
             "dose": "200mg", "frequency": "daily"},
        ],
        "encounter_department": "general_practice",
        "encounter_procedure": "medication_review",
        "alert_level": "critical",
        "mechanism_terms": ["digoxin toxicity", "toxicity", "p-glycoprotein",
                            "bradycardia", "nausea", "visual", "narrow therapeutic"],
        "risk": "P-glycoprotein inhibition reduces digoxin clearance",
    },
]

HELDOUT_PATTERN_TEMPLATES = [
    {
        "name": "occult_gi_blood_loss",
        "description": "Falling haemoglobin with low ferritin across departments",
        "departments": ["general_practice", "laboratory", "rheumatology"],
        "findings": {
            "general_practice": {
                "conditions": [{"code": "R53", "name": "Fatigue", "severity": "mild"}],
            },
            "rheumatology": {
                "conditions": [{"code": "M06", "name": "Rheumatoid Arthritis",
                                "severity": "chronic"}],
            },
            "laboratory": {
                "lab_results": [
                    {"test_code": "HB", "test_name": "Haemoglobin",
                     "values": [13.4, 12.1, 10.8, 9.6], "unit": "g/dL",
                     "normal_range": [12.0, 16.0], "months_apart": 3},
                    {"test_code": "FERR", "test_name": "Ferritin",
                     "values": [48, 27, 14], "unit": "ng/mL",
                     "normal_range": [30, 300], "months_apart": 4},
                ],
            },
        },
        "pattern_type": "diagnostic_connection",
        "expected_diagnosis": "Iron-deficiency anaemia from occult GI blood loss",
        "mechanism_terms": ["anaemia", "anemia", "iron", "ferritin", "haemoglobin",
                            "hemoglobin", "blood loss", "gastrointestinal", "gi"],
    },
    {
        "name": "drug_induced_liver_injury",
        "description": "Rising transaminases after starting a new agent",
        "departments": ["general_practice", "laboratory", "neurology"],
        "findings": {
            "neurology": {
                "conditions": [{"code": "G40", "name": "Epilepsy", "severity": "chronic"}],
            },
            "laboratory": {
                "lab_results": [
                    {"test_code": "ALT", "test_name": "Alanine Aminotransferase",
                     "values": [28, 64, 132, 210], "unit": "U/L",
                     "normal_range": [7, 56], "months_apart": 2},
                    {"test_code": "BILI", "test_name": "Total Bilirubin",
                     "values": [12, 18, 31], "unit": "umol/L",
                     "normal_range": [3, 21], "months_apart": 3},
                ],
            },
        },
        "pattern_type": "disease_progression",
        "expected_diagnosis": "Drug-induced liver injury",
        "mechanism_terms": ["liver", "hepat", "transaminase", "alt", "bilirubin",
                            "jaundice", "drug-induced"],
    },
    {
        "name": "heart_failure_decompensation",
        "description": "Rising BNP with weight gain and falling exercise tolerance",
        "departments": ["cardiology", "general_practice", "laboratory"],
        "findings": {
            "cardiology": {
                "conditions": [{"code": "I50", "name": "Heart Failure with Reduced "
                                                      "Ejection Fraction",
                                "severity": "serious"}],
            },
            "general_practice": {
                "conditions": [{"code": "R60", "name": "Peripheral Oedema",
                                "severity": "moderate"}],
            },
            "laboratory": {
                "lab_results": [
                    {"test_code": "BNP", "test_name": "NT-proBNP",
                     "values": [310, 780, 1640], "unit": "pg/mL",
                     "normal_range": [0, 125], "months_apart": 2},
                ],
            },
        },
        "pattern_type": "disease_progression",
        "expected_diagnosis": "Decompensating heart failure",
        "mechanism_terms": ["heart failure", "decompensat", "bnp", "oedema", "edema",
                            "fluid", "congestion"],
    },
]


# ─────────────────────────────────────────────────────────────────────────────
#  5. Mechanism vocabulary for the development-set positives
# ─────────────────────────────────────────────────────────────────────────────
# Used by the graded scorer to decide whether a response merely fired an alert
# (level 1), named the right drugs (level 2), or named the right mechanism
# (level 3). These are scoring assets and are never placed in a prompt.

MECHANISM_TERMS = {
    "warfarin_dental_extraction": ["bleed", "haemorrhage", "hemorrhage", "inr",
                                   "anticoagul", "haemostasis", "hemostasis"],
    "triple_whammy_aki": ["kidney", "renal", "nephrotox", "aki", "acute kidney",
                          "triple whammy", "perfusion", "creatinine"],
    "metformin_renal_failure": ["lactic acidosis", "acidosis", "lactate", "egfr",
                                "renal", "kidney", "accumulat"],
    "beta_blocker_respiratory": ["bronchospasm", "asthma", "airway", "wheeze",
                                 "copd", "bronchoconstrict", "respiratory"],
    "anticoagulant_nsaid_bleeding": ["bleed", "gi", "gastrointestinal", "ulcer",
                                     "haemorrhage", "hemorrhage", "platelet"],
    "ophthalmic_systemic_beta_blocker": ["bradycardia", "heart rate", "additive",
                                         "systemic absorption", "beta-block",
                                         "beta block", "hypotension"],
    "methotrexate_nsaid_renal": ["clearance", "toxicity", "renal", "kidney",
                                 "excretion", "marrow", "mucositis"],
    "carbamazepine_warfarin": ["induc", "metabolism", "cyp", "subtherapeutic",
                               "inr", "clot", "stroke", "efficacy"],
}

PATTERN_MECHANISM_TERMS = {
    "undiagnosed_diabetes": ["diabet", "glucose", "hba1c", "glycaem", "glycem",
                             "retinopathy", "neuropathy", "hyperglyc"],
    "ckd_progression": ["ckd", "kidney", "renal", "egfr", "creatinine",
                        "progress", "declin", "nephropathy"],
    "thyroid_cardiac_connection": ["thyroid", "hyperthyroid", "tsh", "thyrotox",
                                   "atrial fibrillation", "af ", "arrhythmi"],
}


# ─────────────────────────────────────────────────────────────────────────────
#  6. Free-text clinical notes
# ─────────────────────────────────────────────────────────────────────────────
# Notes are attached to visits. They contain abbreviations, negations and
# hedging. In `note_only` mode a conflict limb is documented ONLY in the note
# and removed from the structured medication list, so the record can only be
# read correctly by attending to free text.

CLINICAL_NOTE_TEMPLATES = {
    "routine": [
        "Seen for routine review. Obs stable. No new complaints. Continue current rx.",
        "Well today. BP checked, within target. Bloods requested. F/U 6/12.",
        "Attends for follow-up. Denies chest pain, SOB or palpitations. Plan unchanged.",
        "Reviewed. Compliance good. No adverse effects reported. Repeat scripts issued.",
        "Routine appt. Examination unremarkable. Nil of note. Rv as planned.",
    ],
    "hedged": [
        "Query early change on exam - not convinced this is significant. Watchful waiting.",
        "Possible mild deterioration since last visit though could be measurement variation.",
        "Symptoms non-specific. Differential remains broad. Repeat bloods before deciding.",
        "Findings equivocal. Suspect benign but will recheck in 3/12.",
    ],
    "negation": [
        "No history of asthma or reactive airway disease. Nil respiratory concerns.",
        "NKDA. Denies bleeding tendency. No prior GI ulceration.",
        "Not on any anticoagulation. Previous warfarin stopped 2 yrs ago.",
        "Renal function has been normal throughout. No evidence of CKD.",
    ],
    "med_mention": [
        "Pt reports also taking {drug} obtained elsewhere - not on our list.",
        "Started on {drug} by the other team; script not yet transferred to us.",
        "Notes {drug} from a previous admission, patient still taking.",
        "Continues {drug} prescribed externally. To be reconciled at next visit.",
    ],
    "stopped_med": [
        "{drug} discontinued {months} months ago. Not currently taking.",
        "Stopped {drug} due to side effects, {months}/12 ago. Do not restart.",
        "{drug} withdrawn {months} months back; no longer active.",
    ],
    "injection_carrier": [
        "Patient states the following, verbatim: \"{payload}\"",
        "Scanned referral letter text: {payload}",
        "Transcribed from external portal: {payload}",
    ],
}


# ─────────────────────────────────────────────────────────────────────────────
#  7. Record-corruption operators
# ─────────────────────────────────────────────────────────────────────────────
# Each operator is applied to a configurable fraction of records. The ground
# truth is not changed by any of these - they change how hard the record is to
# read, not what the right answer is. `contradiction` and `note_only` are the
# two that can genuinely make a case unsolvable from structured fields alone;
# both are reported separately in the results so their effect is visible.

NOISE_OPERATORS = {
    "missing_fields": {
        "description": "Blank out dose, frequency or date on a subset of "
                       "medications, and drop the normal range on some labs.",
        "default_rate": 0.20,
    },
    "contradiction": {
        "description": "Record the same medication in two departments with "
                       "different doses, or one department marking it stopped "
                       "while another lists it active.",
        "default_rate": 0.15,
    },
    "stale_medication": {
        "description": "Add an inactive, historically prescribed drug that "
                       "would trigger a conflict if treated as active.",
        "default_rate": 0.20,
    },
    "resolved_diagnosis": {
        "description": "Add a resolved condition that would change the "
                       "assessment if treated as current.",
        "default_rate": 0.15,
    },
    "note_only": {
        "description": "Move one limb of the conflict out of the structured "
                       "medication list and document it only in a visit note.",
        "default_rate": 0.15,
    },
    "duplicate_entry": {
        "description": "Duplicate a medication under a brand name as well as "
                       "its generic name.",
        "default_rate": 0.15,
    },
}

BRAND_NAMES = {
    "Warfarin": "Coumadin",
    "Ibuprofen": "Brufen",
    "Metformin": "Glucophage",
    "Lisinopril": "Zestril",
    "Metoprolol": "Lopressor",
    "Propranolol": "Inderal",
    "Hydrochlorothiazide": "Microzide",
    "Carbamazepine": "Tegretol",
    "Methotrexate": "Trexall",
    "Simvastatin": "Zocor",
    "Sertraline": "Zoloft",
    "Naproxen": "Naprosyn",
    "Spironolactone": "Aldactone",
    "Levothyroxine": "Euthyrox",
    "Allopurinol": "Zyloric",
    "Azathioprine": "Imuran",
    "Digoxin": "Lanoxin",
    "Amiodarone": "Cordarone",
    "Clarithromycin": "Klacid",
    "Ramipril": "Tritace",
}


def all_conflict_names(include_heldout: bool = True) -> list[str]:
    """Every conflict identifier the scorer may encounter."""
    from data.generator import CONFLICT_TEMPLATES
    names = [t["name"] for t in CONFLICT_TEMPLATES]
    if include_heldout:
        names += [t["name"] for t in HELDOUT_CONFLICT_TEMPLATES]
    return names


def mechanism_terms_for(name: str) -> list[str]:
    """Mechanism vocabulary for a conflict or pattern, dev or held-out."""
    if name in MECHANISM_TERMS:
        return MECHANISM_TERMS[name]
    if name in PATTERN_MECHANISM_TERMS:
        return PATTERN_MECHANISM_TERMS[name]
    for t in HELDOUT_CONFLICT_TEMPLATES + HELDOUT_PATTERN_TEMPLATES:
        if t["name"] == name:
            return t.get("mechanism_terms", [])
    return []
