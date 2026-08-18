"""
MedAgentNet - Interaction knowledge base (revision R1)
======================================================

A curated, machine-readable interaction table used for two purposes:

1. **Grounding.** The orchestrator's synthesis step consults it, so that a
   cross-departmental alert is anchored to a named interaction with a stated
   mechanism rather than resting on the language model's recall. This is the
   retrieval-grounding component the reviewers asked for; in a deployment the
   table would be populated from DrugBank / RxNorm rather than hand-curated.

2. **Baseline.** The same table drives the standalone rule-based
   drug-interaction engine that MedAgentNet is compared against, so the
   comparison isolates the contribution of the agent architecture rather than
   of the underlying pharmacological knowledge.

Each rule states what must be simultaneously true. A rule fires only when every
required limb is present *and active*, which is what makes the drug-matched
negative controls discriminative: they contain the drugs but not the
combination.

`requires` entries may be:
    ("drug",      <name>)          a named medication, active
    ("category",  <category>)      any active medication of that class
    ("condition", <substring>)     an active condition whose name matches
    ("lab_below", <test>, <value>) a recent result below a threshold
    ("lab_above", <test>, <value>) a recent result above a threshold
    ("procedure", <substring>)     the encounter involves such a procedure
"""
from __future__ import annotations

INTERACTION_RULES = [
    # ── development-set interactions ──
    {
        "id": "warfarin_invasive_procedure",
        "severity": "high_risk",
        "label": "Anticoagulation with a planned invasive procedure",
        "requires": [("drug", "Warfarin"),
                     ("procedure", ("extraction", "surgery", "implant",
                                    "biopsy", "excision"))],
        "mechanism": "Anticoagulant therapy substantially increases peri-procedural "
                     "bleeding risk. Check INR and agree a bridging or "
                     "interruption plan with the prescribing team.",
        "entities": ["Warfarin"],
    },
    {
        "id": "anticoagulant_nsaid",
        "severity": "critical",
        "label": "Anticoagulant with NSAID",
        "requires": [("category", "anticoagulant"), ("category", "nsaid")],
        "mechanism": "Additive haemostatic impairment plus NSAID gastric mucosal "
                     "injury gives a high risk of upper gastrointestinal "
                     "haemorrhage.",
        "entities": ["Warfarin", "Ibuprofen"],
    },
    {
        "id": "triple_whammy",
        "severity": "critical",
        "label": "Triple whammy (ACE inhibitor + NSAID + diuretic)",
        "requires": [("category", "ace_inhibitor"), ("category", "nsaid"),
                     ("category", "diuretic")],
        "mechanism": "Loss of afferent arteriolar prostaglandin dilatation, "
                     "efferent constriction and volume depletion act together to "
                     "cause acute kidney injury.",
        "entities": ["Lisinopril", "Ibuprofen", "Hydrochlorothiazide"],
    },
    {
        "id": "ace_nsaid",
        "severity": "high_risk",
        "label": "ACE inhibitor with NSAID",
        "requires": [("category", "ace_inhibitor"), ("category", "nsaid")],
        "mechanism": "Reduced renal perfusion pressure with impaired "
                     "prostaglandin-mediated compensation raises the risk of "
                     "renal impairment.",
        "entities": ["Lisinopril", "Ibuprofen"],
    },
    {
        "id": "metformin_renal_impairment",
        "severity": "critical",
        "label": "Metformin with impaired renal function",
        "requires": [("drug", "Metformin"),
                     ("any_of", [("lab_below", "eGFR", 45),
                                 ("condition", ("chronic kidney disease stage 3",
                                                "chronic kidney disease stage 4",
                                                "chronic kidney disease stage 5",
                                                "acute kidney injury"))])],
        "mechanism": "Reduced clearance leads to metformin accumulation and a "
                     "risk of lactic acidosis. Review dose or withhold.",
        "entities": ["Metformin"],
    },
    {
        "id": "nonselective_bb_airway",
        "severity": "high_risk",
        "label": "Non-selective beta-blocker with reactive airway disease",
        "requires": [("drug", "Propranolol"),
                     ("condition", ("asthma", "copd", "chronic obstructive",
                                    "reactive airway"))],
        "mechanism": "Beta-2 blockade in the airway can precipitate "
                     "bronchospasm. A cardioselective agent is preferred.",
        "entities": ["Propranolol"],
    },
    {
        "id": "ophthalmic_systemic_beta_blockade",
        "severity": "high_risk",
        "label": "Topical and systemic beta-blockade",
        "requires": [("category", "beta_blocker_ophthalmic"),
                     ("category", "beta_blocker")],
        "mechanism": "Systemic absorption of ophthalmic timolol adds to oral "
                     "beta-blockade, giving additive bradycardia and hypotension.",
        "entities": ["Timolol Eye Drops", "Metoprolol"],
    },
    {
        "id": "methotrexate_nsaid",
        "severity": "high_risk",
        "label": "Methotrexate with NSAID",
        "requires": [("drug", "Methotrexate"), ("category", "nsaid")],
        "mechanism": "NSAIDs reduce renal tubular clearance of methotrexate, "
                     "raising exposure and the risk of marrow suppression and "
                     "mucositis.",
        "entities": ["Methotrexate", "Ibuprofen"],
    },
    {
        "id": "carbamazepine_warfarin",
        "severity": "high_risk",
        "label": "Enzyme inducer with warfarin",
        "requires": [("drug", "Carbamazepine"), ("drug", "Warfarin")],
        "mechanism": "Carbamazepine induces warfarin metabolism, giving a "
                     "subtherapeutic INR and a raised thrombotic risk.",
        "entities": ["Carbamazepine", "Warfarin"],
    },

    # ── held-out interactions (not referenced by any prompt or agent rule) ──
    {
        "id": "ssri_nsaid",
        "severity": "high_risk",
        "label": "SSRI with NSAID",
        "requires": [("category", "ssri"), ("category", "nsaid")],
        "mechanism": "Serotonin-dependent platelet aggregation is impaired, "
                     "adding to NSAID mucosal injury and raising upper "
                     "gastrointestinal bleeding risk.",
        "entities": ["Sertraline", "Naproxen"],
    },
    {
        "id": "statin_macrolide",
        "severity": "critical",
        "label": "Simvastatin with a macrolide",
        "requires": [("drug", "Simvastatin"), ("category", "macrolide_antibiotic")],
        "mechanism": "CYP3A4 inhibition raises simvastatin exposure several-fold, "
                     "with a risk of myopathy and rhabdomyolysis.",
        "entities": ["Simvastatin", "Clarithromycin"],
    },
    {
        "id": "ace_potassium_sparing",
        "severity": "critical",
        "label": "ACE inhibitor with a potassium-sparing diuretic",
        "requires": [("category", "ace_inhibitor"),
                     ("category", "potassium_sparing_diuretic")],
        "mechanism": "Additive potassium retention risks hyperkalaemia and "
                     "cardiac arrhythmia. Monitor serum potassium.",
        "entities": ["Ramipril", "Spironolactone"],
    },
    {
        "id": "levothyroxine_calcium",
        "severity": "high_risk",
        "label": "Levothyroxine with calcium supplementation",
        "requires": [("drug", "Levothyroxine"), ("category", "mineral_supplement")],
        "mechanism": "Calcium chelates levothyroxine in the gut and lowers its "
                     "absorption, producing a rising TSH. Separate the doses.",
        "entities": ["Levothyroxine", "Calcium Carbonate"],
    },
    {
        "id": "allopurinol_azathioprine",
        "severity": "critical",
        "label": "Allopurinol with azathioprine",
        "requires": [("drug", "Allopurinol"), ("drug", "Azathioprine")],
        "mechanism": "Xanthine oxidase inhibition blocks azathioprine catabolism, "
                     "causing accumulation and profound bone-marrow suppression.",
        "entities": ["Allopurinol", "Azathioprine"],
    },
    {
        "id": "digoxin_amiodarone",
        "severity": "critical",
        "label": "Digoxin with amiodarone",
        "requires": [("drug", "Digoxin"), ("drug", "Amiodarone")],
        "mechanism": "P-glycoprotein inhibition reduces digoxin clearance; with a "
                     "narrow therapeutic index this readily reaches toxicity.",
        "entities": ["Digoxin", "Amiodarone"],
    },
]


PATTERN_RULES = [
    {
        "id": "undiagnosed_diabetes",
        "severity": "high_risk",
        "label": "Multi-system picture consistent with type 2 diabetes",
        "requires": [
            ("any_of", [("lab_rising", "HbA1c", 5.6), ("lab_rising", "Fasting Glucose", 5.5)]),
            ("condition_count", ("retinopathy", "neuropathy", "elevated fasting glucose"), 2),
        ],
        "mechanism": "Rising glycaemic indices together with retinal and "
                     "peripheral nerve findings across departments indicate "
                     "diabetes that has not been formally diagnosed.",
        "entities": ["Diabetes", "Mellitus"],
    },
    {
        "id": "ckd_progression",
        "severity": "high_risk",
        "label": "Progressive chronic kidney disease",
        "requires": [("lab_falling", "eGFR", 10), ("lab_below", "eGFR", 90)],
        "mechanism": "A sustained fall in eGFR with rising creatinine indicates "
                     "progression and requires nephrotoxin review and restaging.",
        "entities": ["CKD", "Progression"],
    },
    {
        "id": "thyroid_cardiac",
        "severity": "high_risk",
        "label": "Thyrotoxic atrial fibrillation",
        "requires": [("lab_below", "TSH", 0.4),
                     ("condition", ("atrial fibrillation",))],
        "mechanism": "A suppressed TSH alongside new atrial fibrillation points "
                     "to thyrotoxicosis as the driver of the arrhythmia.",
        "entities": ["Hyperthyroid", "Atrial", "Fibrillation"],
    },
    {
        "id": "occult_gi_blood_loss",
        "severity": "high_risk",
        "label": "Iron-deficiency anaemia suggesting occult blood loss",
        "requires": [("lab_falling", "Haemoglobin", 1.5),
                     ("lab_below", "Ferritin", 30)],
        "mechanism": "Falling haemoglobin with depleted ferritin indicates iron "
                     "deficiency; in this context occult gastrointestinal blood "
                     "loss must be excluded.",
        "entities": ["anaemia", "iron", "blood", "loss"],
    },
    {
        "id": "drug_induced_liver_injury",
        "severity": "critical",
        "label": "Rising transaminases suggesting drug-induced liver injury",
        "requires": [("lab_rising", "Alanine Aminotransferase", 56)],
        "mechanism": "A progressive transaminase rise with rising bilirubin "
                     "after starting a new agent indicates hepatocellular injury.",
        "entities": ["liver", "injury", "drug-induced"],
    },
    {
        "id": "heart_failure_decompensation",
        "severity": "high_risk",
        "label": "Decompensating heart failure",
        "requires": [("lab_rising", "NT-proBNP", 125),
                     ("condition", ("heart failure", "oedema", "edema"))],
        "mechanism": "A steeply rising natriuretic peptide with peripheral "
                     "oedema indicates worsening congestion.",
        "entities": ["heart", "failure", "decompensat"],
    },
]


# ─────────────────────────────────────────────────────────────────────────────
#  Evidence model and rule evaluation
# ─────────────────────────────────────────────────────────────────────────────

class ClinicalEvidence:
    """The union of what department agents disclosed, in structured form.

    This is deliberately built from *agent responses*, not from raw patient
    records, so the synthesis step never sees more than the disclosure tiers
    permitted.
    """

    def __init__(self):
        self.drugs: dict[str, dict] = {}      # name -> {category, department}
        self.categories: set[str] = set()
        self.conditions: list[dict] = []      # {name, department}
        self.labs: dict[str, list] = {}       # test -> [(date, value)]
        self.procedure: str = ""
        self.departments: set[str] = set()

    def add_drug(self, name: str, category: str = "", department: str = ""):
        if not name:
            return
        key = name.strip()
        self.drugs[key] = {"category": (category or "").strip(),
                           "department": department}
        if category:
            self.categories.add(category.strip().lower())
        if department:
            self.departments.add(department)

    def add_condition(self, name: str, department: str = ""):
        if name:
            self.conditions.append({"name": name.strip(), "department": department})
            if department:
                self.departments.add(department)

    def add_lab(self, test: str, value, date: str = "", department: str = ""):
        if not test:
            return
        try:
            v = float(value)
        except (TypeError, ValueError):
            return
        self.labs.setdefault(test.strip(), []).append((date or "", v))
        if department:
            self.departments.add(department)

    # ── predicates ──

    def has_drug(self, name: str) -> bool:
        target = name.lower()
        return any(target in d.lower() for d in self.drugs)

    def has_category(self, cat: str) -> bool:
        c = cat.lower()
        return c in self.categories or any(c in x for x in self.categories)

    def has_condition(self, needles) -> bool:
        if isinstance(needles, str):
            needles = (needles,)
        for cond in self.conditions:
            low = cond["name"].lower()
            if any(n.lower() in low for n in needles):
                return True
        return False

    def condition_count(self, needles) -> int:
        seen = set()
        for cond in self.conditions:
            low = cond["name"].lower()
            for n in needles:
                if n.lower() in low:
                    seen.add(n)
        return len(seen)

    def _series(self, test: str):
        for name, vals in self.labs.items():
            if test.lower() in name.lower():
                return [v for _, v in sorted(vals, key=lambda x: x[0])]
        return []

    def lab_below(self, test: str, threshold: float) -> bool:
        s = self._series(test)
        return bool(s) and min(s) < threshold

    def lab_above(self, test: str, threshold: float) -> bool:
        s = self._series(test)
        return bool(s) and max(s) > threshold

    def lab_rising(self, test: str, above: float) -> bool:
        s = self._series(test)
        return len(s) >= 3 and s[-1] > s[0] and s[-1] > above

    def lab_falling(self, test: str, by: float) -> bool:
        s = self._series(test)
        return len(s) >= 3 and (s[0] - s[-1]) >= by

    def drug_departments(self, names) -> set:
        out = set()
        for n in names:
            for d, meta in self.drugs.items():
                if n.lower() in d.lower() and meta.get("department"):
                    out.add(meta["department"])
        return out

    def summary(self) -> dict:
        return {
            "n_drugs": len(self.drugs),
            "n_conditions": len(self.conditions),
            "n_lab_series": len(self.labs),
            "departments": sorted(self.departments),
        }


def _check(requirement, ev: ClinicalEvidence) -> bool:
    kind = requirement[0]
    if kind == "drug":
        return ev.has_drug(requirement[1])
    if kind == "category":
        return ev.has_category(requirement[1])
    if kind == "condition":
        return ev.has_condition(requirement[1])
    if kind == "condition_count":
        return ev.condition_count(requirement[1]) >= requirement[2]
    if kind == "lab_below":
        return ev.lab_below(requirement[1], requirement[2])
    if kind == "lab_above":
        return ev.lab_above(requirement[1], requirement[2])
    if kind == "lab_rising":
        return ev.lab_rising(requirement[1], requirement[2])
    if kind == "lab_falling":
        return ev.lab_falling(requirement[1], requirement[2])
    if kind == "procedure":
        needles = requirement[1]
        if isinstance(needles, str):
            needles = (needles,)
        return any(n in ev.procedure for n in needles)
    if kind == "any_of":
        return any(_check(sub, ev) for sub in requirement[1])
    return False


def evaluate_rules(ev: ClinicalEvidence, rules=None) -> list[dict]:
    """Return every rule whose requirements are all satisfied."""
    hits = []
    for rule in (rules if rules is not None else INTERACTION_RULES):
        if all(_check(req, ev) for req in rule["requires"]):
            involved = [d for d in ev.drugs
                        if any(e.lower() in d.lower() or d.lower() in e.lower()
                               for e in rule["entities"])]
            hits.append({
                "id": rule["id"],
                "severity": rule["severity"],
                "label": rule["label"],
                "mechanism": rule["mechanism"],
                "entities": rule["entities"],
                "involved_medications": involved or rule["entities"],
                "involved_departments": sorted(
                    ev.drug_departments(rule["entities"]) or ev.departments
                ),
            })
    return hits


def evaluate_patterns(ev: ClinicalEvidence) -> list[dict]:
    return evaluate_rules(ev, PATTERN_RULES)
