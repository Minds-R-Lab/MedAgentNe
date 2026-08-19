"""Evidence fidelity must fall when assembled evidence is not in the record."""
from protocol.interactions import ClinicalEvidence
from simulation.fidelity import evidence_fidelity, aggregate_fidelity


class _Med:
    def __init__(self, name, category):
        self.name, self.category = name, category


class _Cond:
    def __init__(self, name):
        self.name = name


class _Lab:
    def __init__(self, test_name):
        self.test_name = test_name


class _Patient:
    medications = [_Med("Warfarin", "anticoagulant"), _Med("Ibuprofen", "nsaid")]
    conditions = [_Cond("Heart Failure")]
    lab_results = [_Lab("eGFR")]


def _ev(drugs=(), conds=(), labs=()):
    e = ClinicalEvidence()
    for d in drugs:
        e.add_drug(d, "", "cardiology")
    for c in conds:
        e.add_condition(c, "cardiology")
    for l in labs:
        e.add_lab(l, 1.0, "2024-01-01", "laboratory")
    return e


def test_faithful_evidence_scores_one():
    f = evidence_fidelity(_ev(["Warfarin", "Ibuprofen"], ["Heart Failure"], ["eGFR"]),
                          _Patient())
    assert f["overall_precision"]["rate"] == 1.0
    assert f["any_fabrication"] is False


def test_invented_drug_lowers_precision():
    f = evidence_fidelity(_ev(["Warfarin", "Metformin"]), _Patient())
    assert f["medications"]["precision"]["rate"] == 0.5
    assert f["any_fabrication"] is True
    assert "Metformin" in f["medications"]["unsupported"]


def test_recall_falls_with_disclosure_not_with_error():
    """Reporting less is minimisation: precision stays perfect."""
    f = evidence_fidelity(_ev(["Warfarin"]), _Patient())
    assert f["medications"]["precision"]["rate"] == 1.0
    assert f["medications"]["recall"]["rate"] == 0.5


def test_aggregate_pools_counts():
    a = evidence_fidelity(_ev(["Warfarin"]), _Patient())
    b = evidence_fidelity(_ev(["Metformin"]), _Patient())
    agg = aggregate_fidelity([a, b])
    assert agg["n_scenarios"] == 2
    assert agg["medications/precision"]["rate"] == 0.5
    assert agg["scenarios_with_any_fabrication"]["rate"] == 0.5
