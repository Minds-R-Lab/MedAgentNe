"""Regression tests for the R0 defect in which partial consent revocation had
no effect.

R0 revoked a department pair with ``del consent_profiles[pid][pair]``. A missing
entry then resolved to the opt-in default and the pair was silently
re-permitted, so the reported "graceful degradation up to 60% revocation"
described an experiment that did not run. The recorded R0 results show zero
consent denials in both partial conditions.
"""
import pytest

from protocol.consent import ConsentManager, DENIED
from protocol.models import DisclosureTier


@pytest.fixture
def cm():
    m = ConsentManager(default_policy="opt_in", emergency_override=True)
    m.register_patient("PAT-1", ["cardiology", "dental", "laboratory"])
    return m


def test_revocation_actually_denies(cm):
    allowed, _ = cm.check_consent("PAT-1", "dental", "cardiology", 2)
    assert allowed
    cm.revoke_consent("PAT-1", "dental", "cardiology")
    allowed, tier = cm.check_consent("PAT-1", "dental", "cardiology", 2)
    assert not allowed and tier == 0


def test_denial_beats_the_default_policy(cm):
    """The R0 bug in one assertion: a denied pair must not fall through."""
    cm.revoke_pairs("PAT-1", [("dental", "cardiology")])
    assert cm.consent_profiles["PAT-1"][("dental", "cardiology")] == DENIED
    allowed, _ = cm.check_consent("PAT-1", "dental", "cardiology", 2)
    assert not allowed


def test_revocation_is_directional(cm):
    cm.revoke_consent("PAT-1", "dental", "cardiology")
    allowed, _ = cm.check_consent("PAT-1", "cardiology", "dental", 2)
    assert allowed, "revoking A->B must not revoke B->A"


def test_emergency_override_does_not_defeat_a_targeted_restriction(cm):
    """A blanket opt-out may be overridden in an emergency; a restriction the
    patient placed on a specific pair may not."""
    cm.revoke_consent("PAT-1", "dental", "cardiology")
    allowed, _ = cm.check_consent("PAT-1", "dental", "cardiology", 3, is_emergency=True)
    assert not allowed

    cm.register_patient("PAT-2", ["cardiology", "dental"])
    cm.revoke_consent("PAT-2")                      # global opt-out
    allowed, tier = cm.check_consent("PAT-2", "dental", "cardiology", 3, is_emergency=True)
    assert allowed and tier == DisclosureTier.FULL_CONTEXT


def test_tier_is_the_minimum_of_requested_and_permitted(cm):
    cm.consent_profiles["PAT-1"][("dental", "cardiology")] = DisclosureTier.FLAG_ONLY
    allowed, tier = cm.check_consent("PAT-1", "dental", "cardiology", 3)
    assert allowed and tier == DisclosureTier.FLAG_ONLY


def test_consent_token_is_single_use_and_bound(cm):
    tok = cm.generate_consent_token("PAT-1", "dental", "cardiology", 2)
    ok, why = cm.validate_consent_token(tok, "PAT-1", "dental", "cardiology", 2)
    assert ok, why
    ok, why = cm.validate_consent_token(tok, "PAT-1", "dental", "cardiology", 2)
    assert not ok and why == "replayed_token"


def test_consent_token_rejects_patient_and_tier_substitution(cm):
    tok = cm.generate_consent_token("PAT-1", "dental", "cardiology", 2)
    ok, why = cm.validate_consent_token(tok, "PAT-OTHER", "dental", "cardiology", 2)
    assert not ok and why == "patient_mismatch"

    tok2 = cm.generate_consent_token("PAT-1", "dental", "cardiology", 2)
    ok, why = cm.validate_consent_token(tok2, "PAT-1", "dental", "cardiology", 3)
    assert not ok and why == "tier_escalation"


def test_restriction_reduces_responses_end_to_end():
    """Integration: revoking pairs must reduce what comes back."""
    from simulation.runner_v2 import HardRunner
    base = HardRunner(seed=11, num_patients=30)
    base.generate(); base.build_scenarios()
    n_before = sum(r["num_responses"] for r in base.run())

    restricted = HardRunner(seed=11, num_patients=30)
    restricted.generate()
    for p in restricted.patients:
        pairs = list(restricted.consent.consent_profiles.get(p.patient_id, {}))
        restricted.consent.revoke_pairs(p.patient_id, pairs)
    restricted.build_scenarios()
    results = restricted.run()
    n_after = sum(r["num_responses"] for r in results)
    denials = sum(r["privacy_report"]["consent_denied"] for r in results)

    assert denials > 0, "revocation produced no denials (the R0 defect)"
    assert n_after < n_before
