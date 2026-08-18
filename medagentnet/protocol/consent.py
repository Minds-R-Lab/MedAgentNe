"""
MedAgentNet - Consent Management & Audit Trail
"""
import os
import json
import uuid
import hmac
import hashlib
import secrets
from datetime import datetime, timedelta
from typing import Optional
from protocol.models import AuditEntry, DisclosureTier

# Explicit denial marker. R0 revoked a department pair by deleting its entry
# from the consent profile; a missing entry then fell through to the opt-in
# default and the pair was silently re-permitted, so partial revocation had no
# effect. Denial is now represented explicitly and takes precedence over any
# default policy.
DENIED = "DENIED"


class ConsentManager:
    """Manages patient consent for inter-department data sharing."""

    def __init__(self, default_policy: str = "opt_in", emergency_override: bool = True,
                 token_ttl_seconds: int = 60, token_secret: str = None):
        self.default_policy = default_policy
        self.emergency_override = emergency_override
        # patient_id -> {(source_dept, target_dept): max_tier | DENIED}
        self.consent_profiles: dict[str, dict] = {}
        # Patients who have opted out entirely
        self.opt_outs: set[str] = set()

        # Consent-token machinery (used when the orchestrator runs with
        # validate_tokens=True). Tokens are bound to the patient, the department
        # pair and the tier, expire, and are single-use.
        self.token_ttl = timedelta(seconds=token_ttl_seconds)
        self._token_secret = (token_secret or secrets.token_hex(16)).encode()
        self._issued_tokens: dict[str, dict] = {}
        self._spent_tokens: set[str] = set()

    def register_patient(self, patient_id: str, departments: list[str]):
        """Register a patient with default consent for all their departments."""
        if self.default_policy == "opt_in":
            profile = {}
            for src in departments:
                for tgt in departments:
                    if src != tgt:
                        profile[(src, tgt)] = DisclosureTier.CLINICAL_SUMMARY
            self.consent_profiles[patient_id] = profile

    def check_consent(self, patient_id: str, source_dept: str,
                      target_dept: str, requested_tier: int,
                      is_emergency: bool = False) -> tuple[bool, int]:
        """
        Check if a query is allowed and return the maximum permitted tier.
        Returns: (is_allowed, max_tier)
        """
        profile = self.consent_profiles.get(patient_id, {})
        pair_setting = profile.get((source_dept, target_dept))

        # An explicit per-pair denial is honoured even under an emergency
        # override unless the patient has separately allowed break-glass access.
        # Emergency override applies to blanket opt-out, not to a targeted
        # restriction the patient set on a specific department pair.
        if pair_setting == DENIED:
            return False, 0

        if patient_id in self.opt_outs:
            if is_emergency and self.emergency_override:
                return True, DisclosureTier.FULL_CONTEXT
            return False, 0

        if is_emergency and self.emergency_override:
            return True, DisclosureTier.FULL_CONTEXT

        max_tier = pair_setting
        if max_tier is None:
            # Default: allow tier 2 for opt-in policy
            if self.default_policy == "opt_in":
                max_tier = DisclosureTier.CLINICAL_SUMMARY
            else:
                return False, 0

        allowed_tier = min(requested_tier, max_tier)
        return True, allowed_tier

    # ── Consent tokens ───────────────────────────────────────────────────

    def generate_consent_token(self, patient_id: str, source_dept: str,
                                target_dept: str, tier: int) -> str:
        """Issue a consent token bound to this exact authorisation."""
        nonce = uuid.uuid4().hex[:12]
        payload = f"{patient_id}|{source_dept}|{target_dept}|{int(tier)}|{nonce}"
        mac = hmac.new(self._token_secret, payload.encode(), hashlib.sha256).hexdigest()[:16]
        token = f"CST-{nonce}-{mac}"
        self._issued_tokens[token] = {
            "patient_id": patient_id,
            "source": source_dept,
            "target": target_dept,
            "tier": int(tier),
            "issued_at": datetime.now(),
        }
        return token

    def validate_consent_token(self, token: str, patient_id: str,
                                source_dept: str, target_dept: str,
                                tier: int) -> tuple[bool, str]:
        """Verify a consent token. Returns (is_valid, reason)."""
        meta = self._issued_tokens.get(token)
        if meta is None:
            return False, "unknown_token"
        if token in self._spent_tokens:
            return False, "replayed_token"
        if datetime.now() - meta["issued_at"] > self.token_ttl:
            return False, "expired_token"
        if meta["patient_id"] != patient_id:
            return False, "patient_mismatch"
        if meta["source"] != source_dept or meta["target"] != target_dept:
            return False, "department_pair_mismatch"
        if int(tier) > meta["tier"]:
            return False, "tier_escalation"
        self._spent_tokens.add(token)
        return True, "ok"

    # ── Revocation ───────────────────────────────────────────────────────

    def revoke_consent(self, patient_id: str, source_dept: str = None,
                       target_dept: str = None):
        """Revoke consent entirely, by department, or for one directed pair."""
        if source_dept is None and target_dept is None:
            self.opt_outs.add(patient_id)
            return

        profile = self.consent_profiles.setdefault(patient_id, {})

        if source_dept and target_dept:
            profile[(source_dept, target_dept)] = DENIED
            return

        # One side specified: deny every pair touching that department.
        for key in list(profile.keys()):
            if source_dept and key[0] == source_dept:
                profile[key] = DENIED
            if target_dept and key[1] == target_dept:
                profile[key] = DENIED

    def revoke_pairs(self, patient_id: str, pairs) -> int:
        """Deny a collection of directed department pairs. Returns count."""
        profile = self.consent_profiles.setdefault(patient_id, {})
        n = 0
        for src, tgt in pairs:
            profile[(src, tgt)] = DENIED
            n += 1
        return n

    def permitted_pairs(self, patient_id: str) -> list[tuple]:
        """Directed pairs currently permitted for this patient."""
        if patient_id in self.opt_outs:
            return []
        return [k for k, v in self.consent_profiles.get(patient_id, {}).items()
                if v != DENIED]

    def denied_pairs(self, patient_id: str) -> list[tuple]:
        return [k for k, v in self.consent_profiles.get(patient_id, {}).items()
                if v == DENIED]


class AuditTrail:
    """Immutable audit trail for all inter-agent communications."""

    def __init__(self, log_file: str = "data/audit_trail.jsonl"):
        self.log_file = log_file
        self.entries: list[AuditEntry] = []
        os.makedirs(os.path.dirname(log_file) if os.path.dirname(log_file) else ".", exist_ok=True)

    def log(self, event_type: str, source_agent: str = "", target_agent: str = "",
            patient_id: str = "", query_type: str = "", disclosure_tier: int = 2,
            consent_granted: bool = True, data_fields_shared: list[str] = None,
            outcome: str = ""):
        """Record an audit entry."""
        entry = AuditEntry(
            event_type=event_type,
            source_agent=source_agent,
            target_agent=target_agent,
            patient_id=patient_id,
            query_type=query_type,
            disclosure_tier=disclosure_tier,
            consent_granted=consent_granted,
            data_fields_shared=data_fields_shared or [],
            outcome=outcome,
        )
        self.entries.append(entry)

        # Append to file
        with open(self.log_file, "a") as f:
            f.write(json.dumps(entry.to_dict()) + "\n")

        return entry

    def get_patient_audit(self, patient_id: str) -> list[AuditEntry]:
        """Retrieve all audit entries for a patient."""
        return [e for e in self.entries if e.patient_id == patient_id]

    def get_privacy_report(self) -> dict:
        """Generate a privacy compliance report."""
        total = len(self.entries)
        if total == 0:
            return {"total_events": 0}

        consent_denied = sum(1 for e in self.entries if not e.consent_granted)
        tier_counts = {}
        for e in self.entries:
            t = int(e.disclosure_tier)  # Normalize DisclosureTier enums to plain ints
            tier_counts[t] = tier_counts.get(t, 0) + 1

        query_types = {}
        for e in self.entries:
            qt = e.query_type
            if qt:
                query_types[qt] = query_types.get(qt, 0) + 1

        return {
            "total_events": total,
            "consent_denied_count": consent_denied,
            "consent_denial_rate": round(consent_denied / total, 4) if total else 0,
            "tier_distribution": tier_counts,
            "query_type_distribution": query_types,
            "unique_patients": len(set(e.patient_id for e in self.entries)),
        }
