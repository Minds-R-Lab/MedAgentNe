"""
MedAgentNet - AI Agents
DepartmentAgent: Specialized agent per medical department.
OrchestratorAgent: Routes queries and synthesizes responses.
"""
import json
import logging
from typing import Optional

from protocol.models import (
    PatientRecord, ClinicalQuery, AgentResponse, ConflictAlert,
    DisclosureTier, AlertLevel
)
from protocol.consent import ConsentManager, AuditTrail
from llm.provider import BaseLLMProvider
from llm.prompts import (
    get_department_system_prompt,
    get_orchestrator_system_prompt,
    get_synthesis_prompt,
)

logger = logging.getLogger("medagentnet.agents")


class DepartmentAgent:
    """
    A department-specialized AI agent. Each agent:
    - Has access ONLY to its department's patient records
    - Processes incoming queries using an LLM
    - Enforces disclosure tier restrictions on responses
    """

    # Fields a requesting clinician is allowed to place in a query context.
    # Anything else is dropped before prompt construction. This is a structural
    # guard: in R0 the scenario driver copied ground-truth fields
    # ("reason", "expected", per-department medication lists) into the context,
    # which reached every agent prompt at every tier.
    ALLOWED_CONTEXT_KEYS = (
        "planned_procedure",
        "query_reason",
        "relevant_categories",
        "clinical_notes",
        "urgency",
    )

    def __init__(self, department_id: str, department_config: dict,
                 llm: BaseLLMProvider, audit: AuditTrail,
                 enforce_tiers: bool = True,
                 structured_output: bool = True,
                 freetext_fallback: bool = True,
                 strict_context: bool = True):
        self.department_id = department_id
        self.config = department_config
        self.name = department_config.get("name", department_id)
        self.description = department_config.get("description", "")
        self.llm = llm
        self.audit = audit

        # ── Ablation switches (R1) ──
        # enforce_tiers=False    -> disclosure tiers ignored, agents answer in full
        # structured_output=False-> free-text prompting, no JSON contract
        # freetext_fallback=False-> a response that is not valid JSON is a failure
        # strict_context=False   -> reproduces the R0 behaviour of passing the
        #                           whole context through (used to quantify the
        #                           size of the R0 leak)
        self.enforce_tiers = enforce_tiers
        self.structured_output = structured_output
        self.freetext_fallback = freetext_fallback
        self.strict_context = strict_context

        # Local patient data store (only this department's slice)
        self.patient_store: dict[str, dict] = {}

        # Per-response disclosure ledger, consumed by the leakage metric
        self.disclosure_log: list[dict] = []

        self.system_prompt = get_department_system_prompt(
            department_id, self.name, self.description,
            structured=self.structured_output,
        )

    def load_patient_data(self, patient: PatientRecord):
        """Load ONLY this department's slice of the patient record."""
        dept_data = patient.get_department_records(self.department_id)
        self.patient_store[patient.patient_id] = dept_data

    def sanitize_context(self, context: dict) -> tuple[dict, list[str]]:
        """Restrict a query context to clinician-supplied fields.

        Returns (clean_context, dropped_keys).
        """
        if not self.strict_context:
            return dict(context), []
        clean, dropped = {}, []
        for k, v in context.items():
            if k in self.ALLOWED_CONTEXT_KEYS:
                clean[k] = v
            else:
                dropped.append(k)
        return clean, dropped

    def process_query(self, query: ClinicalQuery) -> AgentResponse:
        """Process an incoming clinical query and return a response."""
        patient_id = query.patient_id
        patient_data = self.patient_store.get(patient_id)

        if not patient_data:
            return AgentResponse(
                query_id=query.query_id,
                source_agent=self.department_id,
                patient_id=patient_id,
                summary=f"No records found for patient in {self.name}.",
            )

        # Build the user prompt with patient context and query
        tier = query.disclosure_tier
        user_prompt = self._build_user_prompt(query, patient_data, tier)

        # Get LLM response (with error handling)
        try:
            raw_response = self.llm.generate(self.system_prompt, user_prompt)
        except Exception as e:
            logger.error(f"LLM error for {self.department_id} (patient {patient_id}): {e}")
            raw_response = json.dumps({
                "findings": [{"type": "llm_error", "severity": "low",
                              "description": f"Agent could not process query: {str(e)[:100]}"}],
                "medications_reported": [],
                "conditions_reported": [],
                "risk_flags": [],
                "summary": f"Error processing query in {self.name}."
            })

        # Parse and filter response based on disclosure tier
        response = self._parse_response(raw_response, query, tier)

        # Record what actually crossed the department boundary, for the
        # information-leakage metric (see simulation/privacy.py).
        self.disclosure_log.append({
            "patient_id": patient_id,
            "query_id": query.query_id,
            "requesting_department": query.source_agent,
            "responding_department": self.department_id,
            "tier": int(tier),
            "disclosed_text": self._disclosed_text(response),
            "n_findings": len(response.findings),
            "n_medications": len(response.medications_reported),
            "n_conditions": len(response.conditions_reported),
            "n_labs": len(response.lab_results_reported),
        })

        # Audit log
        self.audit.log(
            event_type="query_processed",
            source_agent=query.source_agent,
            target_agent=self.department_id,
            patient_id=patient_id,
            query_type=query.query_type,
            disclosure_tier=tier,
            data_fields_shared=self._get_shared_fields(response, tier),
            outcome="response_sent",
        )

        return response

    @staticmethod
    def _disclosed_text(response: AgentResponse) -> str:
        """Concatenate everything the response actually transmitted."""
        parts = [response.summary or ""]
        for f in response.findings:
            if isinstance(f, dict):
                parts.append(str(f.get("description", "")))
                parts.append(str(f.get("type", "")))
        for coll in (response.medications_reported, response.conditions_reported,
                     response.lab_results_reported):
            for item in coll:
                parts.append(json.dumps(item) if isinstance(item, dict) else str(item))
        parts.extend(str(x) for x in response.risk_flags)
        return " ".join(p for p in parts if p)

    def _build_user_prompt(self, query: ClinicalQuery, patient_data: dict,
                            tier: int) -> str:
        """Build the user prompt from query and local patient data.

        Disclosure tiers govern the *record* section:
          Tier 1  category-level flags only, no names, values or dates
          Tier 2  medication names and doses, coded diagnoses, numeric labs
          Tier 3  Tier 2 plus free-text clinical notes and full histories
        With ``enforce_tiers=False`` every section is rendered at Tier 3.
        """
        context, dropped = self.sanitize_context(query.clinical_context)
        if dropped:
            logger.debug(
                f"[{self.department_id}] dropped {len(dropped)} non-clinician "
                f"context key(s): {dropped}"
            )
        effective_tier = 3 if not self.enforce_tiers else tier

        prompt_parts = [
            f"QUERY TYPE: {query.query_type}",
            f"FROM: {query.source_agent}",
            f"DISCLOSURE TIER: {effective_tier}",
            f"CLINICAL CONTEXT: {json.dumps(context, indent=2)}",
            "",
            f"PATIENT DATA (from {self.name} records only):",
        ]

        # ── Medications ──
        meds = patient_data.get("medications", [])
        if meds:
            active = [m for m in meds if m.get("active", True)]
            inactive = [m for m in meds if not m.get("active", True)]
            prompt_parts.append(
                f"Medications on file ({len(active)} active, {len(inactive)} inactive):"
            )
            for m in meds:
                status = "ACTIVE" if m.get("active", True) else "DISCONTINUED"
                if effective_tier >= 2:
                    dose = m.get("dose") or "dose not recorded"
                    freq = m.get("frequency") or "frequency not recorded"
                    date = m.get("prescribed_date") or "date not recorded"
                    prompt_parts.append(
                        f"  - [{status}] {m['name']} ({m['category']}): "
                        f"{dose} {freq}, prescribed {date}"
                    )
                else:
                    prompt_parts.append(
                        f"  - [{status}] [medication in category: {m['category']}]"
                    )

        # ── Conditions ──
        conditions = patient_data.get("conditions", [])
        if conditions:
            active_c = [c for c in conditions if c.get("active", True)]
            prompt_parts.append(
                f"Conditions on file ({len(active_c)} active, "
                f"{len(conditions) - len(active_c)} resolved/inactive):"
            )
            for c in conditions:
                status = "ACTIVE" if c.get("active", True) else "RESOLVED"
                if effective_tier >= 2:
                    prompt_parts.append(
                        f"  - [{status}] {c['name']} ({c['code']}): "
                        f"severity={c['severity']}, "
                        f"since {c.get('diagnosed_date', 'unknown')}"
                    )
                else:
                    prompt_parts.append(
                        f"  - [{status}] [condition, severity: {c['severity']}]"
                    )

        # ── Laboratory ──
        labs = patient_data.get("lab_results", [])
        if labs:
            prompt_parts.append(f"Laboratory results ({len(labs)}):")
            ordered = sorted(labs, key=lambda x: x.get("date", ""))
            for l in ordered:
                if effective_tier >= 2:
                    flag = " [OUT OF RANGE]" if l.get("is_abnormal") else ""
                    nr = l.get("normal_range")
                    nr_txt = (f"reference {nr[0]}-{nr[1]}"
                              if nr and tuple(nr) != (0, 0) else "reference not recorded")
                    unit = l.get("unit") or ""
                    prompt_parts.append(
                        f"  - {l.get('date', '')}: {l['test_name']} = "
                        f"{l['value']}{unit}{flag} ({nr_txt})"
                    )
                else:
                    if l.get("is_abnormal"):
                        prompt_parts.append(
                            f"  - [out-of-range result in {l['test_name']}]"
                        )

        # ── Free-text notes: Tier 3 only ──
        visits = patient_data.get("visits", [])
        noted = [v for v in visits if v.get("notes")]
        if noted:
            if effective_tier >= 3:
                prompt_parts.append(f"Clinical notes ({len(noted)}):")
                for v in sorted(noted, key=lambda x: x.get("date", "")):
                    prompt_parts.append(
                        f"  - {v.get('date','')} ({v.get('physician','')}): {v['notes']}"
                    )
            elif effective_tier == 2:
                prompt_parts.append(
                    f"Clinical notes: {len(noted)} note(s) on file, withheld at "
                    f"Tier 2. Request Tier 3 if narrative detail is required."
                )

        prompt_parts.append("")
        if self.structured_output:
            prompt_parts.append(
                "Answer the clinical query using only the data above. "
                "Report only findings relevant to the query. Treat DISCONTINUED "
                "medications and RESOLVED conditions as no longer in effect. "
                "Return structured JSON."
            )
        else:
            prompt_parts.append(
                "Answer the clinical query using only the data above, in plain "
                "prose. Report only findings relevant to the query. Treat "
                "DISCONTINUED medications and RESOLVED conditions as no longer "
                "in effect."
            )

        return "\n".join(prompt_parts)

    def _parse_response(self, raw: str, query: ClinicalQuery,
                         tier: int) -> AgentResponse:
        """Parse LLM response and enforce disclosure tier."""
        data = None
        try:
            # Try to extract JSON from response
            raw = raw.strip()

            # Try multiple extraction strategies
            # Strategy 1: Direct JSON parse
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                pass

            # Strategy 2: Extract from markdown code block
            if data is None and "```" in raw:
                for block_type in ["```json", "```JSON", "```"]:
                    if block_type in raw:
                        try:
                            json_str = raw.split(block_type, 1)[1].split("```", 1)[0].strip()
                            data = json.loads(json_str)
                            break
                        except (json.JSONDecodeError, IndexError):
                            continue

            # Strategy 3: Find first { ... } block
            if data is None:
                start = raw.find("{")
                end = raw.rfind("}")
                if start != -1 and end != -1 and end > start:
                    try:
                        data = json.loads(raw[start:end + 1])
                    except json.JSONDecodeError:
                        pass

            # Strategy 4: Extract key information from free text.
            # With freetext_fallback disabled (ablation), a non-JSON response is
            # recorded as a format failure rather than being rescued by keyword
            # matching. This separates the LLM's contribution from the parser's.
            if data is None:
                if self.freetext_fallback:
                    data = self._extract_from_freetext(raw)
                else:
                    data = {
                        "findings": [{"type": "format_error", "severity": "low",
                                      "description": "Response was not valid JSON."}],
                        "risk_flags": [],
                        "summary": "",
                    }

        except Exception as e:
            logger.warning(f"Response parse error in {self.department_id}: {e}")
            data = {
                "findings": [{"type": "parse_error", "severity": "low",
                              "description": raw[:300]}],
                "risk_flags": [],
                "summary": raw[:200] if raw else "No response.",
            }

        # Normalize findings: LLMs sometimes return strings instead of dicts
        raw_findings = data.get("findings", [])
        normalized_findings = []
        for f in raw_findings:
            if isinstance(f, dict):
                normalized_findings.append(f)
            elif isinstance(f, str):
                # Convert string finding to a structured dict
                f_lower = f.lower()
                severity = "moderate"
                ftype = "observation"
                if any(kw in f_lower for kw in ["critical", "contraindicated", "emergency"]):
                    severity = "critical"
                    ftype = "medication_conflict"
                elif any(kw in f_lower for kw in ["conflict", "interaction", "risk", "bleeding",
                                                    "warfarin", "renal", "kidney", "metformin"]):
                    severity = "high"
                    ftype = "medication_conflict"
                elif any(kw in f_lower for kw in ["no concern", "no conflict", "no issue",
                                                    "no significant", "routine"]):
                    severity = "low"
                    ftype = "no_conflict"
                normalized_findings.append({
                    "type": ftype,
                    "severity": severity,
                    "description": f,
                })
            else:
                normalized_findings.append({
                    "type": "observation",
                    "severity": "low",
                    "description": str(f),
                })
        data["findings"] = normalized_findings

        # Filter based on tier
        if self.enforce_tiers and tier == DisclosureTier.FLAG_ONLY:
            # Strip specific medication/condition names
            filtered_findings = []
            for f in data.get("findings", []):
                if f.get("type") != "no_conflict":
                    filtered_findings.append({
                        "type": f.get("type"),
                        "severity": f.get("severity"),
                        "description": f"Relevant {f.get('type', 'finding')} exists in {self.name}",
                    })
            data["findings"] = filtered_findings
            data["medications_reported"] = []
            data["conditions_reported"] = []
            data["summary"] = f"Relevant findings exist in {self.name}. Escalate for details."

        # Normalize risk_flags to strings (LLMs sometimes return dicts)
        raw_flags = data.get("risk_flags", [])
        if not isinstance(raw_flags, list):
            raw_flags = [raw_flags] if raw_flags else []
        clean_flags = []
        for rf in raw_flags:
            if isinstance(rf, str):
                clean_flags.append(rf)
            elif isinstance(rf, dict):
                clean_flags.append(rf.get("type", rf.get("name", rf.get("flag", str(rf)))))
            else:
                clean_flags.append(str(rf))

        # Normalize list fields (LLMs sometimes return strings or dicts)
        def _ensure_list(val):
            if isinstance(val, list):
                return val
            if isinstance(val, str):
                return [val] if val.strip() else []
            return []

        # Ensure summary is a string
        summary = data.get("summary", "")
        if not isinstance(summary, str):
            summary = str(summary) if summary else ""

        return AgentResponse(
            query_id=query.query_id,
            source_agent=self.department_id,
            patient_id=query.patient_id,
            disclosure_tier=tier,
            findings=data.get("findings", []),
            medications_reported=_ensure_list(data.get("medications_reported", [])),
            conditions_reported=_ensure_list(data.get("conditions_reported", [])),
            lab_results_reported=_ensure_list(data.get("lab_results_reported", [])),
            risk_flags=clean_flags,
            summary=summary,
            raw_llm_response=raw[:500],
        )

    def _extract_from_freetext(self, text: str) -> dict:
        """Extract structured info from free-text LLM response when JSON fails."""
        text_lower = text.lower()
        findings = []
        risk_flags = []

        # Keyword-based extraction
        risk_keywords = {
            "bleeding": ("bleeding_risk", "high", "Bleeding risk identified"),
            "warfarin": ("medication_conflict", "high", "Warfarin-related concern identified"),
            "kidney": ("renal_risk", "high", "Renal risk concern identified"),
            "renal": ("renal_risk", "high", "Renal risk concern identified"),
            "triple whammy": ("medication_conflict", "critical", "Triple whammy nephrotoxic combination detected"),
            "lactic acidosis": ("medication_conflict", "critical", "Lactic acidosis risk with metformin"),
            "bradycardia": ("medication_conflict", "high", "Bradycardia risk from beta-blocker interaction"),
            "bronchospasm": ("medication_conflict", "high", "Bronchospasm risk with beta-blocker in airway disease"),
            "methotrexate": ("medication_conflict", "high", "Methotrexate interaction concern"),
            "subtherapeutic": ("medication_conflict", "high", "Subtherapeutic drug levels detected"),
            "contraindicated": ("medication_conflict", "critical", "Contraindicated medication combination"),
            "interaction": ("medication_conflict", "moderate", "Drug interaction identified"),
            "conflict": ("medication_conflict", "moderate", "Medication conflict noted"),
            "risk": ("risk_flag", "moderate", "Clinical risk factor identified"),
            "caution": ("risk_flag", "moderate", "Clinical caution advised"),
            "no .{0,20}concern": ("no_conflict", "low", "No significant concerns identified"),
            "no .{0,20}conflict": ("no_conflict", "low", "No conflicts identified"),
            "no .{0,20}issue": ("no_conflict", "low", "No issues identified"),
        }

        import re
        matched_any = False
        for keyword, (ftype, severity, desc) in risk_keywords.items():
            if re.search(keyword, text_lower):
                if ftype != "no_conflict":
                    matched_any = True
                findings.append({"type": ftype, "severity": severity, "description": desc})
                if ftype != "no_conflict":
                    risk_flags.append(ftype)

        if not matched_any:
            findings.append({"type": "no_conflict", "severity": "low",
                             "description": "No significant conflicts detected"})

        return {
            "findings": findings,
            "medications_reported": [],
            "conditions_reported": [],
            "risk_flags": list(set(risk_flags)),
            "summary": text[:300],
        }

    def _get_shared_fields(self, response: AgentResponse, tier: int) -> list[str]:
        """Track what data fields were shared, with counts, for audit purposes."""
        fields = []
        if response.medications_reported:
            fields.append(f"medications:{len(response.medications_reported)}")
        if response.conditions_reported:
            fields.append(f"conditions:{len(response.conditions_reported)}")
        if response.lab_results_reported:
            fields.append(f"lab_results:{len(response.lab_results_reported)}")
        if response.risk_flags:
            fields.append(f"risk_flags:{len(response.risk_flags)}")
        if response.findings:
            fields.append(f"findings:{len(response.findings)}")
        if response.summary:
            fields.append("summary")
        return fields


class OrchestratorAgent:
    """
    Central routing agent. Does NOT access patient data.
    Routes queries to relevant departments and synthesizes responses.
    """

    def __init__(self, department_agents: dict[str, DepartmentAgent],
                 llm: BaseLLMProvider, consent_manager: ConsentManager,
                 audit: AuditTrail, dept_config: dict,
                 routing_mode: str = "relevance",
                 synthesis_mode: str = "hybrid",
                 enforce_consent: bool = True,
                 validate_tokens: bool = False,
                 query_budget: int = 0,
                 corroborate_critical: bool = False):
        self.agents = department_agents
        self.llm = llm
        self.consent = consent_manager
        self.audit = audit
        self.dept_config = dept_config

        # ── Ablation / hardening switches (R1) ──
        # routing_mode:
        #   "relevance" - the MedAgentNet orchestrator (clinical relevance map)
        #   "broadcast" - query every department, no relevance filtering
        #   "local"     - query nothing; the requesting agent answers alone
        # enforce_consent=False    -> consent service removed from the path
        # validate_tokens=True     -> consent tokens checked (replay mitigation)
        # query_budget>0           -> per-patient per-requester query cap
        #                             (differencing-attack mitigation)
        # corroborate_critical     -> a critical alert must be supported by more
        #                             than one department (fabrication mitigation)
        # synthesis_mode:
        #   "none"   - relay each department's own findings only. This is the R0
        #              behaviour: no cross-departmental reasoning happens, so a
        #              conflict whose limbs sit in different departments cannot
        #              be found unless the query itself carries them.
        #   "rules"  - assemble the disclosed (tier-limited) evidence from all
        #              responding departments and evaluate it against the
        #              grounded interaction knowledge base.
        #   "llm"    - assemble the same evidence and ask the language model to
        #              reason over it.
        #   "hybrid" - both; the default configuration of MedAgentNet.
        self.routing_mode = routing_mode
        self.synthesis_mode = synthesis_mode
        self.enforce_consent = enforce_consent
        self.validate_tokens = validate_tokens
        self.query_budget = query_budget
        self.corroborate_critical = corroborate_critical

        self._budget_used: dict[tuple, int] = {}
        self.rejected_tokens = 0
        self.budget_blocks = 0
        self.suppressed_uncorroborated = 0

    def reset_counters(self):
        self._budget_used.clear()
        self.rejected_tokens = 0
        self.budget_blocks = 0
        self.suppressed_uncorroborated = 0

    def process_request(self, requesting_dept: str, patient_id: str,
                         clinical_context: dict,
                         query_type: str = "MED_CONFLICT",
                         is_emergency: bool = False,
                         force_disclosure_tier: int = None) -> dict:
        """
        Main entry point: process a cross-departmental query.

        Returns dict with:
        - alerts: list of ConflictAlert
        - responses: list of AgentResponse
        - privacy_report: what was shared
        """
        # 1. Determine target departments
        targets = self._determine_targets(
            requesting_dept, clinical_context, query_type, is_emergency
        )

        logger.info(
            f"[Orchestrator] {requesting_dept} querying about patient {patient_id}. "
            f"Targets: {targets}"
        )

        # 2. Send queries to each target and collect responses
        responses = []
        denied = 0
        unreachable = []
        for target_dept in targets:
            # In "local" mode the requesting department's own agent answers, so
            # the self-query is the whole point and must not be skipped.
            if target_dept == requesting_dept and self.routing_mode != "local":
                continue
            if target_dept not in self.agents:
                continue

            # Per-patient query budget: a mitigation against extracting a record
            # by repeated low-tier differencing queries.
            if self.query_budget:
                key = (patient_id, requesting_dept, target_dept)
                if self._budget_used.get(key, 0) >= self.query_budget:
                    self.budget_blocks += 1
                    self.audit.log(
                        event_type="query_budget_exceeded",
                        source_agent=requesting_dept, target_agent=target_dept,
                        patient_id=patient_id, query_type=query_type,
                        consent_granted=False, disclosure_tier=0,
                        outcome="blocked_by_query_budget",
                    )
                    continue
                self._budget_used[key] = self._budget_used.get(key, 0) + 1

            # Check consent (or use forced tier for experiments)
            if not self.enforce_consent:
                # Ablation: consent service removed entirely.
                allowed = True
                max_tier = force_disclosure_tier if force_disclosure_tier is not None else 3
            elif force_disclosure_tier is not None:
                # Forced-tier experiments still honour a denial, so that tier
                # sweeps and consent sweeps compose correctly. R0 bypassed the
                # consent check whenever a tier was forced.
                allowed, _ = self.consent.check_consent(
                    patient_id, requesting_dept, target_dept,
                    requested_tier=force_disclosure_tier,
                    is_emergency=is_emergency,
                )
                max_tier = force_disclosure_tier
            else:
                allowed, max_tier = self.consent.check_consent(
                    patient_id, requesting_dept, target_dept,
                    requested_tier=2 if not is_emergency else 3,
                    is_emergency=is_emergency,
                )

            self.audit.log(
                event_type="consent_check",
                source_agent=requesting_dept,
                target_agent=target_dept,
                patient_id=patient_id,
                query_type=query_type,
                consent_granted=allowed,
                disclosure_tier=max_tier if allowed else 0,
            )

            if not allowed:
                denied += 1
                logger.info(f"  Consent denied: {requesting_dept} -> {target_dept}")
                continue

            # Build and send query
            query = ClinicalQuery(
                source_agent=requesting_dept,
                target_agent=target_dept,
                patient_id=patient_id,
                query_type=query_type,
                clinical_context=clinical_context,
                disclosure_tier=max_tier,
                consent_token=self.consent.generate_consent_token(
                    patient_id, requesting_dept, target_dept, max_tier
                ),
                priority="critical" if is_emergency else "high",
            )

            # Token validation (off by default in R0; a replay attack therefore
            # succeeded). When enabled, the responding side checks that the
            # token is bound to this patient, this department pair and this tier,
            # is unexpired, and has not been seen before.
            if self.validate_tokens:
                ok, reason = self.consent.validate_consent_token(
                    query.consent_token, patient_id, requesting_dept,
                    target_dept, max_tier,
                )
                if not ok:
                    self.rejected_tokens += 1
                    self.audit.log(
                        event_type="token_rejected",
                        source_agent=requesting_dept, target_agent=target_dept,
                        patient_id=patient_id, query_type=query_type,
                        consent_granted=False, disclosure_tier=0, outcome=reason,
                    )
                    continue

            self.audit.log(
                event_type="query_sent",
                source_agent=requesting_dept,
                target_agent=target_dept,
                patient_id=patient_id,
                query_type=query_type,
                disclosure_tier=max_tier,
            )

            # Fault tolerance: an unreachable or failing department must not
            # abort the whole request. The exchange continues with the
            # departments that did answer, and the shortfall is reported so the
            # clinician knows the assessment is partial.
            try:
                responses.append(self.agents[target_dept].process_query(query))
            except Exception as e:
                unreachable.append(target_dept)
                logger.warning(f"  {target_dept} unreachable: "
                               f"{type(e).__name__}: {e}")
                self.audit.log(
                    event_type="agent_unreachable", source_agent=requesting_dept,
                    target_agent=target_dept, patient_id=patient_id,
                    query_type=query_type, disclosure_tier=max_tier,
                    outcome=f"{type(e).__name__}",
                )

        # 2b. Include the requesting department's own record.
        # The clinician issuing the query already holds their own department's
        # data, so it forms part of the picture to be reasoned over. No consent
        # check applies (no boundary is crossed) and the response is excluded
        # from the disclosure ledger for the same reason. R0 skipped this, which
        # meant a two-drug conflict with one limb in the requesting department
        # could never be assembled.
        local_responses = []
        if self.routing_mode != "local" and requesting_dept in self.agents:
            local_q = ClinicalQuery(
                source_agent=requesting_dept, target_agent=requesting_dept,
                patient_id=patient_id, query_type=query_type,
                clinical_context=clinical_context,
                disclosure_tier=DisclosureTier.FULL_CONTEXT,
                priority="critical" if is_emergency else "high",
            )
            n_before = len(self.agents[requesting_dept].disclosure_log)
            try:
                local_responses.append(
                    self.agents[requesting_dept].process_query(local_q))
            except Exception as e:
                unreachable.append(requesting_dept)
                logger.warning(f"  local agent {requesting_dept} unavailable: {e}")
            del self.agents[requesting_dept].disclosure_log[n_before:]

        # 3. Synthesize responses and generate alerts
        alerts = self._synthesize_responses(
            requesting_dept, patient_id, clinical_context,
            responses + local_responses, is_emergency,
            query_type=query_type,
        )

        # 4. Build privacy report
        privacy_report = {
            "departments_queried": targets,
            "responses_received": len(responses),
            "consent_denied": denied,
            "unreachable_departments": unreachable,
            "coverage": round(
                len(responses) / max(1, len(targets) - (1 if requesting_dept in targets else 0)),
                3),
            "max_tier_used": int(max((r.disclosure_tier for r in responses), default=0)),
            "data_shared_summary": {
                r.source_agent: self._summarize_shared(r) for r in responses
            },
            "disclosed_text": {
                r.source_agent: DepartmentAgent._disclosed_text(r) for r in responses
            },
        }

        return {
            "alerts": alerts,
            "responses": responses,
            "privacy_report": privacy_report,
        }

    def _determine_targets(self, requesting_dept: str, context: dict,
                            query_type: str, is_emergency: bool) -> list[str]:
        """Determine which departments to query based on clinical context."""
        # ── Ablation modes ──
        if self.routing_mode == "local":
            # No cross-departmental exchange: the requesting agent answers alone.
            return [requesting_dept]
        if self.routing_mode == "broadcast":
            # No relevance filtering: ask everyone, always.
            return sorted(self.agents.keys())

        if is_emergency or query_type == "EMERGENCY_BROADCAST":
            return list(self.agents.keys())

        context_str = json.dumps(context).lower()

        # ── Check if the doctor explicitly requested specific departments ──
        # Parse only the doctor's free-text input fields (planned_procedure,
        # query_reason) — NOT the full context which contains medication and
        # condition data with department names embedded in keys/values.
        doctor_text_parts = []
        for key in ("planned_procedure", "query_reason", "clinical_notes"):
            val = context.get(key, "")
            if val:
                doctor_text_parts.append(str(val).lower())
        doctor_text = " ".join(doctor_text_parts)

        explicit_targets = self._parse_explicit_targets(doctor_text)
        if explicit_targets:
            logger.info(f"[Orchestrator] Doctor explicitly requested: {explicit_targets}")
            return sorted(list(explicit_targets))

        # ── Default rule-based routing ──
        targets = set()

        # Always include laboratory for routine checks
        if "laboratory" in self.agents:
            targets.add("laboratory")

        # Rule-based routing
        for dept_id, dept_data in self.dept_config.items():
            if dept_id == requesting_dept:
                continue

            # Check if department's keywords match the clinical context
            keywords = dept_data.get("specialization_keywords", [])
            for keyword in keywords:
                if keyword.lower() in context_str:
                    targets.add(dept_id)
                    break

            # Check procedure relevance
            procedures = dept_data.get("relevant_for_procedures", [])
            for proc in procedures:
                if proc.lower() in context_str or proc == "any":
                    targets.add(dept_id)
                    break

        # For medication conflicts, query all departments that prescribe
        if query_type in ("MED_CONFLICT", "PROC_RISK"):
            for dept_id, dept_data in self.dept_config.items():
                if dept_data.get("common_medications"):
                    targets.add(dept_id)

        # A longitudinal or cross-system pattern query is cross-departmental by
        # definition: the point is to find a connection nobody has made yet, so
        # relevance cannot be judged in advance from the requesting department's
        # context. R0 routed these through the same keyword map as everything
        # else, which meant a pattern query typically reached only the
        # laboratory.
        if query_type in ("LONG_PATTERN", "VITAL_TREND", "REFERRAL_HX"):
            targets.update(self.agents.keys())

        return sorted(list(targets))

    def _parse_explicit_targets(self, context_str: str) -> set[str] | None:
        """
        Parse the clinical context to see if the doctor explicitly named
        departments to query. Returns the set of matching department IDs,
        or None if no explicit instruction was detected.

        Recognises patterns like:
          "only ask laboratory"
          "only cardiology and dental"
          "query laboratory only"
          "ask only the endocrinology department"
          "check with cardiology"
          "consult nephrology and pulmonology"
        """
        import re

        # Trigger words that signal the doctor is specifying departments
        restrict_patterns = [
            r'\bonly\b',       # "only ask ...", "... only"
            r'\bjust\b',       # "just ask ..."
            r'\bask\b',        # "ask laboratory"
            r'\bquery\b',      # "query cardiology"
            r'\bconsult\b',    # "consult nephrology"
            r'\bcheck with\b', # "check with dental"
        ]

        has_directive = any(re.search(p, context_str) for p in restrict_patterns)
        if not has_directive:
            return None

        # Build a lookup: all possible names/aliases -> department ID
        name_to_id: dict[str, str] = {}
        for dept_id, dept_data in self.dept_config.items():
            # Map the department ID itself: "cardiology", "general_practice"
            name_to_id[dept_id.lower()] = dept_id
            # Map the display name: "Cardiology", "General Practice"
            display = dept_data.get("name", "").lower()
            if display:
                name_to_id[display] = dept_id
            # Map underscore-free variant: "general practice"
            name_to_id[dept_id.replace("_", " ").lower()] = dept_id
            # Map common short forms
            short_forms = {
                "cardiology": ["cardio", "heart"],
                "dental": ["dentistry", "dentist"],
                "general_practice": ["gp", "primary care", "general"],
                "endocrinology": ["endo", "diabetes"],
                "ophthalmology": ["eye", "ophthal", "vision"],
                "nephrology": ["nephro", "kidney", "renal"],
                "neurology": ["neuro", "brain"],
                "rheumatology": ["rheumat", "joints"],
                "pulmonology": ["pulmo", "lung", "respiratory"],
                "laboratory": ["lab", "labs", "pathology"],
            }
            for alias in short_forms.get(dept_id, []):
                name_to_id[alias] = dept_id

        # Find all department mentions in the context string
        # Sort by length descending so "general practice" matches before "general"
        matched = set()
        for name in sorted(name_to_id.keys(), key=len, reverse=True):
            if re.search(r'\b' + re.escape(name) + r'\b', context_str):
                matched.add(name_to_id[name])

        return matched if matched else None

    # ── Cross-departmental synthesis ─────────────────────────────────────

    def _assemble_evidence(self, context: dict, responses: list[AgentResponse]):
        """Build the shared clinical picture from what departments disclosed.

        Only information that actually crossed a department boundary, at the
        tier the consent service permitted, enters this structure. The
        orchestrator never reads a patient record.
        """
        from protocol.interactions import ClinicalEvidence

        ev = ClinicalEvidence()
        ev.procedure = str(context.get("planned_procedure", "")).lower()

        for r in responses:
            dept = r.source_agent
            for m in r.medications_reported:
                if isinstance(m, dict):
                    ev.add_drug(m.get("name", ""), m.get("category", ""), dept)
                    if not m.get("name") and m.get("category"):
                        ev.categories.add(str(m["category"]).lower())
                elif isinstance(m, str):
                    ev.add_drug(m, "", dept)
            for c in r.conditions_reported:
                if isinstance(c, dict):
                    ev.add_condition(c.get("name", ""), dept)
                elif isinstance(c, str):
                    ev.add_condition(c, dept)
            for l in r.lab_results_reported:
                if isinstance(l, dict):
                    ev.add_lab(l.get("test_name", ""), l.get("value"),
                               l.get("date", ""), dept)
        return ev

    def _grounded_alerts(self, patient_id: str, ev, query_type: str,
                          is_emergency: bool) -> list[ConflictAlert]:
        """Evaluate assembled evidence against the interaction knowledge base."""
        from protocol.interactions import evaluate_rules, evaluate_patterns

        hits = evaluate_rules(ev)
        if query_type in ("LONG_PATTERN", "VITAL_TREND"):
            hits = hits + evaluate_patterns(ev)
        else:
            hits = hits + [h for h in evaluate_patterns(ev)
                           if h["severity"] == "critical"]

        alerts = []
        for h in hits:
            level = (AlertLevel.CRITICAL.value if h["severity"] == "critical"
                     else AlertLevel.HIGH_RISK.value)
            alerts.append(ConflictAlert(
                patient_id=patient_id,
                alert_level=level,
                alert_type="cross_department_interaction"
                if h in hits[:len(hits)] and query_type != "LONG_PATTERN"
                else "cross_department_pattern",
                description=f"{h['label']}: {h['mechanism']}",
                involved_departments=h["involved_departments"],
                involved_medications=h["involved_medications"],
                recommendation=self._generate_recommendation(
                    {"severity": h["severity"], "type": "medication_conflict"},
                    is_emergency),
            ))
        return alerts

    def _llm_synthesis_alerts(self, patient_id: str, context: dict,
                               responses: list[AgentResponse],
                               is_emergency: bool) -> list[ConflictAlert]:
        """Ask the language model to reason over the disclosed evidence."""
        if not responses:
            return []
        lines = [
            "You are combining answers from several hospital departments about "
            "one patient. Each department sees only its own records; you see "
            "only what each has disclosed below.",
            "",
            f"ENCOUNTER: {context.get('planned_procedure', 'clinical review')}",
            f"REQUEST: {context.get('query_reason', '')}",
            "",
            "DISCLOSED EVIDENCE:",
        ]
        for r in responses:
            meds = ", ".join(
                (m.get("name") or m.get("category", "")) if isinstance(m, dict) else str(m)
                for m in r.medications_reported) or "none reported"
            conds = ", ".join(
                (c.get("name", "") if isinstance(c, dict) else str(c))
                for c in r.conditions_reported) or "none reported"
            labs = "; ".join(
                f"{l.get('test_name')} {l.get('value')} ({l.get('date','')})"
                for l in r.lab_results_reported if isinstance(l, dict)) or "none reported"
            lines.append(f"- {r.source_agent}: medications: {meds}. "
                         f"conditions: {conds}. results: {labs}. "
                         f"note: {r.summary[:200]}")
        lines += [
            "",
            "Identify only interactions or patterns that require evidence from "
            "MORE THAN ONE department. Do not report a finding that a single "
            "department could already see on its own. If nothing crosses "
            "departments, return an empty list.",
            "",
            'Return JSON: {"alerts": [{"severity": "critical|high|moderate", '
            '"description": "...", "departments": [...], "medications": [...]}]}',
        ]
        try:
            raw = self.llm.generate(get_orchestrator_system_prompt(), "\n".join(lines))
        except Exception as e:
            logger.warning(f"Synthesis LLM error: {e}")
            return []

        data = None
        for attempt in (raw, raw[raw.find("{"):raw.rfind("}") + 1] if "{" in raw else ""):
            try:
                data = json.loads(attempt)
                break
            except Exception:
                continue
        if not isinstance(data, dict):
            return []

        # Accept either the alert schema requested above or the department
        # finding schema, so any backend able to answer one of the two can be
        # used for synthesis without a bespoke adapter.
        if not data.get("alerts") and data.get("findings"):
            data = {"alerts": [
                {"severity": f.get("severity", "moderate"),
                 "description": f.get("description", ""),
                 "medications": [
                     (m.get("name") if isinstance(m, dict) else str(m))
                     for m in data.get("medications_reported", []) or []],
                 "departments": sorted({r.source_agent for r in responses})}
                for f in data["findings"]
                if isinstance(f, dict)
                and f.get("type") not in ("no_conflict", "parse_error",
                                          "llm_error", "format_error")
            ]}

        alerts = []
        for a in data.get("alerts", []) or []:
            if not isinstance(a, dict):
                continue
            sev = str(a.get("severity", "moderate")).lower()
            level = {"critical": AlertLevel.CRITICAL.value,
                     "high": AlertLevel.HIGH_RISK.value,
                     "high_risk": AlertLevel.HIGH_RISK.value,
                     "moderate": AlertLevel.WARNING.value}.get(
                         sev, AlertLevel.WARNING.value)
            alerts.append(ConflictAlert(
                patient_id=patient_id,
                alert_level=level,
                alert_type="cross_department_synthesis",
                description=str(a.get("description", ""))[:400],
                involved_departments=[str(d) for d in (a.get("departments") or [])],
                involved_medications=[str(m) for m in (a.get("medications") or [])],
                recommendation=self._generate_recommendation(
                    {"severity": sev, "type": "medication_conflict"}, is_emergency),
            ))
        return alerts

    def _synthesize_responses(self, requesting_dept: str, patient_id: str,
                                context: dict, responses: list[AgentResponse],
                                is_emergency: bool,
                                query_type: str = "MED_CONFLICT") -> list[ConflictAlert]:
        """Synthesize all department responses into unified alerts."""
        alerts = []

        # Build a map: for each response, track its findings and medications together
        # so we can associate the right medications with each finding.
        seen_descriptions = set()

        for resp in responses:
            # Medications reported by THIS department agent
            # Normalize: LLMs may return strings or dicts
            resp_med_names = []
            for m in resp.medications_reported:
                if isinstance(m, dict):
                    name = m.get("name", "")
                elif isinstance(m, str):
                    name = m
                else:
                    name = str(m)
                if name and name not in resp_med_names:
                    resp_med_names.append(name)

            for finding in resp.findings:
                if finding.get("type") == "no_conflict":
                    continue

                # Deduplicate findings by description so the same alert
                # doesn't appear multiple times from different departments
                desc = finding.get("description", "")
                if desc in seen_descriptions:
                    # Already created an alert for this finding — just
                    # add the current department to the existing alert
                    for existing in alerts:
                        if existing.description == desc:
                            if resp.source_agent not in existing.involved_departments:
                                existing.involved_departments.append(resp.source_agent)
                            for mn in resp_med_names:
                                if mn not in existing.involved_medications:
                                    existing.involved_medications.append(mn)
                    continue

                seen_descriptions.add(desc)

                severity = finding.get("severity", "low")
                alert_level = {
                    "critical": AlertLevel.CRITICAL.value,
                    "high": AlertLevel.HIGH_RISK.value,
                    "moderate": AlertLevel.WARNING.value,
                    "low": AlertLevel.INFO.value,
                }.get(severity, AlertLevel.INFO.value)

                # Extract medication names mentioned in this finding's description
                # to include only the relevant ones
                desc_lower = desc.lower()
                finding_meds = []
                for mn in resp_med_names:
                    if mn.lower() in desc_lower:
                        finding_meds.append(mn)
                # If no specific meds matched, include all meds from this response
                if not finding_meds:
                    finding_meds = resp_med_names

                alert = ConflictAlert(
                    patient_id=patient_id,
                    alert_level=alert_level,
                    alert_type=finding.get("type", "unknown"),
                    description=desc,
                    involved_departments=[resp.source_agent],
                    involved_medications=finding_meds,
                    recommendation=self._generate_recommendation(finding, is_emergency),
                )
                alerts.append(alert)

                self.audit.log(
                    event_type="alert_raised",
                    source_agent="orchestrator",
                    target_agent=requesting_dept,
                    patient_id=patient_id,
                    # R0 logged the finding type in the query_type field and
                    # omitted the tier, which contaminated both the query-type
                    # and tier distributions in the privacy report.
                    query_type="",
                    disclosure_tier=int(resp.disclosure_tier),
                    outcome=f"Alert: {alert_level} - {finding.get('type', '')}",
                )

        # ── Cross-departmental layer ──
        # Everything above only relays what each department said about itself.
        # The step below is the one that combines them.
        if self.synthesis_mode in ("rules", "hybrid"):
            ev = self._assemble_evidence(context, responses)
            for a in self._grounded_alerts(patient_id, ev, query_type, is_emergency):
                if a.description not in seen_descriptions:
                    seen_descriptions.add(a.description)
                    alerts.append(a)
                    self.audit.log(
                        event_type="alert_raised", source_agent="orchestrator",
                        target_agent=requesting_dept, patient_id=patient_id,
                        disclosure_tier=int(max((r.disclosure_tier for r in responses),
                                                default=0)),
                        outcome=f"Cross-department alert: {a.alert_level}",
                    )

        if self.synthesis_mode in ("llm", "hybrid"):
            for a in self._llm_synthesis_alerts(patient_id, context, responses,
                                                is_emergency):
                if a.description and a.description not in seen_descriptions:
                    seen_descriptions.add(a.description)
                    alerts.append(a)

        # Corroboration requirement: a critical alert supported by exactly one
        # department is downgraded. This is the mitigation evaluated against the
        # fabricated-response and compromised-agent threats.
        if self.corroborate_critical:
            for a in alerts:
                if a.alert_level == AlertLevel.CRITICAL.value and \
                        len(a.involved_departments) < 2:
                    a.alert_level = AlertLevel.HIGH_RISK.value
                    a.recommendation = (
                        "Single-source critical finding, not corroborated by a "
                        "second department. " + a.recommendation
                    )
                    self.suppressed_uncorroborated += 1

        return alerts

    def _generate_recommendation(self, finding: dict, is_emergency: bool) -> str:
        """Generate actionable recommendation based on finding."""
        severity = finding.get("severity", "low")
        ftype = finding.get("type", "")

        if severity == "critical":
            if is_emergency:
                return "IMMEDIATE ACTION REQUIRED. Discontinue conflicting agents and consult specialists."
            return "URGENT: Contact relevant specialist before proceeding. Do not administer conflicting medications."

        if severity == "high":
            if "medication_conflict" in ftype:
                return "Review and resolve medication conflict before treatment. Consider specialist consultation."
            if "pattern" in ftype:
                return "Cross-departmental pattern detected. Order confirmatory tests and specialist referral."

        return "Monitor and document. Consider follow-up if clinically indicated."

    def _summarize_shared(self, response: AgentResponse) -> dict:
        """Summarize what data was shared for privacy reporting."""
        return {
            "tier": response.disclosure_tier,
            "medications_shared": len(response.medications_reported),
            "conditions_shared": len(response.conditions_reported),
            "risk_flags_raised": len(response.risk_flags),
            "findings_reported": len(response.findings),
        }
