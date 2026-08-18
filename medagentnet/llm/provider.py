"""
MedAgentNet - LLM Provider Abstraction
Supports: Mock (rule-based), Ollama, HuggingFace, OpenAI-compatible APIs.
Switch providers by changing config/settings.yaml -> llm.provider
"""
import json
import logging
from abc import ABC, abstractmethod
from typing import Optional

logger = logging.getLogger("medagentnet.llm")


class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers."""

    # Instrumentation shared by every backend, used by the multi-model
    # comparison table (calls, latency, approximate token volume, and how often
    # the backend failed to honour the JSON contract).
    def _init_stats(self):
        self.n_calls = 0
        self.total_latency_s = 0.0
        self.prompt_chars = 0
        self.completion_chars = 0
        self.format_failures = 0
        self.errors = 0

    @property
    def stats(self) -> dict:
        if not hasattr(self, "n_calls"):
            self._init_stats()
        # ~4 characters per token is the usual rule of thumb for English text.
        return {
            "provider": self.describe(),
            "calls": self.n_calls,
            "total_latency_s": round(self.total_latency_s, 3),
            "mean_latency_s": round(self.total_latency_s / self.n_calls, 4)
            if self.n_calls else 0.0,
            "approx_prompt_tokens": int(self.prompt_chars / 4),
            "approx_completion_tokens": int(self.completion_chars / 4),
            "format_failures": self.format_failures,
            "format_failure_rate": round(self.format_failures / self.n_calls, 4)
            if self.n_calls else 0.0,
            "errors": self.errors,
            **({"truncation_suspected": self.truncation_suspected,
                "max_prompt_tokens_seen": self.max_prompt_chars_seen // 4,
                "num_ctx": self.num_ctx}
               if hasattr(self, "num_ctx") else {}),
        }

    def reset_stats(self):
        self._init_stats()

    def describe(self) -> str:
        """Human-readable identity of the backend actually being used."""
        return self.__class__.__name__

    def _record(self, system_prompt: str, user_prompt: str,
                completion: str, elapsed: float):
        if not hasattr(self, "n_calls"):
            self._init_stats()
        self.n_calls += 1
        self.total_latency_s += elapsed
        self.prompt_chars += len(system_prompt) + len(user_prompt)
        self.completion_chars += len(completion or "")
        stripped = (completion or "").strip()
        looks_json = stripped.startswith("{") or "```" in stripped or "{" in stripped
        if not looks_json:
            self.format_failures += 1

    @abstractmethod
    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """Generate a response given system and user prompts."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if the provider is available."""
        pass


class MockLLMProvider(BaseLLMProvider):
    """
    Rule-based mock LLM for testing without GPU/API.
    Uses deterministic clinical rules to simulate agent reasoning.
    This allows full system testing without any external dependencies.
    """

    def is_available(self) -> bool:
        return True

    def describe(self) -> str:
        return "Mock (rule-based)"

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """Parse the structured prompt and apply clinical rules."""
        import time as _time
        t0 = _time.time()
        try:
            response = self._apply_clinical_rules(system_prompt, user_prompt)
        except Exception as e:
            logger.warning(f"MockLLM error: {e}")
            if not hasattr(self, "n_calls"):
                self._init_stats()
            self.errors += 1
            response = json.dumps({
                "findings": [],
                "risk_flags": [],
                "summary": "Unable to process query."
            })
        self._record(system_prompt, user_prompt, response, _time.time() - t0)
        return response

    # ── Prompt reading helpers (R1) ──────────────────────────────────────
    # R0 matched keywords against the concatenated system+user prompt. Two
    # consequences: the JSON *key* "planned_procedure" made the substring
    # "procedure" true for every query, so the warfarin bleeding rule fired on
    # routine checkups; and discontinued medications were indistinguishable
    # from active ones. Both are corrected by reading the rendered record.

    @staticmethod
    def _active_lines(user_prompt: str) -> str:
        """Only the lines describing currently-active clinical facts."""
        keep = []
        for line in user_prompt.splitlines():
            if "[DISCONTINUED]" in line or "[RESOLVED]" in line:
                continue
            keep.append(line)
        return "\n".join(keep).lower()

    @staticmethod
    def _planned_procedure(user_prompt: str) -> str:
        """Extract the clinician-declared procedure from the context block."""
        try:
            start = user_prompt.index("CLINICAL CONTEXT:") + len("CLINICAL CONTEXT:")
            brace = user_prompt.index("{", start)
            depth, end = 0, brace
            for i, ch in enumerate(user_prompt[brace:], start=brace):
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        end = i + 1
                        break
            ctx = json.loads(user_prompt[brace:end])
            return str(ctx.get("planned_procedure", "")).lower()
        except Exception:
            return ""

    def _apply_clinical_rules(self, system_prompt: str, user_prompt: str) -> str:
        """Apply deterministic clinical rules based on prompt content."""
        prompt_lower = self._active_lines(user_prompt)
        procedure = self._planned_procedure(user_prompt)
        invasive = any(w in procedure for w in
                       ["extraction", "surgery", "surgical", "biopsy", "implant",
                        "excision", "invasive"])

        findings = []
        risk_flags = []
        medications_found = []
        conditions_found = []

        # ── Medication conflict detection rules ──

        # Warfarin + invasive procedure
        if "warfarin" in prompt_lower:
            medications_found.append({
                "name": "Warfarin", "category": "anticoagulant",
                "relevance": "Active anticoagulant therapy"
            })
            if invasive:
                risk_flags.append("bleeding_risk")
                findings.append({
                    "type": "medication_conflict",
                    "severity": "high",
                    "description": "Patient on Warfarin - bleeding risk for planned procedure. "
                                   "Recommend INR check and possible bridging protocol."
                })

        # Triple whammy detection
        has_ace = any(w in prompt_lower for w in ["lisinopril", "ace_inhibitor", "enalapril"])
        has_nsaid = any(w in prompt_lower for w in ["ibuprofen", "nsaid", "naproxen"])
        has_diuretic = any(w in prompt_lower for w in ["hydrochlorothiazide", "diuretic", "furosemide"])

        if has_ace and has_nsaid:
            risk_flags.append("renal_risk")
            findings.append({
                "type": "medication_conflict",
                "severity": "high",
                "description": "ACE inhibitor + NSAID combination increases risk of renal impairment."
            })

        if has_ace and has_nsaid and has_diuretic:
            risk_flags.append("triple_whammy")
            findings.append({
                "type": "medication_conflict",
                "severity": "critical",
                "description": "CRITICAL: Triple whammy detected (ACE inhibitor + NSAID + Diuretic). "
                               "High risk of acute kidney injury. Immediate review required."
            })

        # Beta blocker + respiratory. Only a non-selective agent plus documented
        # airway disease is a contraindication; R0 fired on the cardioselective
        # agent and on the word "respiratory" appearing anywhere.
        if any(w in prompt_lower for w in ["metoprolol", "propranolol", "beta_blocker"]):
            medications_found.append({
                "name": "Propranolol" if "propranolol" in prompt_lower else "Metoprolol",
                "category": "beta_blocker",
                "relevance": "Beta-blocker therapy"
            })
            non_selective = "propranolol" in prompt_lower
            airway_disease = any(w in prompt_lower for w in
                                 ["asthma", "copd", "reactive airway",
                                  "chronic obstructive"])
            if non_selective and airway_disease:
                risk_flags.append("bronchospasm_risk")
                findings.append({
                    "type": "medication_conflict",
                    "severity": "high",
                    "description": "Beta-blocker use with reactive airway disease - bronchospasm risk."
                })

        # Metformin + renal impairment. R0 fired whenever the words "egfr" or
        # "renal" appeared, including on patients with a normal eGFR.
        if "metformin" in prompt_lower:
            medications_found.append({
                "name": "Metformin", "category": "biguanide",
                "relevance": "Diabetes medication"
            })
            impaired = self._renal_impairment(user_prompt)
            if impaired:
                risk_flags.append("lactic_acidosis_risk")
                findings.append({
                    "type": "medication_conflict",
                    "severity": "critical",
                    "description": "Metformin with impaired renal function - lactic acidosis risk. "
                                   "Hold metformin if eGFR < 30."
                })

        # Methotrexate + NSAIDs
        if "methotrexate" in prompt_lower and has_nsaid:
            risk_flags.append("methotrexate_toxicity")
            findings.append({
                "type": "medication_conflict",
                "severity": "high",
                "description": "NSAIDs reduce methotrexate clearance - increased toxicity risk."
            })

        # Carbamazepine + Warfarin
        if "carbamazepine" in prompt_lower and "warfarin" in prompt_lower:
            risk_flags.append("subtherapeutic_anticoagulation")
            findings.append({
                "type": "medication_conflict",
                "severity": "high",
                "description": "Carbamazepine induces warfarin metabolism - risk of subtherapeutic INR."
            })

        # Timolol + systemic beta blocker
        if "timolol" in prompt_lower and "metoprolol" in prompt_lower:
            risk_flags.append("additive_bradycardia")
            findings.append({
                "type": "medication_conflict",
                "severity": "high",
                "description": "Ophthalmic timolol + systemic beta-blocker - additive bradycardia risk."
            })

        # ── Pattern detection rules ──
        # R0 matched on literal planted values ("48", "58") and on the word
        # "rising", which the R0 scenario driver placed in the prompt itself.
        # Trends are now computed from the rendered laboratory series.

        series = self._lab_series(user_prompt)

        # Diabetes pattern: sustained glycaemic rise above the reference range
        gly = series.get("hba1c") or series.get("fasting glucose")
        if gly and len(gly) >= 3 and gly[-1] > gly[0] and self._any_out_of_range(
                user_prompt, ["hba1c", "fasting glucose"]):
            if any(w in prompt_lower for w in ["retinopathy", "retinal", "neuropathy"]):
                findings.append({
                    "type": "pattern_detected",
                    "severity": "high",
                    "description": "PATTERN: Rising glycaemic indices with retinal and "
                                   "peripheral nerve findings, suggestive of Type 2 "
                                   "Diabetes. Recommend formal workup."
                })

        # CKD progression: monotonic eGFR decline of clinical magnitude
        egfr = series.get("egfr")
        if egfr and len(egfr) >= 3 and (egfr[0] - egfr[-1]) >= 10 and egfr[-1] < 90:
            findings.append({
                "type": "pattern_detected",
                "severity": "high",
                "description": "PATTERN: Progressive eGFR decline indicates CKD "
                               "progression. Review nephrotoxic medications."
            })

        # Thyroid-cardiac connection: suppressed TSH with new AF
        tsh = series.get("tsh")
        if tsh and tsh[-1] < 0.4 and any(
                w in prompt_lower for w in ["atrial fibrillation", "hyperthyroid"]):
            findings.append({
                "type": "pattern_detected",
                "severity": "high",
                "description": "PATTERN: Suppressed TSH alongside new atrial "
                               "fibrillation, suggestive of thyrotoxic AF."
            })

        # ── Report this department's own relevant record ──
        # A department agent answering a safety or pattern query must return the
        # medications, conditions and results it holds, at the granularity the
        # disclosure tier permits. R0's mock reported only drugs it happened to
        # have a rule for, which meant the orchestrator received nothing to
        # combine; all cross-departmental "detection" came from the leaked
        # context instead.
        record = self._parse_record(user_prompt)
        tier = self._tier(user_prompt)

        if tier >= 2:
            for m in record["medications"]:
                if not m["active"]:
                    continue
                if not any(x.get("name") == m["name"] for x in medications_found):
                    medications_found.append({
                        "name": m["name"], "category": m["category"],
                        "relevance": "Active prescription in this department",
                    })
            for c in record["conditions"]:
                if c["active"]:
                    conditions_found.append({"name": c["name"], "code": c.get("code", "")})
        else:
            for m in record["medications"]:
                if m["active"] and m["category"]:
                    medications_found.append({"category": m["category"]})

        # At Tier 3 the narrative is visible, so a drug documented only in a note
        # (transferred care, external prescription) can be picked up. This is the
        # only route by which a note-only limb of a conflict can be recovered.
        if tier >= 3:
            for name in self._drugs_in_notes(user_prompt):
                if not any(x.get("name") == name for x in medications_found):
                    medications_found.append({
                        "name": name, "category": self._category_for(name),
                        "relevance": "Documented in narrative, not on the "
                                     "structured medication list",
                    })

        labs_found = []
        if tier >= 2:
            for name, entries in record["labs"].items():
                for date, value, flagged in entries:
                    labs_found.append({"test_name": name, "value": value,
                                       "date": date, "is_abnormal": flagged})

        # ── Build response ──

        if not findings:
            findings.append({
                "type": "no_conflict",
                "severity": "low",
                "description": "No significant conflicts or patterns detected for this query."
            })

        response = {
            "findings": findings,
            "medications_reported": medications_found,
            "conditions_reported": conditions_found,
            "lab_results_reported": labs_found,
            "risk_flags": risk_flags,
            "summary": self._build_summary(findings, risk_flags),
        }

        return json.dumps(response, indent=2)

    # ── Record readers ───────────────────────────────────────────────────

    _DRUG_LEXICON = {
        "warfarin": "anticoagulant", "ibuprofen": "nsaid", "naproxen": "nsaid",
        "lisinopril": "ace_inhibitor", "ramipril": "ace_inhibitor",
        "hydrochlorothiazide": "diuretic", "spironolactone":
            "potassium_sparing_diuretic", "metformin": "biguanide",
        "metoprolol": "beta_blocker", "propranolol": "beta_blocker",
        "timolol": "beta_blocker_ophthalmic", "methotrexate": "dmard",
        "carbamazepine": "anticonvulsant", "sertraline": "ssri",
        "simvastatin": "statin", "clarithromycin": "macrolide_antibiotic",
        "levothyroxine": "thyroid_hormone", "calcium carbonate":
            "mineral_supplement", "allopurinol": "xanthine_oxidase_inhibitor",
        "azathioprine": "immunosuppressant", "digoxin": "cardiac_glycoside",
        "amiodarone": "antiarrhythmic", "paracetamol": "analgesic",
        "aspirin": "antiplatelet", "prednisolone": "corticosteroid",
    }

    @classmethod
    def _category_for(cls, name: str) -> str:
        return cls._DRUG_LEXICON.get(name.lower(), "unknown")

    @classmethod
    def _drugs_in_notes(cls, user_prompt: str) -> list:
        """Medication names mentioned in the narrative but not on the list.

        A mention inside a negation or a discontinuation statement is ignored.
        """
        import re
        block = []
        capture = False
        for line in user_prompt.splitlines():
            if line.startswith("Clinical notes"):
                capture = True
                continue
            if capture:
                if line.strip().startswith("-"):
                    block.append(line.lower())
                elif line.strip() == "":
                    continue
                else:
                    break
        found = []
        stop_words = ("stopped", "discontinued", "withdrawn", "no longer",
                      "do not restart", "not on", "denies", "no history")
        for line in block:
            if any(sw in line for sw in stop_words):
                continue
            for drug in cls._DRUG_LEXICON:
                if re.search(r"\b" + re.escape(drug) + r"\b", line):
                    proper = drug.title()
                    if proper not in found:
                        found.append(proper)
        return found

    @staticmethod
    def _tier(user_prompt: str) -> int:
        import re
        m = re.search(r"DISCLOSURE TIER:\s*(\d)", user_prompt)
        return int(m.group(1)) if m else 2

    @staticmethod
    def _parse_record(user_prompt: str) -> dict:
        """Recover the structured record from the rendered prompt."""
        import re
        meds, conds = [], []
        labs: dict[str, list] = {}
        for line in user_prompt.splitlines():
            s = line.strip()

            # Category-only rendering (Tier 1)
            m = re.match(
                r"-\s*\[(ACTIVE|DISCONTINUED)\]\s*\[medication in category:\s*(.+?)\]", s)
            if m:
                meds.append({"name": "", "category": m.group(2).strip(),
                             "active": m.group(1) == "ACTIVE"})
                continue

            # Medication and condition lines share a shape; a condition line is
            # the one carrying "severity=".
            m = re.match(r"-\s*\[(ACTIVE|DISCONTINUED|RESOLVED)\]\s*(.+?)\s*\((.+?)\):(.*)$", s)
            if m:
                status, name, code, tail = (m.group(1), m.group(2).strip(),
                                            m.group(3).strip(), m.group(4))
                if "severity=" in tail:
                    conds.append({"name": name, "code": code,
                                  "active": status == "ACTIVE"})
                else:
                    meds.append({"name": name, "category": code,
                                 "active": status == "ACTIVE"})
                continue

            m = re.match(
                r"-\s*(\d{4}-\d{2}-\d{2}):\s*(.+?)\s*=\s*([-\d.]+)\S*(\s*\[OUT OF RANGE\])?",
                s)
            if m:
                labs.setdefault(m.group(2).strip(), []).append(
                    (m.group(1), float(m.group(3)), bool(m.group(4)))
                )

        return {"medications": meds, "conditions": conds, "labs": labs}

    # ── Laboratory readers ───────────────────────────────────────────────

    @staticmethod
    def _lab_series(user_prompt: str) -> dict:
        """Recover {test name -> [values in date order]} from the record block."""
        import re
        series: dict[str, list] = {}
        for line in user_prompt.splitlines():
            m = re.match(r"\s*-\s*\d{4}-\d{2}-\d{2}:\s*(.+?)\s*=\s*([-\d.]+)", line)
            if m:
                series.setdefault(m.group(1).strip().lower(), []).append(float(m.group(2)))
        return series

    @staticmethod
    def _any_out_of_range(user_prompt: str, tests: list) -> bool:
        low = user_prompt.lower()
        for line in low.splitlines():
            if "[out of range]" in line and any(t in line for t in tests):
                return True
        return False

    def _renal_impairment(self, user_prompt: str) -> bool:
        """True only if the record shows a genuinely reduced eGFR."""
        egfr = self._lab_series(user_prompt).get("egfr")
        if egfr and min(egfr) < 45:
            return True
        low = user_prompt.lower()
        return any(w in low for w in
                   ["chronic kidney disease stage 3", "chronic kidney disease stage 4",
                    "chronic kidney disease stage 5", "acute kidney injury",
                    "end-stage renal"]) and "resolved" not in low

    def _build_summary(self, findings: list, risk_flags: list) -> str:
        """Build a human-readable summary."""
        if not risk_flags:
            return "No significant clinical concerns identified."

        parts = []
        severity_order = {"critical": 0, "high": 1, "moderate": 2, "low": 3}
        sorted_findings = sorted(findings, key=lambda x: severity_order.get(x.get("severity", "low"), 3))

        for f in sorted_findings:
            if f["type"] != "no_conflict":
                parts.append(f"[{f['severity'].upper()}] {f['description']}")

        return " | ".join(parts) if parts else "No significant concerns."


class OllamaProvider(BaseLLMProvider):
    """Ollama local LLM provider - supports any model available in Ollama.
    Auto-detects API version (/api/chat vs /api/generate) for compatibility."""

    def __init__(self, base_url: str = "http://localhost:11434",
                 model: str = "llama3:8b-instruct",
                 temperature: float = 0.3, max_tokens: int = 1024,
                 num_ctx: int = 8192, request_timeout: int = 600):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        # Ollama defaults to a 2048-token context. A Tier-3 prompt carrying a
        # full departmental record with a longitudinal laboratory series and
        # clinical notes exceeds that, and Ollama truncates SILENTLY from the
        # start of the prompt -- which would remove the system prompt and the
        # clinical context while still returning a plausible-looking answer.
        # Set it explicitly. `truncation_suspected` counts prompts that come
        # close to the limit so a run can be checked afterwards.
        self.num_ctx = num_ctx
        self.request_timeout = request_timeout
        self.truncation_suspected = 0
        self.max_prompt_chars_seen = 0
        self._use_chat_api = None  # Auto-detect on first call
        self._available_models = []

    def _options(self) -> dict:
        return {
            "temperature": self.temperature,
            "num_predict": self.max_tokens,
            "num_ctx": self.num_ctx,
        }

    def _check_prompt_fits(self, system_prompt: str, user_prompt: str):
        """Warn if a prompt approaches the configured context window."""
        n = len(system_prompt) + len(user_prompt)
        self.max_prompt_chars_seen = max(self.max_prompt_chars_seen, n)
        # ~4 characters per token, and leave room for the completion.
        budget = (self.num_ctx - self.max_tokens) * 4
        if n > budget:
            self.truncation_suspected += 1
            logger.warning(
                f"prompt of ~{n // 4} tokens may not fit num_ctx={self.num_ctx} "
                f"with num_predict={self.max_tokens}; raise --num-ctx")

    def is_available(self) -> bool:
        try:
            import requests
            # Check if Ollama is running
            resp = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                self._available_models = [
                    m.get("name", "") for m in data.get("models", [])
                ]
                if self._available_models:
                    logger.info(f"Ollama available models: {self._available_models}")
                # Check if our target model is available
                model_base = self.model.split(":")[0]
                model_found = any(
                    model_base in m for m in self._available_models
                )
                if not model_found and self._available_models:
                    # R0 silently substituted the first model the server happened
                    # to have, so a results file could be labelled with a model
                    # that never ran. Refuse instead.
                    logger.error(
                        f"Model '{self.model}' is not present in Ollama. "
                        f"Available: {self._available_models}. "
                        f"Run: ollama pull {self.model}"
                    )
                    return False
                return True
            # Some Ollama versions respond at root /
            resp2 = requests.get(f"{self.base_url}/", timeout=5)
            return resp2.status_code == 200
        except Exception as e:
            logger.warning(f"Ollama connection failed: {e}")
            return False

    def _detect_api_version(self):
        """Auto-detect whether to use /api/chat or /api/generate."""
        if self._use_chat_api is not None:
            return
        import requests
        try:
            # Try /api/chat with a minimal request
            resp = requests.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": self.model,
                    "messages": [{"role": "user", "content": "hi"}],
                    "stream": False,
                    "options": {"num_predict": 1},
                },
                timeout=30,
            )
            if resp.status_code == 200:
                self._use_chat_api = True
                logger.info("Ollama: using /api/chat endpoint")
                return
        except Exception:
            pass

        # Fall back to /api/generate
        self._use_chat_api = False
        logger.info("Ollama: using /api/generate endpoint (older version detected)")

    def describe(self) -> str:
        return (f"Ollama ({self.model}, T={self.temperature}, "
                f"num_ctx={self.num_ctx})")

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        import requests
        import time as _time

        self._detect_api_version()
        self._check_prompt_fits(system_prompt, user_prompt)
        t0 = _time.time()

        try:
            if self._use_chat_api:
                out = self._generate_chat(system_prompt, user_prompt)
            else:
                out = self._generate_legacy(system_prompt, user_prompt)
            self._record(system_prompt, user_prompt, out, _time.time() - t0)
            return out
        except requests.exceptions.HTTPError as e:
            # If chat API fails with 404, retry with generate API
            if e.response is not None and e.response.status_code == 404:
                logger.warning("Chat API returned 404, switching to /api/generate")
                self._use_chat_api = False
                out = self._generate_legacy(system_prompt, user_prompt)
                self._record(system_prompt, user_prompt, out, _time.time() - t0)
                return out
            raise
        except requests.exceptions.ConnectionError as e:
            logger.error(f"Ollama connection error: {e}")
            logger.error("Make sure Ollama is running: ollama serve")
            raise
        except requests.exceptions.Timeout:
            logger.error("Ollama request timed out. Model may be loading.")
            raise

    def _generate_chat(self, system_prompt: str, user_prompt: str) -> str:
        """Use the newer /api/chat endpoint."""
        import requests
        resp = requests.post(
            f"{self.base_url}/api/chat",
            json={
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "stream": False,
                "options": self._options(),
            },
            timeout=self.request_timeout,
        )
        resp.raise_for_status()
        return resp.json()["message"]["content"]

    def _generate_legacy(self, system_prompt: str, user_prompt: str) -> str:
        """Use the older /api/generate endpoint (compatible with all Ollama versions)."""
        import requests
        # Combine system + user prompt for the generate endpoint
        combined_prompt = f"### System:\n{system_prompt}\n\n### User:\n{user_prompt}\n\n### Assistant:\n"
        resp = requests.post(
            f"{self.base_url}/api/generate",
            json={
                "model": self.model,
                "prompt": combined_prompt,
                "system": system_prompt,
                "stream": False,
                "options": self._options(),
            },
            timeout=self.request_timeout,
        )
        resp.raise_for_status()
        return resp.json().get("response", "")


class OpenAICompatibleProvider(BaseLLMProvider):
    """OpenAI-compatible API provider (vLLM, LMStudio, text-generation-webui, etc.)."""

    def __init__(self, base_url: str = "http://localhost:8000/v1",
                 model: str = "BioMistral/BioMistral-7B",
                 api_key: str = "not-needed",
                 temperature: float = 0.3, max_tokens: int = 1024):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens

    def is_available(self) -> bool:
        try:
            import requests
            resp = requests.get(f"{self.base_url}/models", timeout=5,
                                headers={"Authorization": f"Bearer {self.api_key}"})
            return resp.status_code == 200
        except Exception:
            return False

    def describe(self) -> str:
        return f"OpenAI-compatible ({self.model}, T={self.temperature})"

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        import time as _time
        t0 = _time.time()
        out = self._generate(system_prompt, user_prompt)
        self._record(system_prompt, user_prompt, out, _time.time() - t0)
        return out

    def _generate(self, system_prompt: str, user_prompt: str) -> str:
        import requests
        resp = requests.post(
            f"{self.base_url}/chat/completions",
            json={
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
            },
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]


class HuggingFaceProvider(BaseLLMProvider):
    """HuggingFace Transformers local provider with optional quantization."""

    def __init__(self, model_id: str = "BioMistral/BioMistral-7B",
                 device: str = "auto", quantization: str = "4bit",
                 max_tokens: int = 1024):
        self.model_id = model_id
        self.device = device
        self.quantization = quantization
        self.max_tokens = max_tokens
        self._pipeline = None

    def _load_model(self):
        if self._pipeline is not None:
            return
        from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
        import torch

        kwargs = {"device_map": self.device, "torch_dtype": torch.float16}
        if self.quantization == "4bit":
            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
        elif self.quantization == "8bit":
            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)

        model = AutoModelForCausalLM.from_pretrained(self.model_id, **kwargs)
        tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self._pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer)

    def is_available(self) -> bool:
        try:
            import transformers
            return True
        except ImportError:
            return False

    def describe(self) -> str:
        return f"HuggingFace ({self.model_id}, {self.quantization})"

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        import time as _time
        self._load_model()
        prompt = f"### System:\n{system_prompt}\n\n### User:\n{user_prompt}\n\n### Assistant:\n"
        t0 = _time.time()
        result = self._pipeline(
            prompt,
            max_new_tokens=self.max_tokens,
            do_sample=True,
            temperature=0.3,
            return_full_text=False,
        )
        out = result[0]["generated_text"]
        self._record(system_prompt, user_prompt, out, _time.time() - t0)
        return out


class ProviderUnavailable(RuntimeError):
    """Raised when a requested backend cannot be reached and strict mode is on."""


def create_llm_provider(config: dict, strict: bool = True) -> BaseLLMProvider:
    """Factory function to create the appropriate LLM provider from config.

    ``strict=True`` (the default from R1 onward) raises rather than silently
    substituting the rule-based Mock provider. In R0 an unreachable Ollama
    server produced a results file labelled with the LLM's name but containing
    Mock output; that failure mode is now impossible unless explicitly allowed.
    """
    provider_name = config.get("provider", "mock")

    def _unavailable(what: str):
        msg = (f"{what} was requested but is not reachable. Start the backend "
               f"or pass strict=False to fall back to the rule-based provider.")
        if strict:
            raise ProviderUnavailable(msg)
        logger.warning(msg + " Falling back to MockLLM.")
        return MockLLMProvider()

    if provider_name == "mock":
        logger.info("Using MockLLM provider (rule-based, no GPU required)")
        return MockLLMProvider()

    elif provider_name == "ollama":
        cfg = dict(config.get("ollama", {}))
        logger.info(f"Using Ollama provider: {cfg.get('model', 'llama3')}")
        provider = OllamaProvider(**cfg)
        if not provider.is_available():
            return _unavailable(f"Ollama model '{cfg.get('model')}'")
        return provider

    elif provider_name == "openai_compatible":
        cfg = config.get("openai_compatible", {})
        logger.info(f"Using OpenAI-compatible API: {cfg.get('base_url', '')}")
        provider = OpenAICompatibleProvider(**cfg)
        if not provider.is_available():
            return _unavailable(f"OpenAI-compatible endpoint {cfg.get('base_url','')}")
        return provider

    elif provider_name == "huggingface":
        cfg = config.get("huggingface", {})
        logger.info(f"Using HuggingFace provider: {cfg.get('model_id', '')}")
        provider = HuggingFaceProvider(**cfg)
        if not provider.is_available():
            return _unavailable(f"HuggingFace model {cfg.get('model_id','')}")
        return provider

    else:
        raise ValueError(f"Unknown LLM provider '{provider_name}'")
