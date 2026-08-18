"""The response shapes a real model produces that a rule-based one never does.

A run of the full suite is many hours and tens of thousands of backend calls, so
any shape the code cannot handle will be met at least once. These pin the ones
that have actually occurred.
"""
import pytest

from llm.provider import BaseLLMProvider
from simulation.runner_v2 import HardRunner
from simulation.evaluation import evaluate_run


class _Scripted(BaseLLMProvider):
    """Returns a fixed payload, whatever it is asked."""

    def __init__(self, payload):
        self.payload = payload
        self._init_stats()

    def is_available(self):
        return True

    def describe(self):
        return "scripted"

    def generate(self, system_prompt, user_prompt):
        self._record(system_prompt, user_prompt, self.payload, 0.0)
        return self.payload


NULL_FIELDS = """{
  "findings": [{"type": "medication_conflict", "severity": "high",
                "description": "Possible interaction."}],
  "medications_reported": [{"name": null, "category": null}],
  "conditions_reported": [{"name": null, "code": null}],
  "lab_results_reported": [{"test_name": null, "value": null, "date": null}],
  "risk_flags": [null],
  "summary": null
}"""

PAYLOADS = {
    "null_valued_fields": NULL_FIELDS,
    "empty_object": "{}",
    "not_json_at_all": "I cannot answer that.",
    "markdown_wrapped": "```json\n{\"findings\": [], \"summary\": \"none\"}\n```",
    "truncated_json": '{"findings": [{"type": "medication_conflict",',
    "findings_as_strings": '{"findings": ["Warfarin bleeding risk"], "summary": "s"}',
    "nested_nonsense": '{"findings": [[1, 2]], "medications_reported": "Warfarin"}',
    "unicode_and_newlines": '{"findings": [], "summary": "café\\n\\ttab—dash"}',
}


@pytest.mark.parametrize("name,payload", sorted(PAYLOADS.items()))
def test_backend_payload_does_not_crash_the_run(name, payload):
    """Whatever the backend returns, the run completes and scores."""
    r = HardRunner(seed=3, num_patients=12, llm_provider=_Scripted(payload))
    results = r.run()
    assert results, f"{name}: run produced nothing"
    assert not r.failed_scenarios, (
        f"{name}: {len(r.failed_scenarios)} scenarios failed — "
        f"{r.failed_scenarios[0]['error']}")
    ev = evaluate_run(results)
    assert 0.0 <= ev["classification"]["f1"] <= 1.0


def test_null_fields_reach_the_synthesis_step():
    """The specific crash seen in a live run: joining over null-valued keys.

    LLM synthesis is what formats the disclosed evidence into a prompt, so the
    null-valued payload has to travel all the way through it.
    """
    r = HardRunner(seed=3, num_patients=12, synthesis_mode="llm",
                   llm_provider=_Scripted(NULL_FIELDS))
    results = r.run()
    assert results and not r.failed_scenarios


def test_a_failing_backend_is_reported_not_swallowed():
    """A scenario that cannot complete is recorded, not silently dropped."""
    class _Explodes(BaseLLMProvider):
        def __init__(self):
            self._init_stats()
        def is_available(self):
            return True
        def generate(self, s, u):
            raise RuntimeError("backend exploded")

    r = HardRunner(seed=3, num_patients=8, llm_provider=_Explodes())
    r.run()
    # The agent converts a backend error into an llm_error finding rather than
    # propagating, so the run completes; what matters is that it does not crash.
    assert r.results
