"""End-to-end tests of the architectural claims, on the deterministic backend.

These run in well under a minute and are what CI executes on every push.
"""
import pytest

from simulation.runner_v2 import HardRunner
from simulation.evaluation import evaluate_run
from simulation import baselines as bl

N = 60
SEED = 42


def _f1(**kw):
    r = HardRunner(seed=SEED, num_patients=N, **kw)
    return evaluate_run(r.run())["classification"]["f1"], r


@pytest.fixture(scope="module")
def full():
    return _f1(**bl.ARCHITECTURE_VARIANTS["medagentnet"])


def test_full_system_is_accurate(full):
    f1, _ = full
    assert f1 > 0.85, f"full system F1 dropped to {f1}"


def test_removing_cross_departmental_assembly_is_catastrophic(full):
    """The paper's central ablation. Without assembly, a conflict whose limbs
    sit in different departments cannot be found."""
    base, _ = full
    ablated, _ = _f1(**bl.ARCHITECTURE_VARIANTS["ablate_synthesis"])
    assert ablated < base / 2, (
        f"assembly ablation should collapse detection: {ablated} vs {base}")


def test_removing_orchestration_is_worse_still(full):
    no_orch, _ = _f1(**bl.ARCHITECTURE_VARIANTS["ablate_orchestration"])
    no_synth, _ = _f1(**bl.ARCHITECTURE_VARIANTS["ablate_synthesis"])
    assert no_orch <= no_synth


def test_orchestrator_never_touches_a_patient_record():
    """The orchestrator must hold no channel to any departmental store."""
    r = HardRunner(seed=SEED, num_patients=10)
    assert not hasattr(r.orchestrator, "patient_store")
    for name in dir(r.orchestrator):
        assert "patient_store" not in name


def test_tier_one_discloses_nothing_identifiable():
    from simulation.privacy import measure_disclosure
    r = HardRunner(seed=SEED, num_patients=40)
    r.run(force_tier=1)
    per_tier = measure_disclosure(r)["per_tier"].get("tier_1", {})
    assert per_tier.get("mean_field_exposure", 1.0) == 0.0


def test_tier_two_reaches_higher_utility_than_tier_one():
    r1 = HardRunner(seed=SEED, num_patients=N)
    f1_t1 = evaluate_run(r1.run(force_tier=1))["classification"]["f1"]
    r2 = HardRunner(seed=SEED, num_patients=N)
    f1_t2 = evaluate_run(r2.run(force_tier=2))["classification"]["f1"]
    assert f1_t2 > f1_t1 * 2


def test_agent_failure_does_not_abort_the_request():
    """Fault tolerance: a partial answer must be returned and labelled."""
    from agents.core import DepartmentAgent

    class Dead(DepartmentAgent):
        def process_query(self, query):
            raise ConnectionError("agent unavailable")

    r = HardRunner(seed=SEED, num_patients=20)
    r.generate()
    victim = "cardiology" if "cardiology" in r.department_agents else \
        sorted(r.department_agents)[0]
    stub = Dead(victim, r.dept_config[victim], r.llm, r.audit)
    stub.patient_store = r.department_agents[victim].patient_store
    r.department_agents[victim] = stub
    r.orchestrator.agents[victim] = stub
    r.build_scenarios()
    results = r.run()
    assert results, "the run aborted instead of degrading"
    assert any(victim in x["privacy_report"].get("unreachable_departments", [])
               for x in results)
    assert any(x["privacy_report"].get("coverage", 1.0) < 1.0 for x in results)


def test_provider_factory_refuses_to_substitute_silently():
    """R0 returned the rule-based provider when a model server was unreachable,
    producing results labelled with a model that never ran."""
    from llm.provider import create_llm_provider, ProviderUnavailable
    cfg = {"provider": "ollama",
           "ollama": {"base_url": "http://127.0.0.1:1", "model": "nope:0b"}}
    with pytest.raises(ProviderUnavailable):
        create_llm_provider(cfg, strict=True)


def test_negative_controls_are_genuinely_negative():
    """A control must contain no real interaction, or it is dropped."""
    from data.generator_hard import HardCaseGenerator, COHORT_DISTRACTOR, COHORT_CONTROL
    from protocol.interactions import evaluate_rules, evaluate_patterns
    gen = HardCaseGenerator(config_dir="config", seed=SEED)
    for p in gen.generate(80):
        if getattr(p, "cohort", "") not in (COHORT_DISTRACTOR, COHORT_CONTROL):
            continue
        ev = gen._record_evidence(p, (p.encounter_hint or {}).get("procedure", ""))
        hits = evaluate_rules(ev) + evaluate_patterns(ev)
        assert not hits, f"{p.patient_id} labelled negative but contains {hits[0]['id']}"
