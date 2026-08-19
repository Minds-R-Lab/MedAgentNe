"""An experiment that does not name a variant must run the reported system.

R1Experiments.runner() previously passed only the caller's kwargs, so E1, E2,
E3, E7, E9 and E5's reference arm inherited HardRunner's constructor defaults.
That was harmless only while those defaults happened to match the reported
configuration. When the operating point moved to grounded synthesis they kept
running the hybrid one, and E1 returned specificity 0.008 against the 0.859 the
diagnostic measured for the configuration described in the paper -- four hours
of GPU time answering a question nobody asked.
"""
import pytest

from simulation.baselines import ARCHITECTURE_VARIANTS
from simulation.experiments_v2 import R1Experiments
from llm.provider import MockLLMProvider


@pytest.fixture(scope="module")
def exp(tmp_path_factory):
    return R1Experiments(llm_provider=MockLLMProvider(), num_patients=4,
                         out_dir=str(tmp_path_factory.mktemp("out")))


def test_bare_runner_is_the_reported_system(exp):
    r = exp.runner()
    ref = ARCHITECTURE_VARIANTS["medagentnet"]
    assert r.orchestrator.synthesis_mode == ref["synthesis_mode"]
    assert r.orchestrator.routing_mode == ref["routing_mode"]
    assert all(a.ground_reports for a in r.department_agents.values())


@pytest.mark.parametrize("name", sorted(ARCHITECTURE_VARIANTS))
def test_a_named_variant_overrides_the_default(exp, name):
    kw = ARCHITECTURE_VARIANTS[name]
    r = exp.runner(**kw)
    if "synthesis_mode" in kw:
        assert r.orchestrator.synthesis_mode == kw["synthesis_mode"]
    if "routing_mode" in kw:
        assert r.orchestrator.routing_mode == kw["routing_mode"]
    if "ground_reports" in kw:
        assert all(a.ground_reports == kw["ground_reports"]
                   for a in r.department_agents.values())


def test_seed_and_patient_overrides_still_work(exp):
    r = exp.runner(seed=99, num_patients=7)
    assert r.seed == 99 and r.num_patients == 7
    assert r.orchestrator.synthesis_mode == "rules"
