"""The coverage claim must rest on a fact about the code, not on a hope.

The paper argues that a conventional decision-support system is unbeatable
inside its interaction table and cannot detect anything outside it, while a
federation of language-model agents covers both. The second half of that is an
empirical result. The first half is not: it follows from the knowledge base
containing no rule for any held-out conflict, and it would silently become
false if one were added. test_heldout_family.py pins that.

What this file pins is the plumbing the claim is measured through -- that the
held-out arm actually runs the held-out generator, that both families report
per-task rates so the two can be set side by side, and that the table renders
the comparison rather than dropping it.
"""
import pytest

from llm.provider import MockLLMProvider
from simulation.experiments_v2 import R1Experiments
from simulation.tables import table_coverage


@pytest.fixture(scope="module")
def exp(tmp_path_factory):
    return R1Experiments(llm_provider=MockLLMProvider(), num_patients=6,
                         out_dir=str(tmp_path_factory.mktemp("out")), n_seeds=1)


def test_heldout_runner_uses_the_heldout_generator(exp):
    assert exp.runner(use_heldout=True).use_heldout is True
    assert exp.runner().use_heldout is False


def test_e5_reports_both_families_with_per_task_rates(exp):
    e5 = exp.e5_baselines()
    assert "heldout_family" in e5, "the held-out block is missing"
    for family in (e5["rows"], e5["heldout_family"]["rows"]):
        for name, row in family.items():
            for k in ("conflict_detection", "pattern_detection"):
                assert k in row, f"{name} has no {k}"
                assert "rate" in row[k]


def test_coverage_table_renders_all_four_columns(exp):
    e5 = exp.e5_baselines()
    tex = table_coverage(e5)
    assert tex, "table_coverage produced nothing"
    assert r"\label{tab:r1_coverage}" in tex
    assert "MedAgentNet" in tex and "Conventional CDSS" in tex
    body = [l for l in tex.splitlines() if l.startswith("MedAgentNet")]
    assert body and body[0].count("&") == 5, body


def test_coverage_table_is_empty_without_the_heldout_block():
    """A results file predating this experiment must not render a broken table."""
    assert table_coverage({"rows": {"medagentnet": {}}}) == ""
