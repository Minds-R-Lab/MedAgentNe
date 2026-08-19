"""Each ablation must differ from the reference system in exactly one thing.

Otherwise a McNemar test against the reference measures the sum of the
differences rather than the component named in the row. This nearly went wrong:
when the reported configuration moved from hybrid to grounded synthesis, every
ablation was still pinned to hybrid, which would have folded an 84-point
specificity gap into all ten rows.
"""
import pytest

from simulation.baselines import ARCHITECTURE_VARIANTS

REFERENCE = dict(
    routing_mode="relevance", synthesis_mode="rules", enforce_consent=True,
    enforce_tiers=True, structured_output=True, freetext_fallback=True,
    ground_reports=True,
)


def test_reference_is_the_documented_operating_point():
    ref = ARCHITECTURE_VARIANTS["medagentnet"]
    for k, v in REFERENCE.items():
        assert ref.get(k, REFERENCE[k]) == v, f"medagentnet.{k} is {ref.get(k)}"


@pytest.mark.parametrize(
    "name", [k for k in ARCHITECTURE_VARIANTS if k != "medagentnet"])
def test_variant_differs_in_exactly_one_dimension(name):
    merged = {**REFERENCE, **ARCHITECTURE_VARIANTS[name]}
    diffs = [k for k in REFERENCE if merged[k] != REFERENCE[k]]
    assert len(diffs) == 1, f"{name} differs in {diffs}, expected exactly one"


def test_every_switch_is_ablated_somewhere():
    covered = set()
    for name, kw in ARCHITECTURE_VARIANTS.items():
        if name == "medagentnet":
            continue
        merged = {**REFERENCE, **kw}
        covered.update(k for k in REFERENCE if merged[k] != REFERENCE[k])
    assert covered == set(REFERENCE), f"never ablated: {set(REFERENCE) - covered}"
