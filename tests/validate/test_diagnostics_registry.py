from rra_population_model.validate.diagnostics.registry import (
    CHECKS,
    SEVERITY_RANK,
    checks_in_lane,
)

MIN_GUIDANCE_CHARS = 100  # a sentence of framing plus a sentence of reading advice, at least


def test_registry_ids_and_severities() -> None:
    for check_id, check in CHECKS.items():
        assert check.id == check_id
        assert check.severity in SEVERITY_RANK
        assert check.lane in {"A", "B", "C"}
        assert all(v >= 0 for v in check.thresholds.values())
        assert check.description
        assert len(check.guidance) > MIN_GUIDANCE_CHARS, f"{check_id} needs reading guidance"


def test_lanes_partition_the_registry() -> None:
    lanes = [c.id for lane in ("A", "B", "C") for c in checks_in_lane(lane)]
    assert sorted(lanes) == sorted(c.id for c in CHECKS.values() if c.enabled)
