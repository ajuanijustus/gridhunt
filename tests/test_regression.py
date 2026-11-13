"""Environment dynamics must match the recorded reference trajectories exactly.

See tests/regression/trajectories.py for what is recorded and when the reference may change.
"""

import gzip
import json
import sys
from pathlib import Path

import pytest

import gridhunt

sys.path.insert(0, str(Path(__file__).parent / "regression"))
import trajectories  # noqa: E402

REFERENCE = Path(__file__).parent / "regression" / "reference_trajectories.json.gz"

with gzip.open(REFERENCE, "rt") as _f:
    _REFERENCE = json.load(_f)


def test_reference_matches_the_recorded_settings():
    assert _REFERENCE["steps"] == trajectories.STEPS
    assert _REFERENCE["max_steps"] == trajectories.MAX_STEPS
    assert _REFERENCE["seeds"] == list(trajectories.SEEDS)
    assert set(_REFERENCE["cases"]) == set(trajectories.CASES)


def test_every_registered_environment_is_covered():
    covered = {case["env_id"] for case in trajectories.CASES.values()}
    assert set(gridhunt.list_envs()) <= covered


@pytest.mark.parametrize("seed", trajectories.SEEDS)
@pytest.mark.parametrize("name", sorted(trajectories.CASES))
def test_trajectory_matches_reference(name, seed):
    env = trajectories.make_case_env(gridhunt, trajectories.CASES[name])
    digests = trajectories.run_case(env, seed)
    env.close()
    expected = _REFERENCE["cases"][name][str(seed)]
    for index, (got, want) in enumerate(zip(digests, expected, strict=False)):
        assert got == want, f"{name}, seed {seed}: first difference at record {index} (0 is the first reset)"
    assert len(digests) == len(expected)
