"""Global state (env.state() / env.state_space) used by centralised critics."""

import random

import numpy as np
import pytest

import gridhunt
from gridhunt import _BENCHMARKS


@pytest.mark.parametrize("name", sorted(_BENCHMARKS))
def test_state_within_state_space(name):
    env = gridhunt.make(name)
    env.reset(seed=0)
    rng = random.Random(0)
    for _ in range(300):
        state = env.state()
        assert env.state_space.contains(state), f"{name}: state out of bounds"
        env.step({a: rng.randrange(5) for a in env.possible_agents})
        if not env.agents:
            env.reset()


@pytest.mark.parametrize("name", sorted(_BENCHMARKS))
def test_state_is_seeded(name):
    first, second = gridhunt.make(name), gridhunt.make(name)
    first.reset(seed=5)
    second.reset(seed=5)
    np.testing.assert_array_equal(first.state(), second.state())


def test_state_tracks_positions_and_toxicity():
    env = gridhunt.make("GridHunt-ToxicStag-v0")
    env.reset(seed=0, options={"predator_positions": {"predator_0": (6, 0)}})
    state = env.state()
    # predator_0 is the first agent: x = 6 / (7 - 1) = 1.0, y = 0
    assert state[0] == 1.0 and state[1] == 0.0
    stag_offset = 2 * len(env.possible_agents) + 4 * sorted(env.prey_ids).index("stag_0")
    assert state[stag_offset + 2] == 1.0  # present
    assert state[stag_offset + 3] == float(env.prey["stag_0"].status.get("toxic", False))
