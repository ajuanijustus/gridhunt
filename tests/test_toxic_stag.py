"""ToxicStag's lifetime: how long an uncaught toxic stag stays before it moves and is redrawn."""

import pytest

import gridhunt
from gridhunt.policies import STILL


def respawn_steps(lifetime, steps=60, seed=0):
    """Steps at which an always-toxic, never-caught stag moves on its own."""
    stag = gridhunt.ToxicStag("stag_0", toxic_probability=1.0, toxic_lifetime=lifetime)
    env = gridhunt.make("GridHunt-ToxicStag-v0", prey=[stag])
    env.reset(seed=seed)
    live = env.prey["stag_0"]
    moved, original, clock = [], live.execute_respawn, {"step": 0}
    live.execute_respawn = lambda e: (moved.append(clock["step"]), original(e))
    while clock["step"] < steps:
        clock["step"] += 1
        env.step({agent: STILL for agent in env.agents})
    return moved


def test_default_lifetime_is_five_steps():
    assert respawn_steps(5, steps=20) == [5, 10, 15, 20]
    assert gridhunt.ToxicStag("stag_0").toxic_lifetime == 5


def test_a_fixed_lifetime_is_configurable():
    assert respawn_steps(3, steps=12) == [3, 6, 9, 12]


def test_no_lifetime_keeps_a_toxic_stag_until_it_is_caught():
    assert respawn_steps(None) == []


def test_a_lifetime_range_varies_and_stays_in_range():
    moved = respawn_steps((2, 6), steps=300)
    gaps = [b - a for a, b in zip([0, *moved], moved)]
    assert all(2 <= gap <= 6 for gap in gaps)
    assert len(set(gaps)) > 1
    assert respawn_steps((2, 6), steps=300) == moved  # drawn from the seeded environment RNG


@pytest.mark.parametrize("bad", [0, -1, 2.5, True, (3, 2), (0, 4), (1, 2, 3), "5"])
def test_invalid_lifetimes_are_rejected(bad):
    with pytest.raises(ValueError, match="toxic_lifetime"):
        gridhunt.ToxicStag("stag_0", toxic_lifetime=bad)
