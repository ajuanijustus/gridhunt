import pytest

import gridhunt


@pytest.fixture
def env_classic():
    return gridhunt.make("GridHunt-ClassicStag-v0")


@pytest.fixture
def env_raid():
    return gridhunt.make("GridHunt-Raid-v0")


def test_exact_power_capture(env_classic):
    env_classic.reset()
    stag_pos = env_classic.prey["stag_0"].position
    env_classic.predators["predator_0"].position = stag_pos
    env_classic.predators["predator_1"].position = stag_pos
    _, rewards, _, _, _ = env_classic.step({"predator_0": 0, "predator_1": 0})
    assert rewards["predator_0"] == 5.0
    assert rewards["predator_1"] == 5.0


def test_insufficient_power_no_capture(env_classic):
    env_classic.reset()
    stag_pos = env_classic.prey["stag_0"].position
    env_classic.predators["predator_0"].position = stag_pos
    _, rewards, _, _, _ = env_classic.step({"predator_0": 0, "predator_1": 0})
    assert rewards["predator_0"] == 0.0
    assert "stag_0" in env_classic.prey
    assert env_classic.prey["stag_0"].position == stag_pos


def test_overcrowded_hare_capture_penalty(env_classic):
    env_classic.reset()
    hare_pos = env_classic.prey["hare_0"].position
    env_classic.predators["predator_0"].position = hare_pos
    env_classic.predators["predator_1"].position = hare_pos
    hare = env_classic.prey["hare_0"]
    expected = hare.reward - hare.overcrowding_penalty
    _, rewards, _, _, _ = env_classic.step({"predator_0": 0, "predator_1": 0})
    assert rewards["predator_0"] == expected
    assert rewards["predator_1"] == expected


def test_raid_building_rewards_and_power(env_raid):
    env_raid.reset()
    assert env_raid.prey["barn_0"].reward == 2
    assert env_raid.prey["house_0"].reward == 5
    assert env_raid.prey["bank_0"].reward == 20
    assert env_raid.prey["barn_0"].capture_power_required == 1
    assert env_raid.prey["house_0"].capture_power_required == 2
    assert env_raid.prey["bank_0"].capture_power_required == 3
