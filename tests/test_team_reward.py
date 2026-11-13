"""
Reward modes: "individual" (only predators on a captured prey are rewarded) and
"team" (every predator receives the team reward). In both modes
infos["__common__"]["team_reward"] counts each capture once.
"""

import random

import pytest

import gridhunt
from gridhunt import _BENCHMARKS

STILL = 0


def _classic(reward_mode, predator_positions, prey_positions):
    env = gridhunt.make("GridHunt-ClassicStag-v0", reward_mode=reward_mode)
    env.reset(
        seed=0,
        options={"predator_positions": predator_positions, "prey_positions": prey_positions},
    )
    return env


# ClassicStag: predator_0/1 (power 1), stag_0 (reward 5, needs 2), hare_0/1 (reward 1, needs 1)
PREY_AWAY = {"stag_0": (4, 4), "hare_0": (4, 0), "hare_1": (0, 4)}


@pytest.mark.parametrize("reward_mode", ["individual", "team"])
def test_joint_stag_capture(reward_mode):
    env = _classic(reward_mode, {"predator_0": (2, 2), "predator_1": (2, 2)}, {**PREY_AWAY, "stag_0": (2, 2)})
    _, rewards, _, _, infos = env.step({a: STILL for a in env.possible_agents})
    assert rewards == {"predator_0": 5, "predator_1": 5}
    assert infos["__common__"]["team_reward"] == 5  # one capture, counted once


@pytest.mark.parametrize("reward_mode,expected", [
    ("individual", {"predator_0": 1, "predator_1": 0}),
    ("team", {"predator_0": 1, "predator_1": 1}),
])
def test_solo_hare_capture(reward_mode, expected):
    env = _classic(reward_mode, {"predator_0": (2, 2), "predator_1": (0, 0)}, {**PREY_AWAY, "hare_0": (2, 2)})
    _, rewards, _, _, infos = env.step({a: STILL for a in env.possible_agents})
    assert rewards == expected
    assert infos["__common__"]["team_reward"] == 1


@pytest.mark.parametrize("reward_mode", ["individual", "team"])
def test_overcrowded_hare_counted_once(reward_mode):
    env = _classic(reward_mode, {"predator_0": (2, 2), "predator_1": (2, 2)}, {**PREY_AWAY, "hare_0": (2, 2)})
    _, rewards, _, _, infos = env.step({a: STILL for a in env.possible_agents})
    assert rewards == {"predator_0": 0.5, "predator_1": 0.5}  # reward 1 minus overcrowding penalty 0.5
    assert infos["__common__"]["team_reward"] == 0.5


def test_zero_power_agent_rewarded_in_team_mode():
    """Military's UAV (capture_power 0) never captures, but shares the team reward."""
    env = gridhunt.make("GridHunt-Military-v0", reward_mode="team")
    env.reset(seed=0)
    rng = random.Random(0)
    for _ in range(3000):
        _, rewards, _, _, infos = env.step({a: rng.randrange(5) for a in env.possible_agents})
        assert rewards["uav_0"] == rewards["ground_vehicle_0"] == infos["__common__"]["team_reward"]
        if not env.agents:
            env.reset()


@pytest.mark.parametrize("reward_mode", ["individual", "team"])
def test_score_is_sum_of_team_rewards(reward_mode):
    env = gridhunt.make("GridHunt-ToxicStag-v0", reward_mode=reward_mode)
    env.reset(seed=1)
    rng = random.Random(1)
    total = 0
    for _ in range(500):
        _, _, _, _, infos = env.step({a: rng.randrange(5) for a in env.possible_agents})
        total += infos["__common__"]["team_reward"]
    assert env.score == pytest.approx(total)


def test_invalid_reward_mode():
    with pytest.raises(ValueError):
        gridhunt.make("GridHunt-ClassicStag-v0", reward_mode="shared")


@pytest.mark.parametrize("reward_mode", ["individual", "team"])
@pytest.mark.parametrize("name", sorted(_BENCHMARKS))
def test_observations_within_declared_space(name, reward_mode):
    env = gridhunt.make(name, reward_mode=reward_mode)
    obs, _ = env.reset(seed=2)
    rng = random.Random(2)
    for _ in range(400):
        for agent, value in obs.items():
            assert env.observation_space(agent).contains(value), f"{name}: {agent} obs out of bounds"
        obs, *_ = env.step({a: rng.randrange(5) for a in env.possible_agents})
        if not env.agents:
            obs, _ = env.reset()


def test_capture_records():
    env = _classic("team", {"predator_0": (2, 2), "predator_1": (2, 2)}, {**PREY_AWAY, "stag_0": (2, 2)})
    _, _, _, _, infos = env.step({a: STILL for a in env.possible_agents})
    assert infos["__common__"]["captures"] == [
        {"prey": "stag_0", "type": "stag", "toxic": False, "overcrowded": False}
    ]
    _, _, _, _, infos = env.step({a: STILL for a in env.possible_agents})
    assert infos["__common__"]["captures"] == [] or infos["__common__"]["captures"][0]["prey"] != "stag_0"
