"""Scripted prey motion and the optional communication channel."""

import random
import warnings

import numpy as np
import pytest
from pettingzoo.test import parallel_api_test, parallel_seed_test

import gridhunt

STILL = 0


# -- prey motion ---------------------------------------------------------------

def test_static_prey_never_move():
    env = gridhunt.make("GridHunt-ClassicStag-v0")
    env.reset(seed=0)
    start = {p: prey.position for p, prey in env.prey.items()}
    for _ in range(50):
        _, _, _, _, infos = env.step({a: STILL for a in env.possible_agents})
        if not infos["__common__"]["captures"]:
            assert {p: prey.position for p, prey in env.prey.items()} == start


def test_invalid_prey_motion():
    with pytest.raises(ValueError):
        gridhunt.make("GridHunt-ClassicStag-v0", prey_motion="teleport")


def _distance_to_predators(env, position):
    return min(abs(position[0] - p.position[0]) + abs(position[1] - p.position[1]) for p in env.predators.values())


def test_fleeing_prey_never_move_closer_or_onto_predators():
    env = gridhunt.make("GridHunt-ClassicStag-v0", prey_motion="flee", prey_move_probability=1.0)
    env.reset(seed=3)
    for _ in range(200):
        before = {p: (prey.position, _distance_to_predators(env, prey.position)) for p, prey in env.prey.items()}
        _, _, _, _, infos = env.step({a: STILL for a in env.possible_agents})  # predators stay put
        captured = {c["prey"] for c in infos["__common__"]["captures"]}
        predator_cells = {p.position for p in env.predators.values()}
        for prey_id, prey in env.prey.items():
            if prey_id in captured:
                continue
            assert prey.position not in predator_cells
            assert _distance_to_predators(env, prey.position) >= before[prey_id][1]


def test_moving_prey_actually_move_and_are_seeded():
    def trajectory():
        env = gridhunt.make("GridHunt-ClassicStag-v0", prey_motion="random", prey_move_probability=1.0)
        env.reset(seed=7)
        rng = random.Random(7)
        positions = []
        for _ in range(100):
            env.step({a: rng.randrange(5) for a in env.possible_agents})
            positions.append(tuple(prey.position for prey in env.prey.values()))
        return positions

    first, second = trajectory(), trajectory()
    assert first == second
    assert len(set(first)) > 10


def test_specialists_prey_flee_by_default():
    env = gridhunt.make("GridHunt-Specialists-v0")
    assert env.prey_motion == "flee" and env.prey["stag_0"].capture_power_required == 2
    assert env.predators["bruiser_0"].status.get("limited_vision")


# -- communication channel ---------------------------------------------------------

def test_communication_passes_pettingzoo_checks():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        parallel_api_test(gridhunt.make("GridHunt-Specialists-v0", communication=3), num_cycles=1100)
        parallel_seed_test(lambda: gridhunt.make("GridHunt-Specialists-v0", communication=3), num_cycles=300)


def test_messages_reach_every_agent_next_step():
    env = gridhunt.make("GridHunt-ClassicStag-v0", communication=3)
    obs, _ = env.reset(seed=0)
    n_agents, vocab = len(env.possible_agents), 3
    for agent in env.possible_agents:
        assert env.action_space(agent).n == 5 * vocab
        np.testing.assert_array_equal(obs[agent][-n_agents * vocab:], 0.0)
    # predator_0 stays still and says token 2; predator_1 moves up and says token 1
    obs, _, _, _, infos = env.step({"predator_0": STILL + 5 * 2, "predator_1": 1 + 5 * 1})
    assert infos["__common__"]["messages"] == {"predator_0": 2, "predator_1": 1}
    expected = np.array([0, 0, 1, 0, 1, 0], dtype=np.float32)
    for agent in env.possible_agents:
        np.testing.assert_array_equal(obs[agent][-n_agents * vocab:], expected)
    np.testing.assert_array_equal(env.state()[-n_agents * vocab:], expected)
    assert env.state_space.contains(env.state())


def test_communication_does_not_change_the_game():
    """With token 0 everywhere, the wrapped env plays exactly like the plain one."""
    plain = gridhunt.make("GridHunt-Specialists-v0")
    talky = gridhunt.make("GridHunt-Specialists-v0", communication=4)
    obs_p, _ = plain.reset(seed=11)
    obs_t, _ = talky.reset(seed=11)
    rng = random.Random(11)
    for _ in range(300):
        actions = {a: rng.randrange(5) for a in plain.possible_agents}
        obs_p, rew_p, _, _, _ = plain.step(actions)
        obs_t, rew_t, _, _, _ = talky.step(actions)
        assert rew_p == rew_t
        for agent in plain.possible_agents:
            np.testing.assert_array_equal(obs_t[agent][: len(obs_p[agent])], obs_p[agent])


# -- protocol variants --------------------------------------------------------------

