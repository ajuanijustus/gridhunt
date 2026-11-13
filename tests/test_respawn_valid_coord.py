import numpy as np
import pytest
import gridhunt


@pytest.fixture
def env_classic():
    return gridhunt.make("GridHunt-ClassicStag-v0")


@pytest.fixture
def env_pig():
    return gridhunt.make("GridHunt-PIG-v0")


def test_hare_respawns_on_valid_coordinate(env_classic):
    env_classic.reset()
    hare_id = "hare_0"
    start = env_classic.prey[hare_id].position
    env_classic.predators["predator_0"].position = start
    env_classic.step({"predator_0": 0, "predator_1": 0})
    assert hare_id in env_classic.prey
    assert env_classic.prey[hare_id].position is not None


def test_respawned_prey_not_on_impassable_terrain(env_pig):
    env_pig.reset()
    pig = env_pig.prey["pig"]
    obstacles = env_pig.terrain_map.get("obstacle", set())
    for _ in range(5):
        pig_pos = pig.position
        env_pig.predators["predator_0"].position = pig_pos
        env_pig.predators["predator_1"].position = pig_pos
        env_pig.step({"predator_0": 0, "predator_1": 0})
        assert pig.position not in obstacles


def test_raid_buildings_do_not_respawn():
    env_raid = gridhunt.make("GridHunt-Raid-v0")
    env_raid.reset()
    barn_id = "barn_0"
    barn_pos = env_raid.prey[barn_id].position
    env_raid.predators["raider_0"].position = barn_pos
    env_raid.step({"raider_0": 0, "raider_1": 0, "raider_2": 0})
    assert barn_id not in env_raid.prey


def test_respawn_coordinates_are_in_bounds(env_classic):
    env_classic.reset()
    hare = env_classic.prey["hare_0"]
    for _ in range(10):
        env_classic.predators["predator_0"].position = hare.position
        env_classic.step({"predator_0": 0, "predator_1": 0})
        x, y = hare.position
        assert 0 <= x < env_classic.grid_size[0]
        assert 0 <= y < env_classic.grid_size[1]


def test_pig_chase_observation_after_capture_matches_reset_positions(env_pig):
    """After a capture everything returns to its start; the observation must show that."""
    env_pig.reset(seed=0)
    start = env_pig.get_observations()
    pig = env_pig.prey["pig"].position
    env_pig.predators["predator_0"].position = pig
    env_pig.predators["predator_1"].position = pig
    obs, _, _, _, infos = env_pig.step({"predator_0": 0, "predator_1": 0})
    assert infos["__common__"]["team_reward"] > 0
    grids = 4 * 81  # predator_0, predator_1, pig, obstacles on the 9x9 board
    for agent in obs:
        np.testing.assert_array_equal(obs[agent][:grids], start[agent][:grids])
        np.testing.assert_array_equal(obs[agent], env_pig.get_observations()[agent])
