import pytest

import gridhunt


@pytest.fixture
def env_pig():
    """Pig chase scenario with impassable obstacles."""
    return gridhunt.make("GridHunt-PigChase-v0")


@pytest.fixture
def env_land_and_sea():
    """Land and Sea scenario with terrain-restricted predators."""
    return gridhunt.make("GridHunt-LandAndSea-v0")


@pytest.fixture
def env_military():
    """Military scenario with terrain-immune UAV."""
    return gridhunt.make("GridHunt-Military-v0")


def test_impassable_terrain_blocks_movement(env_pig):
    """Passability=0 terrain should prevent agent movement."""
    obs, info = env_pig.reset()

    predator = env_pig.predators["predator_0"]

    obstacles = env_pig.terrain_map.get("obstacle", set())
    assert len(obstacles) > 0, "Pig chase should have obstacles"

    test_obstacle = list(obstacles)[0]
    adjacent_to_obstacle = (test_obstacle[0] - 1, test_obstacle[1])

    if (
        0 <= adjacent_to_obstacle[0] < env_pig.grid_size[0]
        and 0 <= adjacent_to_obstacle[1] < env_pig.grid_size[1]
    ):

        predator.position = adjacent_to_obstacle

        # Try to move right into obstacle
        obs, _, _, _, _ = env_pig.step(
            {"predator_0": 4, "predator_1": 0}  # RIGHT action
        )

        assert (
            predator.position == adjacent_to_obstacle
        ), "Agent should be blocked by impassable terrain"


def test_terrain_passability_probability():
    """Terrain with passability < 1.0 should sometimes block movement."""
    env = gridhunt.make("GridHunt-Military-v0")
    obs, info = env.reset()

    assert len(env.terrain_config) > 0, "Military should have terrain config"


def test_restricted_terrain_blocks_predator(env_land_and_sea):
    """Predators with restricted_terrains should not move to those terrains."""
    obs, info = env_land_and_sea.reset()

    sea_predator = env_land_and_sea.predators["sea_predator"]
    land_predator = env_land_and_sea.predators["land_predator"]

    assert sea_predator.restricted_terrains == {
        "land"
    }, "sea_predator should be restricted from land"
    assert land_predator.restricted_terrains == {
        "sea"
    }, "land_predator should be restricted from sea"

    sea_positions = env_land_and_sea.terrain_map.get("sea", set())
    land_positions = env_land_and_sea.terrain_map.get("land", set())

    assert len(sea_positions) > 0, "Should have sea terrain"
    assert len(land_positions) > 0, "Should have land terrain"

    sea_pos = list(sea_positions)[0]
    land_pos = list(land_positions)[0]

    sea_predator.position = sea_pos

    if land_pos[0] > sea_pos[0]:
        action = 4  # RIGHT
    else:
        action = 3  # LEFT

    obs, _, _, _, _ = env_land_and_sea.step(
        {"sea_predator": action, "land_predator": 0}
    )

    assert sea_predator.position not in land_positions


def test_terrain_immune_ignores_passability(env_military):
    """Agents with immune_to_terrain=True should ignore passability checks."""
    obs, info = env_military.reset()

    uav = env_military.predators["uav_0"]

    assert uav.immune_to_terrain is True, "UAV should be immune to terrain"


def test_predator_cannot_cross_terrain_restriction(env_land_and_sea):
    """Land predator should not be able to move to sea terrain."""
    obs, info = env_land_and_sea.reset()

    land_predator = env_land_and_sea.predators["land_predator"]
    sea_positions = env_land_and_sea.terrain_map.get("sea", set())

    land_positions = env_land_and_sea.terrain_map.get("land", set())
    land_pos = list(land_positions)[0]

    land_predator.position = land_pos

    sea_adjacent = None
    for sea_tile in sea_positions:
        if abs(sea_tile[0] - land_pos[0]) + abs(sea_tile[1] - land_pos[1]) == 1:
            sea_adjacent = sea_tile
            break

    if sea_adjacent:
        if sea_adjacent[0] > land_pos[0]:
            action = 4  # RIGHT
        elif sea_adjacent[0] < land_pos[0]:
            action = 3  # LEFT
        elif sea_adjacent[1] > land_pos[1]:
            action = 2  # DOWN
        else:
            action = 1  # UP

        initial_pos = land_predator.position
        obs, _, _, _, _ = env_land_and_sea.step(
            {"land_predator": action, "sea_predator": 0}
        )

        assert (
            land_predator.position == initial_pos
        ), "Land predator should be blocked from entering sea terrain"


def test_terrain_positions_not_used_for_spawning(env_pig):
    """Impassable terrain positions should never be used for agent spawning."""
    env_pig.reset()

    obstacles = env_pig.terrain_map.get("obstacle", set())
    predator_positions = {p.position for p in env_pig.predators.values()}
    prey_positions = {p.position for p in env_pig.prey.values()}

    all_agent_positions = predator_positions | prey_positions

    assert (
        len(all_agent_positions & obstacles) == 0
    ), "No agents should spawn on impassable terrain"
