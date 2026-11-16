"""Movement rules: multi-cell moves, terrain restrictions and action masks."""

import numpy as np

import gridhunt
from gridhunt import Hare, Predator
from gridhunt.core import DOWN, LEFT, RIGHT, STILL, UP
from gridhunt.scenarios import StagHuntScenario


def _strip(terrain, speed=2, **predator):
    """A 6x1 corridor with one predator at (0, 0) and the given terrain."""
    env = StagHuntScenario(
        grid_size=(6, 1),
        max_steps=10,
        predators=[Predator("runner", speed=speed, position=(0, 0), **predator)],
        prey=[Hare("hare_0", position=(5, 0))],
        terrain_config=terrain,
        observation_mode="coords",
    )
    env.reset(seed=0)
    return env


def _cells(name, cells, **settings):
    return {name: {"generation": {"type": "predefined", "positions": cells}, **settings}}


def test_a_fast_agent_cannot_jump_a_wall():
    open_strip = _strip({})
    assert open_strip._move((0, 0), RIGHT, agent=open_strip.predators["runner"]) == (2, 0)
    walled = _strip(_cells("obstacle", [(1, 0)], passability=0.0))
    runner = walled.predators["runner"]
    assert walled._move((0, 0), RIGHT, agent=runner) == (0, 0)
    walled.step({"runner": RIGHT})
    assert runner.position == (0, 0)


def test_a_fast_agent_stops_before_a_wall_further_on():
    env = _strip(_cells("obstacle", [(2, 0)], passability=0.0), speed=3)
    assert env._move((0, 0), RIGHT, agent=env.predators["runner"]) == (1, 0)


def test_slow_terrain_can_be_entered_but_ends_the_move():
    env = _strip(_cells("swamp", [(1, 0)], speed_multiplier=0.5), speed=3)
    assert env._move((0, 0), RIGHT, agent=env.predators["runner"]) == (1, 0)
    slow_walker = _strip(_cells("swamp", [(1, 0)], speed_multiplier=0.5), speed=1)
    assert slow_walker._move((0, 0), RIGHT, agent=slow_walker.predators["runner"]) == (1, 0)


def test_immune_agents_ignore_walls_but_not_restrictions():
    env = _strip({**_cells("obstacle", [(1, 0)], passability=0.0), **_cells("sea", [(3, 0)])},
                 immune_to_terrain=True, restricted_terrains=["sea"])
    runner = env.predators["runner"]
    assert env._move((0, 0), RIGHT, agent=runner) == (2, 0)
    assert env._move((2, 0), RIGHT, agent=runner) == (2, 0)


def test_restricted_to_blocks_movement_and_the_mask():
    # Role 0 is the predators' role unless agent_role_map says otherwise
    env = _strip(_cells("lava", [(1, 0)], restricted_to={0}), speed=1)
    env.step({"runner": RIGHT})
    assert env.predators["runner"].position == (0, 0)
    assert env.action_mask("runner")[RIGHT] == 0.0


def test_mask_marks_edges_and_keeps_stay_valid():
    env = _strip({}, speed=1)
    mask = env.action_mask("runner")
    assert mask.dtype == np.float32
    assert mask[STILL] == 1.0 and mask[RIGHT] == 1.0
    assert mask[LEFT] == mask[UP] == mask[DOWN] == 0.0


def test_land_and_sea_masks_keep_each_unit_on_its_side():
    env = gridhunt.make("GridHunt-LandAndSea-v0", use_action_mask=True)
    # (6, 8) is land with sea at (5, 8); (5, 8) is sea with land at (6, 8)
    env.reset(seed=0, options={"predator_positions": {"land_predator": (6, 8), "sea_predator": (5, 8)}})
    assert env.terrain_at[(6, 8)] == "land" and env.terrain_at[(5, 8)] == "sea"
    assert env.action_mask("land_predator")[LEFT] == 0.0
    assert env.action_mask("sea_predator")[RIGHT] == 0.0
    _, _, _, _, infos = env.step({"land_predator": STILL, "sea_predator": STILL})
    np.testing.assert_array_equal(infos["__common__"]["action_masks"]["land_predator"],
                                  env.action_mask("land_predator"))


def test_military_mask_lets_the_uav_cross_walls_but_not_the_ground_vehicle():
    env = gridhunt.make("GridHunt-Military-v0")
    env.reset(seed=0)
    walls = [cell for cell, kind in env.terrain_at.items() if kind == "obstacle"]
    start = next((x - 1, y) for (x, y) in walls if x > 0 and (x - 1, y) not in env.terrain_at)
    env.predators["uav_0"].position = start
    env.predators["ground_vehicle_0"].position = start
    assert env.action_mask("uav_0")[RIGHT] == 1.0
    assert env.action_mask("ground_vehicle_0")[RIGHT] == 0.0
