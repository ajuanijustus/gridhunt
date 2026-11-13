"""Benchmark protocol: scenario overrides, variants and budgets."""

import sys
from pathlib import Path

import gridhunt  # noqa: F401
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from protocol import SCENARIOS, make_env  # noqa: E402

@pytest.mark.parametrize("scenario", sorted(SCENARIOS))
def test_protocol_scenarios_build(scenario):
    env = make_env(scenario)
    env.reset(seed=0)
    assert env.reward_mode == "team" and env.max_steps == SCENARIOS[scenario]["max_steps"]
    assert env.state_space.contains(env.state())


def test_full_vision_variant_changes_information_not_encoding():
    plain, full = make_env("specialists"), make_env("specialists_full_vision")
    obs_plain, _ = plain.reset(seed=0)
    obs_full, _ = full.reset(seed=0)
    for agent in plain.possible_agents:
        assert plain.observation_space(agent).shape == full.observation_space(agent).shape
    assert full.predators["bruiser_0"].limited_vision_radius >= max(full.grid_size)
    # Same positions, but the Bruiser sees more entities with full vision
    assert obs_full["bruiser_0"].sum() >= obs_plain["bruiser_0"].sum()
    assert [p.capture_power for p in plain.predators.values()] == [p.capture_power for p in full.predators.values()]
    # The registry's own predators are untouched
    assert gridhunt.make("GridHunt-Specialists-v0").predators["bruiser_0"].limited_vision_radius == 2


def test_blind_variant_hides_toxicity_only():
    plain, blind = make_env("toxic_stag"), make_env("toxic_stag_blind")
    assert plain.predators["predator_0"].status.get("toxicity_aware")
    assert not any(p.status.get("toxicity_aware") for p in blind.predators.values())
    assert plain.observation_space("predator_0").shape == blind.observation_space("predator_0").shape


def test_military_full_vision_widens_ground_vehicle_vision():
    full = make_env("military_full_vision")
    assert full.predators["ground_vehicle_0"].limited_vision_radius >= max(full.grid_size)
    assert make_env("military").observation_space("uav_0").shape == full.observation_space("uav_0").shape


def test_raid_and_swarm_use_coordinate_observations():
    swarm, raid = make_env("swarm"), make_env("raid")
    assert swarm.observation_mode == raid.observation_mode == "flat_coords"
    assert swarm.observation_space("predator_0").shape == (2 * (10 + 8),)
    obs, _ = swarm.reset(seed=0)
    assert obs["predator_0"].shape == (36,)


def test_every_scenario_has_a_budget_and_variants_match_their_main_scenario():
    main = {spec["env_id"]: spec["budget"] for spec in SCENARIOS.values() if spec["group"] == "main"}
    for name, spec in SCENARIOS.items():
        assert isinstance(spec["budget"], int) and spec["budget"] > 0, name
        assert spec["budget"] == main[spec["env_id"]], name
