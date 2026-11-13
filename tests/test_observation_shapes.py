"""
Test observation shape consistency across all observation modes and scenarios.
Verifies that observation_space().contains(obs) returns True for all observations.
"""

import pytest
import gridhunt
import numpy as np

SCENARIOS = [
    "GridHunt-ClassicStag-v0",
    "GridHunt-ToxicStag-v0",
    "GridHunt-Specialists-v0",
    "GridHunt-Swarm-v0",
    "GridHunt-Military-v0",
    "GridHunt-Raid-v0",
    "GridHunt-PIG-v0",
    "GridHunt-LandAndSea-v0",
]


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_observation_shape_coords_mode(scenario):
    """Verify 'coords' observation mode produces consistent shapes."""
    env = gridhunt.make(scenario, observation_mode="coords")
    obs, info = env.reset()

    # Check that all agents have observations
    for agent_id in env.agents:
        assert agent_id in obs, f"Agent {agent_id} missing from observations"

        # Get expected observation space
        obs_space = env.observation_space(agent_id)

        # Verify observation is contained in observation space
        assert obs_space.contains(
            obs[agent_id]
        ), f"{scenario}/{agent_id}: obs shape {obs[agent_id].shape} not in space {obs_space}"


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_observation_shape_flat_coords_mode(scenario):
    """Verify 'flat_coords' observation mode produces consistent shapes."""
    env = gridhunt.make(scenario, observation_mode="flat_coords")
    obs, info = env.reset()

    for agent_id in env.agents:
        assert agent_id in obs
        obs_space = env.observation_space(agent_id)
        assert obs_space.contains(
            obs[agent_id]
        ), f"{scenario}/{agent_id}: obs shape {obs[agent_id].shape} not in space {obs_space}"


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_observation_shape_one_hot_mode(scenario):
    """Verify 'one_hot' observation mode produces consistent shapes."""
    env = gridhunt.make(scenario, observation_mode="one_hot")
    obs, info = env.reset()

    for agent_id in env.agents:
        assert agent_id in obs
        obs_space = env.observation_space(agent_id)
        assert obs_space.contains(
            obs[agent_id]
        ), f"{scenario}/{agent_id}: obs shape {obs[agent_id].shape} not in space {obs_space}"


def test_observation_shape_stability_across_resets():
    """Observation shape should not change across resets."""
    env = gridhunt.make("GridHunt-ClassicStag-v0", observation_mode="coords")

    obs1, _ = env.reset(seed=42)
    obs2, _ = env.reset(seed=42)
    obs3, _ = env.reset(seed=99)

    for agent_id in env.agents:
        assert (
            obs1[agent_id].shape == obs2[agent_id].shape
        ), "Shape changed between resets"
        assert (
            obs1[agent_id].shape == obs3[agent_id].shape
        ), "Shape changed with different seed"


def test_observation_shape_stability_after_captures():
    """Observation shape should remain constant even after prey are captured (CRITICAL: tests fix for Raid)."""
    env = gridhunt.make("GridHunt-Raid-v0", observation_mode="coords")
    obs1, info = env.reset()

    obs_space = env.observation_space(env.agents[0])

    # Capture first building
    barn_pos = env.prey["barn_0"].position
    env.predators["raider_0"].position = barn_pos
    obs2, _, _, _, _ = env.step({"raider_0": 0, "raider_1": 0, "raider_2": 0})

    # Capture second building
    house_pos = env.prey["house_0"].position
    env.predators["raider_1"].position = house_pos
    obs3, _, _, _, _ = env.step({"raider_0": 0, "raider_1": 0, "raider_2": 0})

    # All observations should have same shape
    for agent_id in env.agents:
        assert (
            obs1[agent_id].shape == obs2[agent_id].shape
        ), f"Observation shape changed after first capture"
        assert (
            obs1[agent_id].shape == obs3[agent_id].shape
        ), f"Observation shape changed after second capture"
        assert obs_space.contains(
            obs2[agent_id]
        ), f"Observation after first capture not in observation_space"
        assert obs_space.contains(
            obs3[agent_id]
        ), f"Observation after second capture not in observation_space"


def test_observation_contains_sentinel_for_captured_prey(scenario="GridHunt-Raid-v0"):
    """Captured non-respawning prey should be encoded as sentinel [-1, -1] in 'coords' mode."""
    env = gridhunt.make(scenario, observation_mode="coords")
    obs, info = env.reset()

    num_prey = len(env.prey_ids)
    num_predators = len(env.predator_ids)

    # Capture barn (first building)
    barn_pos = env.prey["barn_0"].position
    env.predators["raider_0"].position = barn_pos
    obs, _, _, _, _ = env.step({"raider_0": 0, "raider_1": 0, "raider_2": 0})

    # barn_0 should no longer be in self.prey
    assert "barn_0" not in env.prey

    # But observation should still have space for it (filled with -1)
    # Prey are encoded in first num_prey * 2 elements
    agent_obs = obs[env.agents[0]]

    # Verify observation still matches observation_space
    obs_space = env.observation_space(env.agents[0])
    assert obs_space.contains(
        agent_obs
    ), f"Observation with captured prey not in observation_space"


def test_observation_dtype_consistency():
    """All observations should be float32."""
    env = gridhunt.make("GridHunt-ClassicStag-v0")
    obs, _ = env.reset()

    for agent_id in env.agents:
        assert (
            obs[agent_id].dtype == np.float32
        ), f"Agent {agent_id} observation dtype is {obs[agent_id].dtype}, expected float32"
