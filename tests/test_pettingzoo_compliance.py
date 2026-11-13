"""
PettingZoo ParallelEnv API compliance tests.
Based on PettingZoo's built-in test standards and best practices.
Validates that GridHunt correctly implements the ParallelEnv contract.
"""

import numpy as np
import pytest

import gridhunt

SCENARIOS = [
    "GridHunt-ClassicStag-v0",
    "GridHunt-ToxicStag-v0",
    "GridHunt-Raid-v0",
]


@pytest.mark.parametrize("scenario", SCENARIOS)
class TestParallelEnvContract:
    """Test PettingZoo ParallelEnv contract compliance."""

    def test_env_creation(self, scenario):
        """Environment must be creatable."""
        env = gridhunt.make(scenario)
        assert env is not None
        assert hasattr(env, "agents")
        assert hasattr(env, "reset")
        assert hasattr(env, "step")

    def test_agents_property(self, scenario):
        """agents property must be list of strings."""
        env = gridhunt.make(scenario)
        assert isinstance(env.agents, list)
        assert all(isinstance(a, str) for a in env.agents)
        assert len(env.agents) > 0

    def test_possible_agents_property(self, scenario):
        """possible_agents must match agents (ParallelEnv standard)."""
        env = gridhunt.make(scenario)
        # For ParallelEnv, possible_agents should equal agents
        assert set(env.possible_agents) == set(env.agents)

    def test_action_space_signature(self, scenario):
        """action_space(agent) must return valid Gym space."""
        env = gridhunt.make(scenario)
        for agent in env.agents:
            space = env.action_space(agent)
            assert space is not None
            assert hasattr(space, "sample")
            assert hasattr(space, "contains")
            # Sample and verify
            sample = space.sample()
            assert space.contains(sample)

    def test_observation_space_signature(self, scenario):
        """observation_space(agent) must return valid Gym space."""
        env = gridhunt.make(scenario)
        for agent in env.agents:
            space = env.observation_space(agent)
            assert space is not None
            assert hasattr(space, "sample")
            assert hasattr(space, "contains")

    def test_reset_returns_tuple(self, scenario):
        """reset() must return (observations, infos)."""
        env = gridhunt.make(scenario)
        result = env.reset()
        assert isinstance(result, tuple)
        assert len(result) == 2
        obs, infos = result
        assert isinstance(obs, dict)
        assert isinstance(infos, dict)

    def test_reset_observations_valid(self, scenario):
        """All reset observations must be in observation_space."""
        env = gridhunt.make(scenario)
        obs, _ = env.reset()
        for agent in env.agents:
            assert agent in obs, f"Agent {agent} missing from observations"
            space = env.observation_space(agent)
            assert space.contains(obs[agent]), f"Observation for {agent} not in space"

    def test_reset_infos_contain_all_agents(self, scenario):
        """reset() infos must contain all agents."""
        env = gridhunt.make(scenario)
        _, infos = env.reset()
        for agent in env.agents:
            assert agent in infos, f"Agent {agent} missing from infos"

    def test_step_takes_dict_actions(self, scenario):
        """step() must accept dict of actions."""
        env = gridhunt.make(scenario)
        env.reset()
        actions = {agent: env.action_space(agent).sample() for agent in env.agents}
        result = env.step(actions)
        assert isinstance(result, tuple)
        assert len(result) == 5

    def test_step_returns_five_tuple(self, scenario):
        """step() must return 5-tuple: (obs, rewards, terms, truncs, infos)."""
        env = gridhunt.make(scenario)
        env.reset()
        actions = {agent: env.action_space(agent).sample() for agent in env.agents}
        obs, rewards, terminations, truncations, infos = env.step(actions)

        # Verify structure
        assert isinstance(obs, dict)
        assert isinstance(rewards, dict)
        assert isinstance(terminations, dict)
        assert isinstance(truncations, dict)
        assert isinstance(infos, dict)

    def test_step_all_agents_in_returns(self, scenario):
        """step() must return data for all agents."""
        env = gridhunt.make(scenario)
        env.reset()
        actions = {agent: env.action_space(agent).sample() for agent in env.agents}
        obs, rewards, terms, truncs, infos = env.step(actions)

        for agent in env.agents:
            assert agent in obs, f"Agent {agent} missing from obs"
            assert agent in rewards, f"Agent {agent} missing from rewards"
            assert agent in terms, f"Agent {agent} missing from terminations"
            assert agent in truncs, f"Agent {agent} missing from truncations"
            assert agent in infos, f"Agent {agent} missing from infos"

    def test_step_observations_valid(self, scenario):
        """All step observations must be in observation_space."""
        env = gridhunt.make(scenario)
        env.reset()
        actions = {agent: env.action_space(agent).sample() for agent in env.agents}
        obs, _, _, _, _ = env.step(actions)

        for agent in env.agents:
            space = env.observation_space(agent)
            assert space.contains(obs[agent]), f"Step obs for {agent} not in space"

    def test_simultaneous_actions_processed(self, scenario):
        """All actions must be processed simultaneously (ParallelEnv)."""
        env = gridhunt.make(scenario)
        env.reset()

        # All agents should be able to act simultaneously
        actions = {agent: env.action_space(agent).sample() for agent in env.agents}
        obs, rewards, _, _, _ = env.step(actions)

        # All agents should receive observations and rewards
        assert len(rewards) == len(env.agents)
        assert len(obs) == len(env.agents)

    def test_rewards_numeric(self, scenario):
        """All rewards must be numeric."""
        env = gridhunt.make(scenario)
        env.reset()
        actions = {agent: env.action_space(agent).sample() for agent in env.agents}
        _, rewards, _, _, _ = env.step(actions)

        for agent in env.agents:
            r = rewards[agent]
            assert isinstance(
                r, (int, float, np.number)
            ), f"Reward for {agent} is not numeric: {type(r)}"

    def test_terminations_truncations_bool(self, scenario):
        """Terminations and truncations must be boolean."""
        env = gridhunt.make(scenario)
        env.reset()
        actions = {agent: env.action_space(agent).sample() for agent in env.agents}
        _, _, terms, truncs, _ = env.step(actions)

        for agent in env.agents:
            assert isinstance(
                terms[agent], (bool, np.bool_)
            ), f"Termination for {agent} is not bool: {type(terms[agent])}"
            assert isinstance(
                truncs[agent], (bool, np.bool_)
            ), f"Truncation for {agent} is not bool: {type(truncs[agent])}"


@pytest.mark.parametrize("scenario", SCENARIOS)
class TestDeterminism:
    """Test seeding and determinism (PettingZoo standard)."""

    def test_seeded_reset_deterministic(self, scenario):
        """Same seed should produce same initial state."""
        env1 = gridhunt.make(scenario)
        env2 = gridhunt.make(scenario)

        obs1, _ = env1.reset(seed=42)
        obs2, _ = env2.reset(seed=42)

        for agent in env1.agents:
            np.testing.assert_array_equal(obs1[agent], obs2[agent])

    def test_different_seeds_different_state(self, scenario):
        """Different seeds should produce different states."""
        env = gridhunt.make(scenario)
        obs1, _ = env.reset(seed=42)
        obs2, _ = env.reset(seed=43)

        # At least one agent's observation should differ
        different = False
        for agent in env.agents:
            if not np.array_equal(obs1[agent], obs2[agent]):
                different = True
                break
        assert different, "Different seeds should produce different observations"

    def test_episode_determinism(self, scenario):
        """Episode trajectory must be deterministic with same seed."""
        env1 = gridhunt.make(scenario)
        env2 = gridhunt.make(scenario)

        env1.reset(seed=123)
        rollout_1 = []
        for _ in range(5):
            actions = {agent: 0 for agent in env1.agents}
            obs1, _, _, _, _ = env1.step(actions)
            rollout_1.append({agent: obs1[agent].copy() for agent in env1.agents})

        env2.reset(seed=123)
        for step_num in range(5):
            actions = {agent: 0 for agent in env2.agents}
            obs2, _, _, _, _ = env2.step(actions)
            for agent in env2.agents:
                np.testing.assert_array_equal(
                    rollout_1[step_num][agent],
                    obs2[agent],
                    err_msg=f"Step {step_num} observations differ for {agent}",
                )


@pytest.mark.parametrize("scenario", SCENARIOS)
class TestAgentPoolConsistency:
    """Test agent pool remains consistent (PettingZoo standard)."""

    def test_agents_constant_across_resets(self, scenario):
        """Agent list must not change across resets."""
        env = gridhunt.make(scenario)
        agents_initial = set(env.agents)

        env.reset()
        assert set(env.agents) == agents_initial

        env.reset(seed=999)
        assert set(env.agents) == agents_initial

    def test_agents_constant_across_steps(self, scenario):
        """Agent list must not change after steps."""
        env = gridhunt.make(scenario)
        agents_initial = set(env.agents)
        env.reset()

        for _ in range(10):
            actions = {agent: env.action_space(agent).sample() for agent in env.agents}
            env.step(actions)
            assert set(env.agents) == agents_initial, "Agent set changed during episode"

    def test_action_space_constant(self, scenario):
        """Action space must be consistent for each agent."""
        env = gridhunt.make(scenario)
        env.reset()

        for _ in range(5):
            for agent in env.agents:
                space1 = env.action_space(agent)
                space2 = env.action_space(agent)
                assert type(space1) is type(space2)

                sample1 = space1.sample()
                sample2 = space2.sample()
                assert space1.contains(sample1)
                assert space2.contains(sample2)

            actions = {agent: env.action_space(agent).sample() for agent in env.agents}
            env.step(actions)
