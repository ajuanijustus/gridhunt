"""
Multi-agent interaction and communication tests.
Tests for proper handling of simultaneous actions and interactions.
Based on PettingZoo best practices.
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
class TestMultiAgentInteractions:
    """Test multi-agent mechanics and simultaneous processing."""

    def test_simultaneous_action_processing(self, scenario):
        """All actions must be processed in same step."""
        env = gridhunt.make(scenario)
        env.reset(seed=42)

        # All agents move simultaneously
        actions = {agent: 1 for agent in env.agents}  # All move UP
        obs, rewards, _, _, _ = env.step(actions)

        # Verify all agents moved (typically observation changes)
        assert all(agent in obs for agent in env.agents)

    def test_partial_actions_error_or_handle(self, scenario):
        """Missing actions should cause error or be handled gracefully."""
        env = gridhunt.make(scenario)
        env.reset()

        if len(env.agents) > 1:
            # Provide actions for only first agent
            partial_actions = {env.agents[0]: 0}

            # Either error or supplement with defaults
            try:
                obs, _, _, _, _ = env.step(partial_actions)
                # If no error, verify all agents still in obs
                assert all(
                    agent in obs for agent in env.agents
                ), "Partial actions: missing agents in obs"
            except (ValueError, KeyError):
                # Expected: missing actions not allowed
                pass

    def test_action_independence(self, scenario):
        """One agent's action shouldn't affect others' action spaces."""
        env = gridhunt.make(scenario)
        env.reset()

        for agent in env.agents:
            space = env.action_space(agent)
            assert space is not None

            # Space should be consistent
            sample1 = space.sample()
            sample2 = space.sample()
            assert space.contains(sample1)
            assert space.contains(sample2)

    def test_observation_independence(self, scenario):
        """All agents should receive valid observations."""
        env = gridhunt.make(scenario)
        env.reset()

        actions = {agent: env.action_space(agent).sample() for agent in env.agents}
        obs, _, _, _, _ = env.step(actions)

        for agent in env.agents:
            o = obs[agent]
            space = env.observation_space(agent)
            assert isinstance(o, (np.ndarray, float, int))
            assert space.contains(o), f"Obs for {agent} not in space"

    def test_all_agents_act_simultaneously(self, scenario):
        """Verify simultaneous processing by checking step count."""
        env = gridhunt.make(scenario)
        obs1, _ = env.reset(seed=123)

        # Take one step with all agents
        actions = {agent: 0 for agent in env.agents}  # STILL
        obs2, _, _, _, _ = env.step(actions)

        # Step count should increment by 1
        assert env.step_count == 1, "Step count not incremented correctly"

    def test_multiple_simultaneous_steps(self, scenario):
        """Multiple simultaneous steps should process all actions."""
        env = gridhunt.make(scenario)
        env.reset(seed=42)

        for expected_step in range(1, 6):
            actions = {agent: env.action_space(agent).sample() for agent in env.agents}
            env.step(actions)
            assert (
                env.step_count == expected_step
            ), f"Step count mismatch at step {expected_step}"


@pytest.mark.parametrize("scenario", SCENARIOS)
class TestRewardConsistency:
    """Test reward structure and consistency."""

    def test_rewards_are_numeric(self, scenario):
        """All rewards must be numeric."""
        env = gridhunt.make(scenario)
        env.reset()

        actions = {agent: env.action_space(agent).sample() for agent in env.agents}
        _, rewards, _, _, _ = env.step(actions)

        for agent in env.agents:
            r = rewards[agent]
            assert isinstance(
                r, (int, float, np.number)
            ), f"Reward for {agent} is {type(r)}, not numeric"

    def test_termination_truncation_bool(self, scenario):
        """Terminations and truncations must be boolean."""
        env = gridhunt.make(scenario)
        env.reset()

        actions = {agent: env.action_space(agent).sample() for agent in env.agents}
        _, _, terms, truncs, _ = env.step(actions)

        for agent in env.agents:
            assert isinstance(
                terms[agent], (bool, np.bool_)
            ), f"Termination for {agent} is not bool"
            assert isinstance(
                truncs[agent], (bool, np.bool_)
            ), f"Truncation for {agent} is not bool"

    def test_infos_contain_all_agents(self, scenario):
        """All agents must be in infos."""
        env = gridhunt.make(scenario)
        env.reset()

        actions = {agent: env.action_space(agent).sample() for agent in env.agents}
        _, _, _, _, infos = env.step(actions)

        for agent in env.agents:
            assert agent in infos, f"Agent {agent} missing from infos"

    def test_reward_consistency_across_episodes(self, scenario):
        """Rewards should be consistent with same seed."""
        env1 = gridhunt.make(scenario)
        env2 = gridhunt.make(scenario)

        env1.reset(seed=100)
        env2.reset(seed=100)

        for _ in range(3):
            actions = {agent: 0 for agent in env1.agents}  # All STILL
            _, rewards1, _, _, _ = env1.step(actions)
            _, rewards2, _, _, _ = env2.step(actions)

            for agent in env1.agents:
                assert (
                    rewards1[agent] == rewards2[agent]
                ), f"Rewards differ for {agent} with same seed"


@pytest.mark.parametrize("scenario", SCENARIOS)
class TestEpisodeBehavior:
    """Test episode-level behavior and state management."""

    def test_episode_reset(self, scenario):
        """Episode should reset properly."""
        env = gridhunt.make(scenario)

        # First episode
        obs1, _ = env.reset(seed=42)
        assert env.step_count == 0

        # Some steps
        for _ in range(5):
            actions = {agent: env.action_space(agent).sample() for agent in env.agents}
            env.step(actions)
        assert env.step_count == 5

        # Reset to new episode
        obs2, _ = env.reset(seed=42)
        assert env.step_count == 0

        # Should get same initial obs with same seed
        for agent in env.agents:
            np.testing.assert_array_equal(obs1[agent], obs2[agent])

    def test_max_steps_truncation(self, scenario):
        """Episode should truncate at max_steps."""
        env = gridhunt.make(scenario)
        env.reset()

        max_steps = env.max_steps
        truncated_any = False

        for step in range(max_steps + 5):
            actions = {agent: env.action_space(agent).sample() for agent in env.agents}
            _, _, _, truncations, _ = env.step(actions)

            if step >= max_steps - 1:
                # At or past max_steps, should be truncated
                if any(truncations.values()):
                    truncated_any = True

        assert truncated_any, "Episode did not truncate at max_steps"

    def test_consistent_agent_pool_within_episode(self, scenario):
        """Agent pool should remain constant within episode."""
        env = gridhunt.make(scenario)
        obs, _ = env.reset()
        agents_initial = set(env.agents)

        for _ in range(10):
            actions = {agent: env.action_space(agent).sample() for agent in env.agents}
            obs, _, _, _, _ = env.step(actions)
            assert set(env.agents) == agents_initial
