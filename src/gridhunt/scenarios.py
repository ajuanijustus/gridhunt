import numpy as np
from gymnasium.spaces import Box

from .agents import ToxicStag
from .core import GridHunt


class StagHuntScenario(GridHunt):
    """Baseline stag-hunt scenario."""

    def get_observations(self):
        return super().get_observations()


class ToxicStagHuntScenario(GridHunt):
    """Stag-hunt variant with toxic stag dynamics.

    Observation mode "toxic": egocentric one-hot grids of the other agents and of
    the prey, a toxicity flag (only for toxicity-aware agents) and the normalized
    last-step rewards of all agents.
    """

    observation_modes = GridHunt.observation_modes + ("toxic",)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._observation_spaces = {}

    def reset(self, seed=None, options=None):
        observations, infos = super().reset(seed=seed, options=options)
        # Draw each episode's starting toxicity from this env's (seeded) RNG
        for prey_item in self.prey.values():
            if isinstance(prey_item, ToxicStag):
                prey_item._reroll_disease_status(self._np_rng)
        return self.get_observations(), infos

    def step(self, actions):
        observations, rewards, terminations, truncations, infos = super().step(actions)
        self._age_toxic_prey()
        if self.observation_mode == "toxic":
            observations = self.get_observations()
        return observations, rewards, terminations, truncations, infos

    def _age_toxic_prey(self):
        """Respawn toxic stags that have been alive for 5 steps."""
        for prey_item in list(self.prey.values()):
            if isinstance(prey_item, ToxicStag) and prey_item.status.get(
                "toxic", False
            ):
                prey_item.toxic_turns_alive += 1
                if prey_item.toxic_turns_alive >= 5:
                    prey_item.execute_respawn(self)
                    prey_item.toxic_turns_alive = 0

    def observation_space(self, agent):
        if self.observation_mode != "toxic":
            return super().observation_space(agent)
        # PettingZoo requires the same space object on every call
        if agent not in self._observation_spaces:
            grid_x, grid_y = self.grid_size
            one_hot_size = ((grid_x - 1) * 2 + 1) * ((grid_y - 1) * 2 + 1)
            num_agents = len(self.possible_agents)
            num_prey = len(self.prey_ids)
            num_other_agents = num_agents - 1
            num_entities = num_other_agents + num_prey
            obs_dim = num_entities * one_hot_size + 1 + len(self.possible_agents)
            # One-hot grids and flag are in {0, 1}; rewards are normalized to [-1, 1]
            self._observation_spaces[agent] = Box(
                low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32
            )
        return self._observation_spaces[agent]

    def _other_agent_ids(self, agent_index):
        """Other agents in the order they appear in an agent's observation."""
        agent_id = self.possible_agents[agent_index]
        return [other_id for other_id in self.possible_agents if other_id != agent_id]

    def _prey_observation_order(self):
        """Prey ids in the order they appear in observations."""
        return sorted(self.prey_ids)

    def get_observations(self):
        if self.observation_mode != "toxic":
            return super().get_observations()

        agents_obs = {}
        rewards_normalized = np.array(
            [
                np.clip(self.rewards_for_one_hot_encoding.get(agent, 0) / self.max_reward, -1.0, 1.0)
                for agent in self.possible_agents
            ],
            dtype=np.float32,
        )
        any_toxic = any(
            prey_item.status.get("toxic", False) for prey_item in self.prey.values()
        )

        for agent_index, agent_id in enumerate(self.possible_agents):
            if agent_id in self.predators:
                agent_pos = self.predators[agent_id].position
                agent = self.predators[agent_id]
            else:
                agent_pos = self.prey[agent_id].position
                agent = self.prey[agent_id]

            agent_vision_radius = self._get_agent_vision_radius(agent)
            agent_has_limited_vision = self._has_limited_vision(agent)

            other_agent_positions = [
                (
                    self.predators[other_id].position
                    if other_id in self.predators
                    else self.prey[other_id].position
                )
                for other_id in self._other_agent_ids(agent_index)
            ]

            if agent_has_limited_vision:
                encode_func = self._get_one_hot_matrix_limited_vision
                vision_args = agent_vision_radius
            else:
                encode_func = self._get_one_hot_matrix_full_vision
                vision_args = None

            state_parts = []
            for pos in other_agent_positions:
                relative_pos = self._get_relative_position(agent_pos, pos)
                encoded = (
                    encode_func(relative_pos, vision_args)
                    if vision_args is not None
                    else encode_func(relative_pos)
                )
                state_parts.append(encoded)

            for prey_id in self._prey_observation_order():
                prey_pos = (
                    self.prey[prey_id].position
                    if prey_id in self.prey
                    else (-999, -999)
                )
                relative_pos = self._get_relative_position(agent_pos, prey_pos)
                encoded = (
                    encode_func(relative_pos, vision_args)
                    if vision_args is not None
                    else encode_func(relative_pos)
                )
                state_parts.append(encoded)

            is_aware = agent.status.get("toxicity_aware", False)
            toxic_flag = np.array(
                [1.0 if is_aware and any_toxic else 0.0], dtype=np.float32
            )
            state_parts.append(toxic_flag)
            state_parts.append(rewards_normalized)
            agents_obs[agent_id] = np.concatenate(state_parts, dtype=np.float32)

        return agents_obs


class RaidScenario(GridHunt):
    """Structure-raid scenario with optional trap penalties."""

    def reset(self, seed=None, options=None):
        observations, infos = super().reset(seed=seed, options=options)
        self._traps_triggered = {prey_id: False for prey_id in self.prey}
        self._trap_trigger_positions = {prey_id: None for prey_id in self.prey}
        return observations, infos

    def step(self, actions):
        prey_before = dict(self.prey)
        observations, rewards, terminations, truncations, infos = super().step(actions)
        captured_prey_ids = set(prey_before.keys()) - set(self.prey.keys())
        
        # Initialize __common__ key if not present (for shared environment-level info)
        if "__common__" not in infos:
            infos["__common__"] = {}
        
        for prey_id in captured_prey_ids:
            prey_item = prey_before[prey_id]
            if prey_id not in self._traps_triggered:
                self._traps_triggered[prey_id] = False
                self._trap_trigger_positions[prey_id] = None
            prey_pos = prey_item.position
            already_triggered = (
                self._traps_triggered[prey_id]
                and self._trap_trigger_positions[prey_id] == prey_pos
            )
            if already_triggered:
                continue
            trap_probability = prey_item.status.get("trap_probability", 0.0)
            if trap_probability > 0.0 and self._np_rng.random() < trap_probability:
                self._traps_triggered[prey_id] = True
                self._trap_trigger_positions[prey_id] = prey_pos
                trap_penalty = prey_item.status.get("trap_penalty", 0)
                for agent_in_game in self.possible_agents:
                    rewards[agent_in_game] += trap_penalty
                infos["__common__"]["team_reward"] = (
                    infos["__common__"].get("team_reward", 0.0) + trap_penalty
                )
                self.score += trap_penalty
                # Store trap-triggered event in __common__ instead of top-level infos
                infos["__common__"][f"{prey_id}_trap_triggered"] = True
        return observations, rewards, terminations, truncations, infos


class PigChaseScenario(GridHunt):
    """Maze chase scenario with respawn-to-origin behavior."""

    def reset(self, seed=None, options=None):
        observations, infos = super().reset(seed=seed, options=options)
        self.original_prey_positions = {
            prey_id: prey.position for prey_id, prey in self.prey.items()
        }
        self.original_predator_positions = {
            pred_id: predator.position for pred_id, predator in self.predators.items()
        }
        return observations, infos

    def step(self, actions):
        observations, rewards, terminations, truncations, infos = super().step(actions)
        pred_rewards = sum(
            reward for agent_id, reward in rewards.items() if agent_id in self.predators
        )
        if pred_rewards > 0:
            for prey_id, original_pos in self.original_prey_positions.items():
                if prey_id in self.prey:
                    self.prey[prey_id].position = original_pos
            for pred_id, original_pos in self.original_predator_positions.items():
                if pred_id in self.predators:
                    self.predators[pred_id].position = original_pos
            # The base step built observations before this reset (showing the capture
            # positions and the pig's random respawn cell); rebuild them from the reset state.
            observations = self.get_observations()
        return observations, rewards, terminations, truncations, infos

    def get_observations(self):
        return super().get_observations()
