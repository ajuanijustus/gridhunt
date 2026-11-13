"""
Run several copies of a GridHunt scenario in lockstep and batch their data.

Episodes in GridHunt end only by the time limit (truncation). When an episode ends,
the copy is reset automatically and the step still returns the last observation and
state of the finished episode (final_obs / final_state), so value targets can
bootstrap from them instead of treating the time limit as a terminal state.
"""

import numpy as np


class VecEnv:
    def __init__(self, make_env, num_envs, seed):
        """make_env: zero-argument callable returning a new environment."""
        self.envs = [make_env() for _ in range(num_envs)]
        self.num_envs = num_envs
        self.seed = seed
        self.agents = list(self.envs[0].possible_agents)
        self._episode_returns = np.zeros(num_envs)

    def obs_dim(self, agent):
        return self.envs[0].observation_space(agent).shape[0]

    @property
    def state_dim(self):
        return self.envs[0].state_space.shape[0]

    def reset(self):
        """Seed every copy once; later resets continue each copy's own RNG stream."""
        observations = []
        for i, env in enumerate(self.envs):
            obs, _ = env.reset(seed=self.seed * 10_000 + i)
            observations.append(obs)
        self._episode_returns[:] = 0.0
        return self._batch(observations), self._states()

    def step(self, actions):
        """actions: {agent: int array [num_envs]}. Returns a dict of batched arrays."""
        observations, final_observations, final_states = [], [], []
        rewards = {agent: np.zeros(self.num_envs, dtype=np.float32) for agent in self.agents}
        team_rewards = np.zeros(self.num_envs, dtype=np.float32)
        dones = np.zeros(self.num_envs, dtype=bool)
        captures, finished_returns = [], []
        for i, env in enumerate(self.envs):
            obs, reward, _, _, infos = env.step({a: int(actions[a][i]) for a in self.agents})
            for agent in self.agents:
                rewards[agent][i] = reward[agent]
            team_rewards[i] = infos["__common__"]["team_reward"]
            captures.extend(infos["__common__"]["captures"])
            self._episode_returns[i] += team_rewards[i]
            final_observations.append(obs)
            final_states.append(env.state())
            if not env.agents:  # time limit reached: start the next episode
                dones[i] = True
                finished_returns.append(self._episode_returns[i])
                self._episode_returns[i] = 0.0
                obs, _ = env.reset()
            observations.append(obs)
        return {
            "obs": self._batch(observations),
            "state": self._states(),
            "rewards": rewards,
            "team_rewards": team_rewards,
            "dones": dones,
            "final_obs": self._batch(final_observations),
            "final_state": np.stack(final_states),
            "episode_returns": finished_returns,
            "captures": captures,
        }

    def _batch(self, observations):
        return {
            agent: np.stack([obs[agent] for obs in observations]).astype(np.float32)
            for agent in self.agents
        }

    def _states(self):
        return np.stack([env.state() for env in self.envs])
