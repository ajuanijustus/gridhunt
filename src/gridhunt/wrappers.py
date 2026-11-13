"""Optional environment wrappers."""

import numpy as np
from gymnasium.spaces import Box, Discrete
from pettingzoo import ParallelEnv


class CommunicationWrapper(ParallelEnv):
    """Adds an explicit communication channel ("cheap talk") to a GridHunt env.

    Each step every agent chooses a move and one of `vocab_size` message tokens,
    encoded as a single action: action = move + num_moves * token. Every agent's
    next observation ends with all agents' tokens from that step (one one-hot block
    per agent, in possible_agents order; all zero after reset). The tokens are also
    appended to state() so centralised critics see them, and are reported in
    infos["__common__"]["messages"].

    Tokens have no meaning to the environment: any meaning must be learned. Moves
    0-4 are therefore still valid actions (token 0).
    """

    def __init__(self, env, vocab_size):
        if vocab_size < 1:
            raise ValueError("vocab_size must be at least 1")
        self.env = env
        self.vocab_size = vocab_size
        self.possible_agents = list(env.possible_agents)
        self.metadata = getattr(env, "metadata", {})
        self.render_mode = getattr(env, "render_mode", None)
        self._num_moves = {a: env.action_space(a).n for a in self.possible_agents}
        self._messages = np.zeros((len(self.possible_agents), vocab_size), dtype=np.float32)
        self._observation_spaces = {}
        self._action_spaces = {
            a: Discrete(self._num_moves[a] * vocab_size) for a in self.possible_agents
        }
        inner = env.state_space
        self.state_space = Box(
            low=np.concatenate([np.broadcast_to(inner.low, inner.shape), np.zeros(self._messages.size)]).astype(np.float32),
            high=np.concatenate([np.broadcast_to(inner.high, inner.shape), np.ones(self._messages.size)]).astype(np.float32),
            dtype=np.float32,
        )

    def __getattr__(self, name):
        # Everything else (predators, prey, grid_size, renderer, ...) comes from the env
        if name == "env":
            raise AttributeError(name)
        return getattr(self.env, name)

    @property
    def agents(self):
        return self.env.agents

    def observation_space(self, agent):
        if agent not in self._observation_spaces:
            inner = self.env.observation_space(agent)
            low = np.broadcast_to(inner.low, inner.shape)
            high = np.broadcast_to(inner.high, inner.shape)
            self._observation_spaces[agent] = Box(
                low=np.concatenate([low, np.zeros(self._messages.size)]).astype(np.float32),
                high=np.concatenate([high, np.ones(self._messages.size)]).astype(np.float32),
                dtype=np.float32,
            )
        return self._observation_spaces[agent]

    def action_space(self, agent):
        return self._action_spaces[agent]

    def _with_messages(self, observations):
        flat = self._messages.ravel()
        return {a: np.concatenate([obs, flat]).astype(np.float32) for a, obs in observations.items()}

    def reset(self, seed=None, options=None):
        observations, infos = self.env.reset(seed=seed, options=options)
        self._messages[:] = 0.0
        return self._with_messages(observations), infos

    def step(self, actions):
        moves, tokens = {}, {}
        for agent, action in actions.items():
            moves[agent] = int(action) % self._num_moves[agent]
            tokens[agent] = int(action) // self._num_moves[agent]
        observations, rewards, terminations, truncations, infos = self.env.step(moves)
        self._messages[:] = 0.0
        for i, agent in enumerate(self.possible_agents):
            if agent in tokens:
                self._messages[i, tokens[agent]] = 1.0
        infos.setdefault("__common__", {})["messages"] = tokens
        return self._with_messages(observations), rewards, terminations, truncations, infos

    def state(self):
        return np.concatenate([self.env.state(), self._messages.ravel()]).astype(np.float32)

    def render(self):
        return self.env.render()

    def close(self):
        return self.env.close()
