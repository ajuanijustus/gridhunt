"""Scripted (non-learning) policies: baselines and ready-made teammates.

- RandomPolicy: uniform random actions.
- IdlePolicy: always STILL.
- GreedyPolicy: each predator walks the shortest path (BFS over the moves it can actually
  make, by the environment's movement rules) to the prey with the best reward per
  step of distance, and waits on it until captured. With its defaults it reads the true state,
  including whether a stag is toxic, so it is an informed heuristic, not an upper bound.

Every policy can be called two ways:

- ``policy.act(env, agent_id)``, as the benchmark does;
- ``policy(observation, agent_id, env)``, the teammate interface used by ``gridhunt.hat``.

Policies read the environment state (env.predators, env.prey, env.terrain_at) and never draw
from the environment's random generators, so they do not change its trajectory.
"""

import random
from collections import deque

STILL, UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3, 4
MOVES = (UP, DOWN, LEFT, RIGHT)


class _Policy:
    def __call__(self, observation, agent_id, env):
        return self.act(env, agent_id)

    def act(self, env, agent_id):
        raise NotImplementedError


class RandomPolicy(_Policy):
    """Uniform random actions from its own seeded generator."""

    def __init__(self, seed=0):
        self.rng = random.Random(seed)

    def act(self, env, agent_id):
        return self.rng.randrange(5)


class IdlePolicy(_Policy):
    """Always STILL."""

    def act(self, env, agent_id):
        return STILL


def next_position(env, agent, position, action, speed=None):
    """Where an action takes an agent, by the environment's own movement rules without
    randomness: a cell that can be entered with some probability counts as reachable."""
    return env._move(position, action, agent=agent, speed=speed, stochastic=False)


def shortest_paths(env, agent):
    """BFS from the agent's cell: {cell: (distance, first action on a shortest path)}."""
    start = agent.position
    best = {start: (0, STILL)}
    queue = deque([start])
    while queue:
        cell = queue.popleft()
        distance, first = best[cell]
        for action in MOVES:
            nxt = next_position(env, agent, cell, action)
            if nxt not in best:
                best[nxt] = (distance + 1, action if cell == start else first)
                queue.append(nxt)
    return best


class GreedyPolicy(_Policy):
    """Scripted team: go to the best reward-per-distance prey and wait there.

    Args:
        active_agents: the predators actually playing (default: all). Prey needing more
            capture power than these agents have together are ignored, so a team that has
            lost a member re-plans instead of waiting for a partner who never comes.
        prey_types: only chase prey of these types (for example ``("hare",)``); default: all.
        avoid_harmful: skip prey whose current reward is negative (a toxic stag). With
            ``False`` the policy ignores that hidden information and values every prey at
            its absolute reward, as a team that cannot tell toxic stags apart would.
    """

    def __init__(self, active_agents=None, prey_types=None, avoid_harmful=True):
        self.active_agents = active_agents
        self.prey_types = None if prey_types is None else set(prey_types)
        self.avoid_harmful = avoid_harmful

    def act(self, env, agent_id):
        agent = env.predators.get(agent_id)
        if agent is None:
            return STILL
        active = env.predators if self.active_agents is None else self.active_agents
        team_power = sum(env.predators[a].capture_power for a in active if a in env.predators)
        paths = shortest_paths(env, agent)
        best_score, best_action = None, STILL
        for prey_id in sorted(env.prey):
            prey = env.prey[prey_id]
            if self.prey_types is not None and prey.type not in self.prey_types:
                continue
            reward = prey.reward if self.avoid_harmful else abs(prey.reward)
            if reward <= 0 or prey.position not in paths:
                continue  # skip harmful (e.g. toxic) and unreachable prey
            if prey.capture_power_required > team_power:
                continue  # the playing team cannot capture it
            distance, first = paths[prey.position]
            score = reward / (1 + distance)
            if best_score is None or score > best_score:
                best_score, best_action = score, first
        return best_action
