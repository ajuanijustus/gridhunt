"""
Reference (non-learning) policies for the GridHunt benchmark.

- RandomPolicy: uniform random actions (the lower reference score).
- IdlePolicy: always STILL (used to remove an agent in the interdependence ablation).
- GreedyPolicy: a scripted team. Each predator walks the shortest path (BFS over the
  moves it can actually make: speed, impassable terrain, restricted terrains) to the
  prey with the best reward per step of distance, and waits on it until captured.
  It reads the true state, including whether a stag is toxic, so it is an
  information-oracle heuristic, not an upper bound on what is achievable.

Policies act on the environment state (env.predators, env.prey, env.terrain_at); they
never draw from the env's RNGs, so evaluation trajectories are unaffected by them.
"""

import random
from collections import deque

STILL, UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3, 4
MOVES = (UP, DOWN, LEFT, RIGHT)


class RandomPolicy:
    def __init__(self, seed=0):
        self.rng = random.Random(seed)

    def act(self, env, agent_id):
        return self.rng.randrange(5)


class IdlePolicy:
    def act(self, env, agent_id):
        return STILL


def next_position(env, agent, position, action, speed=None):
    """Deterministic version of GridHunt._move: where an action takes an agent.

    Cells with 0 < passability < 1 are treated as passable (the env may randomly
    block such a move); impassable and restricted terrain block the move.
    """
    speed = agent.speed if speed is None else speed
    if action == STILL:
        return position
    x, y = position
    width, height = env.grid_size
    if action == UP:
        y = max(0, y - speed)
    elif action == DOWN:
        y = min(height - 1, y + speed)
    elif action == LEFT:
        x = max(0, x - speed)
    elif action == RIGHT:
        x = min(width - 1, x + speed)
    new_position = (x, y)
    terrain_type = env.terrain_at.get(new_position)
    if terrain_type is None:
        return new_position
    if terrain_type in getattr(agent, "restricted_terrains", set()):
        return position
    if getattr(agent, "immune_to_terrain", False):
        return new_position
    config = env.terrain_config.get(terrain_type, {})
    if config.get("passability", 1.0) == 0.0:
        return position
    multiplier = config.get("speed_multiplier", 1.0)
    if multiplier < 1.0:
        reduced = max(0, int(speed * multiplier))
        return position if reduced == 0 else next_position(env, agent, position, action, reduced)
    return new_position


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


class GreedyPolicy:
    """Scripted team: go to the best reward-per-distance prey and wait there.

    active_agents: the predators actually playing (default: all). Prey needing more
    capture power than these agents have together are ignored, so a team that has
    lost a member re-plans instead of waiting for a partner who never comes.
    """

    def __init__(self, active_agents=None):
        self.active_agents = active_agents

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
            if prey.reward <= 0 or prey.position not in paths:
                continue  # skip harmful (e.g. toxic) and unreachable prey
            if prey.capture_power_required > team_power:
                continue  # the playing team cannot capture it
            distance, first = paths[prey.position]
            score = prey.reward / (1 + distance)
            if best_score is None or score > best_score:
                best_score, best_action = score, first
        return best_action


def run_episode(env, policies, seed):
    """Play one episode; policies maps agent_id -> policy. Returns the team return."""
    env.reset(seed=seed)
    total = 0
    while env.agents:
        actions = {agent: policies[agent].act(env, agent) for agent in env.possible_agents}
        _, _, _, _, infos = env.step(actions)
        total += infos["__common__"]["team_reward"]
    return total
