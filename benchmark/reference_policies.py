"""
Reference (non-learning) policies for the GridHunt benchmark.

- RandomPolicy: uniform random actions (the lower reference score).
- IdlePolicy: always STILL (used to remove an agent in the interdependence ablation).
- GreedyPolicy: a scripted team (the upper reference score). It reads the true state,
  including whether a stag is toxic, so it is an information-oracle heuristic, not an upper
  bound on what is achievable.

The policies live in the package (gridhunt.policies), where they also serve as scripted
teammates for human-agent teaming studies.
"""

from gridhunt.policies import (  # noqa: F401
    DOWN,
    LEFT,
    MOVES,
    RIGHT,
    STILL,
    UP,
    GreedyPolicy,
    IdlePolicy,
    RandomPolicy,
    next_position,
    shortest_paths,
)


def run_episode(env, policies, seed):
    """Play one episode; policies maps agent_id -> policy. Returns the team return."""
    env.reset(seed=seed)
    total = 0
    while env.agents:
        actions = {agent: policies[agent].act(env, agent) for agent in env.possible_agents}
        _, _, _, _, infos = env.step(actions)
        total += infos["__common__"]["team_reward"]
    return total
