# Training agents

GridHunt is a standard PettingZoo `ParallelEnv`, so any multi-agent library that accepts one can
train on it. The repository's [benchmark](../benchmark/index.md) has ready-made IPPO, MAPPO and
HAPPO implementations in plain PyTorch.

## The bundled baselines

From a clone of the repository:

```bash
pip install -e ".[benchmark]"
python benchmark/train_marl.py --scenario toxic_stag --algo mappo --seed 0
```

This trains with the benchmark's settings (team reward, 200-step episodes, the scenario's budget),
evaluates on held-out seeds every 1% of training, and writes `progress.csv`, `evals.jsonl`,
`final.json` and `model.pt` to `benchmark/results/marl/toxic_stag/mappo/seed_0/`. Pass `--budget`
for a shorter run. Interrupted runs resume from their checkpoint.

## Your own training loop

What a learner needs from GridHunt:

- per-agent observations and rewards from `env.step`, or the team reward from
  `infos["__common__"]["team_reward"]`;
- the global state from `env.state()` for centralised critics;
- `truncations`: episodes end only by the time limit, so bootstrap the value of the last state
  instead of treating it as terminal.

```python
import numpy as np
import gridhunt

env = gridhunt.make("GridHunt-Specialists-v0", reward_mode="team", max_steps=200)
observations, _ = env.reset(seed=0)
rng = np.random.default_rng(0)
episode_return = 0.0
while env.agents:
    actions = {agent: int(rng.integers(env.action_space(agent).n)) for agent in env.agents}
    observations, rewards, terminations, truncations, infos = env.step(actions)
    episode_return += infos["__common__"]["team_reward"]
    state = env.state()          # input for a centralised critic
print(episode_return)
```

If several agents share one network, give it the agent's identity: with `coords` and `flat_coords`
observations every agent receives the same vector ([observations](observations.md)).

## Reference policies

`gridhunt.policies` has scripted teams that need no training: `RandomPolicy`, `IdlePolicy` and
`GreedyPolicy`, which walks each predator to the prey with the best reward per step of distance.
`GreedyPolicy` reads the true state, including hidden information such as stag toxicity, so it is a
reference point rather than an upper bound.

```python
import gridhunt
from gridhunt.policies import GreedyPolicy

env = gridhunt.make("GridHunt-ToxicStag-v0", max_steps=100)
policy = GreedyPolicy()
env.reset(seed=0)
while env.agents:
    env.step({agent: policy.act(env, agent) for agent in env.agents})
```

The benchmark's trained teams are available as numpy policies in [`gridhunt.zoo`](../benchmark/zoo.md).
