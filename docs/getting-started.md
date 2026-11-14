# Getting started

## Installation

GridHunt needs Python 3.10 or later.

```bash
pip install gridhunt
```

The core package depends only on NumPy, Gymnasium and PettingZoo. Optional extras add more:

| Extra | Adds | For |
|---|---|---|
| `render` | pygame-ce, pygame-gui, Pillow | windows, `rgb_array` frames, the interactive GUI |
| `hat` | the `render` extra | [human-agent teaming](hat/index.md) studies |
| `benchmark` | PyTorch, Matplotlib | the [baselines and analysis](benchmark/index.md) in the repository |
| `dev` | `render`, pytest, ruff | working on GridHunt itself |

## The environment loop

Every scenario is a PettingZoo `ParallelEnv`: all agents act at once, and every call takes and
returns dictionaries keyed by agent.

```python
import gridhunt

print(gridhunt.list_envs())

env = gridhunt.make("GridHunt-ToxicStag-v0")
observations, infos = env.reset(seed=0)
print(env.agents)                                   # ['predator_0', 'predator_1', 'predator_2']
print(env.observation_space("predator_0").shape)

while env.agents:
    actions = {agent: env.action_space(agent).sample() for agent in env.agents}
    observations, rewards, terminations, truncations, infos = env.step(actions)
```

- **Actions** are `Discrete(5)`: 0 stay, 1 up, 2 down, 3 left, 4 right.
- **Episodes** end by the time limit only (`truncations`), after `max_steps` steps.
- **Rewards** are individual by default: a predator is rewarded for the captures it takes part in.
  With `reward_mode="team"` every predator receives the team reward instead.
- **Infos** hold per-step team information in `infos["__common__"]`: `team_reward` (each capture
  counted once), `captures` (one record per capture, with the prey's type and whether it was toxic)
  and `prey_positions`.
- **Global state** for centralised critics: `env.state()` and `env.state_space`.

## Changing a scenario

`gridhunt.make` accepts any setting of the registered scenario as a keyword argument:

```python
import gridhunt

env = gridhunt.make(
    "GridHunt-ToxicStag-v0",
    max_steps=200,                 # shorter episodes
    reward_mode="team",            # shared reward
    observation_mode="one_hot",    # a different observation encoding
)
```

A misspelled setting raises an error instead of being ignored. To build a scenario from scratch,
see [custom scenarios](guides/custom-scenarios.md).

## PettingZoo integration

With PettingZoo 1.27 or later, importing GridHunt also registers every environment with
PettingZoo's registry:

```python
import gridhunt  # noqa: F401  (registers the environments)
import pettingzoo

env = pettingzoo.make("parallel", "GridHunt-ClassicStag-v0")
```

All environments pass PettingZoo's `parallel_api_test` and `parallel_seed_test`. Seeding
`env.reset(seed=...)` makes an episode fully reproducible; GridHunt never uses the global `random`
or `numpy.random` state.
