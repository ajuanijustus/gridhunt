# GridHunt

[![Tests](https://github.com/ajuanijustus/gridhunt/actions/workflows/tests.yml/badge.svg)](https://github.com/ajuanijustus/gridhunt/actions/workflows/tests.yml)
[![Docs](https://github.com/ajuanijustus/gridhunt/actions/workflows/docs.yml/badge.svg)](https://ajuanijustus.github.io/gridhunt/)
[![PyPI](https://img.shields.io/pypi/v/gridhunt)](https://pypi.org/project/gridhunt/)
[![Python](https://img.shields.io/pypi/pyversions/gridhunt)](https://pypi.org/project/gridhunt/)
[![PettingZoo](https://img.shields.io/badge/PettingZoo-1.27.0-blue)](https://pettingzoo.farama.org)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](https://github.com/ajuanijustus/gridhunt/blob/main/LICENSE)

Heterogeneous, mixed-motive predator-prey environments for multi-agent reinforcement learning
and human-agent teaming, with a [PettingZoo](https://pettingzoo.farama.org) `ParallelEnv` API.
Documentation: [ajuanijustus.github.io/gridhunt](https://ajuanijustus.github.io/gridhunt/).

- **Heterogeneous agents:** capture power, vision radius, speed and terrain restrictions are set per agent.
- **Mixed-motive scenarios:** stag hunts, toxic prey, risky targets, roles that only scout.
- **Human-agent teaming:** `gridhunt.hat` runs studies with people in or on the loop: keyboard
  control with reaction times, AI teammates, buttons that switch teams of policies, cues only
  the participant sees, questionnaires and per-participant JSON Lines logs.
- **Baselines:** IPPO, MAPPO and HAPPO over 20 seeds, reported as IQM with 95% bootstrap CIs
  ([benchmark/](https://github.com/ajuanijustus/gridhunt/blob/main/benchmark/README.md)).

## Install

```bash
pip install gridhunt            # environments only (numpy, gymnasium, pettingzoo)
pip install "gridhunt[render]"  # + pygame window, rgb_array frames and the interactive GUI
pip install "gridhunt[hat]"     # + the human-agent teaming toolkit
```

From source:

```bash
git clone https://github.com/ajuanijustus/gridhunt.git
cd gridhunt
pip install -e ".[dev]"
```

## Quick start

```python
import gridhunt

env = gridhunt.make("GridHunt-ClassicStag-v0")
obs, infos = env.reset(seed=42)
while env.agents:
    actions = {agent: env.action_space(agent).sample() for agent in env.agents}
    obs, rewards, terminations, truncations, infos = env.step(actions)
```

`gridhunt.list_envs()` lists the registered environments. Any scenario setting can be overridden,
for example `gridhunt.make("GridHunt-ToxicStag-v0", max_steps=200, reward_mode="team")`. With
PettingZoo 1.27 or later, `import gridhunt` also registers every environment with PettingZoo, so
`pettingzoo.make("parallel", "GridHunt-ClassicStag-v0")` works too.

## Human-agent teaming

```python
import gridhunt
from gridhunt.hat import ARROW_KEYS, HumanSession, Likert, Questionnaire, SessionLog
from gridhunt.policies import GreedyPolicy

env = gridhunt.make("GridHunt-ClassicStag-v0")
session = HumanSession(
    env,
    humans={"predator_0": ARROW_KEYS},              # the participant plays this hunter
    teammates={"predator_1": GreedyPolicy()},       # an AI teammate plays the other
    log=SessionLog("hat_logs", participant="P01"),
)
with session:
    for round_ in range(3):
        session.run_episode(seed=round_, max_steps=60)
        session.ask(Questionnaire([Likert("trust", "I trusted my AI teammate.")]))
```

Teammates can also be the trained benchmark agents, which run on numpy alone (no PyTorch):

```python
from gridhunt import zoo

team = zoo.load("toxic_stag", "mappo", rank="best")   # downloaded once, then cached
env = team.make_env()                                 # the scenario as it was trained
session = HumanSession(env, humans={"predator_0": ARROW_KEYS},
                       teammates={"predator_1": team, "predator_2": team})
```

[`examples/hitl_play_with_ai.py`](https://github.com/ajuanijustus/gridhunt/blob/main/examples/hitl_play_with_ai.py) is a complete version, and
[`examples/hotl_policy_delegation.py`](https://github.com/ajuanijustus/gridhunt/blob/main/examples/hotl_policy_delegation.py) is a human-on-the-loop
study in which the participant chooses which AI team plays from cues only they can see. Both run
headless with `--dry-run`.

## Scenarios

| Environment | Grid | Agents | Challenge |
|---|---|---|---|
| `GridHunt-ClassicStag-v0` | 5×5 | 2 | Stag hunt: coordinate on the stag or settle for hares |
| `GridHunt-ToxicStag-v0` | 7×7 | 3 | Only one predator can tell whether the stag is toxic |
| `GridHunt-Specialists-v0` | 10×10 | 2 | A fast scout with full vision and a strong but short-sighted bruiser |
| `GridHunt-Military-v0` | 10×10 | 2 | A UAV that can only scout guides a slow ground vehicle through terrain |
| `GridHunt-Raid-v0` | 12×12 | 3 | Targets of rising value need more raiders and may be trapped |
| `GridHunt-PigChase-v0` | 9×9 | 2 | Two predators must reach the pig together through a maze |
| `GridHunt-LandAndSea-v0` | 10×10 | 2 | A boat and a land unit, each confined to its terrain |
| `GridHunt-Swarm-v0` | 23×23 | 10 | Ten predators, three stags and five hares; crowding onto a hare costs points |

## Repository layout

| Path | Contents |
|---|---|
| [`src/gridhunt/`](https://github.com/ajuanijustus/gridhunt/tree/main/src/gridhunt) | The installable package |
| [`benchmark/`](https://github.com/ajuanijustus/gridhunt/tree/main/benchmark) | Benchmark protocol, IPPO/MAPPO/HAPPO baselines, analysis and results (not installed) |
| [`tests/`](https://github.com/ajuanijustus/gridhunt/tree/main/tests) | Library tests |
| [`examples/`](https://github.com/ajuanijustus/gridhunt/tree/main/examples) | Runnable examples |
| [`docs/`](https://github.com/ajuanijustus/gridhunt/tree/main/docs) | Documentation site sources (`mkdocs serve` to preview) |

## Citing

If you use GridHunt, please cite it; see [`CITATION.cff`](https://github.com/ajuanijustus/gridhunt/blob/main/CITATION.cff).

## License

[MIT](https://github.com/ajuanijustus/gridhunt/blob/main/LICENSE)
