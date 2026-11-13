# GridHunt

Heterogeneous, mixed-motive predator-prey environments for multi-agent reinforcement learning
and human-agent teaming, with a [PettingZoo](https://pettingzoo.farama.org) `ParallelEnv` API.

- **Heterogeneous agents:** capture power, vision radius, speed and terrain restrictions are set per agent.
- **Mixed-motive scenarios:** stag hunts, toxic prey, risky targets, roles that only scout.
- **Human-agent teaming:** a configurable pygame interface with buttons, an information bar and
  human-only visual cues, for human-in-the-loop and human-on-the-loop studies.
- **Baselines:** IPPO, MAPPO and HAPPO over 20 seeds, reported as IQM with 95% bootstrap CIs
  ([benchmark/](benchmark/BENCHMARK.md)).

## Install

```bash
pip install gridhunt            # environments only (numpy, gymnasium, pettingzoo)
pip install "gridhunt[render]"  # + pygame window, rgb_array frames and the interactive GUI
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
| `GridHunt-Swarm-v0` | 23×23 | 10 | Many predators, shared reward and overcrowding penalties |

## Repository layout

| Path | Contents |
|---|---|
| [`src/gridhunt/`](src/gridhunt) | The installable package |
| [`benchmark/`](benchmark) | Benchmark protocol, IPPO/MAPPO/HAPPO baselines, analysis and results (not installed) |
| [`tests/`](tests) | Library tests |
| [`examples/`](examples) | Runnable examples |
| [`docs/`](docs) | Documentation site sources |

## Citing

If you use GridHunt, please cite it; see [`CITATION.cff`](CITATION.cff).

## License

[MIT](LICENSE)
