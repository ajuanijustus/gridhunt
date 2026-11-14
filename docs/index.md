# GridHunt

Heterogeneous, mixed-motive predator-prey environments for multi-agent reinforcement learning
and human-agent teaming, with a [PettingZoo](https://pettingzoo.farama.org) `ParallelEnv` API.

<div class="grid" markdown>

![Toxic stag](assets/scenarios/toxic_stag.gif){ width="240" }
![Military](assets/scenarios/military.gif){ width="240" }
![Swarm](assets/scenarios/swarm.gif){ width="240" }

</div>

- **Heterogeneous agents.** Capture power, speed, vision radius and terrain restrictions are set
  per agent: scouts that cannot capture, strong but short-sighted bruisers, boats confined to the sea.
- **Mixed-motive scenarios.** Stag hunts, toxic prey that only one agent can recognise, trapped
  targets, crowding penalties. Rewards are individual by default, or shared with `reward_mode="team"`.
- **Human-agent teaming.** [`gridhunt.hat`](hat/index.md) runs studies with people in or on the
  loop: keyboard control with reaction times, AI teammates, buttons that switch which team of
  policies plays, cues only the participant can see, questionnaires and per-participant logs.
- **Baselines and pretrained teams.** IPPO, MAPPO and HAPPO over 20 seeds with confidence intervals
  ([benchmark](benchmark/index.md)), and the trained teams as numpy-only policies
  ([`gridhunt.zoo`](benchmark/zoo.md)).

## Install

```bash
pip install gridhunt            # environments only (numpy, gymnasium, pettingzoo)
pip install "gridhunt[render]"  # + pygame window, rgb_array frames and the interactive GUI
pip install "gridhunt[hat]"     # + the human-agent teaming toolkit
```

## Quick start

```python
import gridhunt

env = gridhunt.make("GridHunt-ClassicStag-v0")
observations, infos = env.reset(seed=42)
while env.agents:
    actions = {agent: env.action_space(agent).sample() for agent in env.agents}
    observations, rewards, terminations, truncations, infos = env.step(actions)
```

Continue with [getting started](getting-started.md), browse the [environments](environments/index.md),
or see how to [run a study](hat/index.md).
