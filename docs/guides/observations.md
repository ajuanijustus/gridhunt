# Observations

Choose an encoding with `observation_mode` (each registered scenario has a default) and add extra
features with `observation_encoders`. Every observation is a flat `float32` vector.

```python
import gridhunt

env = gridhunt.make("GridHunt-Military-v0", observation_mode="flat_coords")
print(env.observation_space("uav_0").shape)
```

## Modes

| Mode | Contents | Same for every agent? |
|---|---|---|
| `coords` | x, y of every prey (sorted by id; -1, -1 once removed), of every predator (sorted by id), and the mean x, y of each terrain type | yes |
| `flat_coords` | x, y of every prey and every predator, as in `coords` | yes |
| `one_hot` | one-hot grids of every predator, every prey and every terrain type, then the agent's own last reward and last action | see below |
| `toxic` | Toxic stag only: egocentric grids of the other agents and the prey, a toxicity flag and every agent's last reward | no |

**`one_hot` grids.** When no agent has limited vision, the grids are absolute: one `width × height`
grid per entity (cell `(x, y)` at index `x * height + y`), identical for every agent apart from the
last reward and action. When any agent has limited vision, every agent instead gets egocentric
grids of `(2·width - 1) × (2·height - 1)` cells centred on itself, and an agent with limited vision
sees entities and terrain only within its radius. The last reward is divided by the largest prey
reward and clipped to [-1, 1]; the last action is a one-hot vector of length 5.

!!! warning "Agent identity"
    `coords` and `flat_coords` give every agent the same vector, so a single network shared by
    several agents cannot tell which agent it is acting for. Train one network per agent, or append
    an agent id yourself (the benchmark's MAPPO appends a one-hot agent index).

## Encoders

`observation_encoders` appends features to any mode:

| Encoder | Adds |
|---|---|
| `"agent_centric"` | a one-hot code of the agent's role (8 values). Roles are predator and trainable prey unless `agent_role_map` assigns them, for example `{"scouts": ["scout_0"], "bruisers": ["bruiser_0"]}` |
| `"action_mask"` | 5 values: whether each move stays on the grid and out of impassable terrain |

With `use_action_mask=True` the same masks are also reported in
`infos["__common__"]["action_masks"]`. The mask considers the grid edges, impassable terrain and
terrain types restricted by the terrain configuration's `restricted_to`; it does not account for an
agent's own `restricted_terrains`, terrain immunity or speed.

## Global state

`env.state()` is a compact vector for centralised critics, all values in [0, 1]:

- every agent's x, y divided by the grid size;
- every prey's x, y, whether it is present and whether it is toxic;
- a one-hot grid for each terrain type.

With the communication channel, the last message tokens are appended too.
