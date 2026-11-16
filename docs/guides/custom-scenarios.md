# Custom scenarios

A scenario is a grid, a list of predators (the agents), a list of prey and optionally terrain and
human-only entities. Override parts of a registered scenario through `gridhunt.make`, or build one
from scratch with a scenario class.

## Agents and prey

```python
import gridhunt
from gridhunt import Hare, Predator, Stag
from gridhunt.scenarios import StagHuntScenario

env = StagHuntScenario(
    grid_size=(8, 8),
    max_steps=200,
    predators=[
        Predator("scout", capture_power=1, speed=2),                       # fast, sees everything
        Predator("tank", capture_power=2, limited_vision_radius=2,
                 status={"limited_vision": True}),                         # strong, short-sighted
    ],
    prey=[Stag("stag_0", capture_power_required=2), Hare("hare_0"), Hare("hare_1")],
    observation_mode="one_hot",
)
observations, infos = env.reset(seed=0)
```

**Predator** settings:

| Setting | Meaning |
|---|---|
| `capture_power` | added up over the predators on a prey's cell; 0 means the agent can never capture |
| `speed` | cells moved per step |
| `limited_vision_radius` with `status={"limited_vision": True}` | the agent observes only cells within this many steps (an egocentric one-hot grid) |
| `immune_to_terrain` | terrain never blocks or slows the agent |
| `restricted_terrains` | terrain types the agent can never enter (immunity does not override this) |
| `position` | a fixed start cell (random otherwise) |
| `render_sprite` | the image drawn for it (`gridhunt.assets` has several) |

**Prey** classes: `Stag` (worth 5, needs capture power 2), `Hare` (worth 1, power 1, crowding costs
0.5), `ToxicStag` (worth 5, or -5 when toxic), `Building` (Raid's targets, with trap probability and
penalty, never reappear) and `Pig`. The base class `Prey` takes every setting directly: `reward`,
`capture_power_required`, `respawn`, `overcrowding_penalty`, `speed` and more.

A capture happens when the predators on a prey's cell together have at least the capture power it
needs. With more than needed, the prey's overcrowding penalty is subtracted from its reward.
Prey with `respawn=True` reappear on a random free cell.

Prey stay still unless `prey_motion` is `"random"` or `"flee"` (away from the nearest predator);
`prey_move_probability` sets how often they move.

An agent with speed above 1 moves one cell at a time and stops at the grid edge, before any cell it
cannot enter, and after entering slow terrain.

## Terrain

Terrain types are given as a dictionary. Each type has a passability (the probability that a move
into it succeeds; 0 is a wall), an optional speed multiplier (below 1, entering the terrain ends
the move) and a generation rule: a fixed list of cells, or random blobs or walls that are generated
anew every episode. `restricted_to` keeps agents of the given roles out of a terrain type
altogether (roles come from `agent_role_map`; by default predators are role 0 and trainable prey
role 1).

```python
import gridhunt

terrain = {
    "swamp": {
        "passability": 0.5,
        "generation": {"type": "blob", "count": 2, "blob_size": 2, "density": 0.5, "min_distance": 2},
    },
    "obstacle": {
        "passability": 0.0,
        "generation": {"type": "line", "count": 2, "length": 5, "orientation": "both", "min_distance": 2},
    },
    "sea": {
        "generation": {"type": "predefined", "positions": [(0, y) for y in range(10)]},
    },
}
env = gridhunt.make("GridHunt-Military-v0", terrain_config=terrain)
```

The types `mountain`, `swamp`, `obstacle`, `land` and `sea` have built-in images; pass `"sprite"`
with an image path for others.

## Human-only entities

Entities are drawn for human viewers but never appear in observations, `env.state()` or the
dynamics. They have their own random generator, so adding them never changes what the agents
experience. Entities without a position are placed on a random cell at every reset.

```python
import gridhunt
from gridhunt import Entity
from gridhunt.assets import YELLOW_BERRY_SPRITE

env = gridhunt.make("GridHunt-ToxicStag-v0",
                    entities=[Entity(f"berry_{i}", render_sprite=YELLOW_BERRY_SPRITE) for i in range(3)])
env.set_entities([Entity("marker", position=(0, 0))])   # takes effect at the next reset
```

`env.set_prey(...)` similarly changes prey settings between episodes (for example a stag's toxic
probability), as long as the prey ids stay the same.

## Communication

`gridhunt.make(..., communication=k)` adds a cheap-talk channel: each action becomes a move plus one
of `k` message tokens (`Discrete(5 * k)`), and every agent's next observation ends with all agents'
tokens. The tokens have no meaning to the environment; agents have to learn one.

```python
import gridhunt

env = gridhunt.make("GridHunt-Specialists-v0", communication=4)
print(env.action_space("scout_0"))   # Discrete(20): move = action % 5, token = action // 5
```

## Versioning

Environment IDs carry a version suffix (`-v0`). Any change to a registered scenario's dynamics,
rewards or observations gets a new suffix, so published results stay comparable; a test replays
recorded trajectories of every environment to enforce this.
