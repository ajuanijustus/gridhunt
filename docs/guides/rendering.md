# Rendering

Rendering needs the `render` extra: `pip install "gridhunt[render]"`.

| `render_mode` | `env.render()` |
|---|---|
| `None` (default) | does nothing |
| `"console"` | prints the grid as text |
| `"rgb_array"` | returns the frame as an RGB array of shape (height, width, 3); no window opens |
| `"gui"` | draws to a pygame window |

```python
import gridhunt

env = gridhunt.make("GridHunt-Raid-v0", render_mode="rgb_array")
env.reset(seed=0)
frames = [env.render()]
while env.agents:
    env.step({agent: env.action_space(agent).sample() for agent in env.agents})
    frames.append(env.render())
env.close()
```

`rgb_array` works without a display (on Linux servers it falls back to SDL's dummy video driver),
which makes it suitable for recording videos on a cluster.

## Window settings

`gui_config` controls what is drawn around the board:

| Key | Meaning |
|---|---|
| `info_bar` | show a bar under the board with text and buttons |
| `buttons` | `[{"text": "Pause"}, ...]`: buttons in the bar |
| `buttons_per_row` | wrap buttons after this many |
| `dropdowns` | `[{"options": ["a", "b"], "width": 120}]` |
| `info_text_lines` | lines reserved for the bar's text, which is `env.info` if set, the score otherwise |
| `vision_mask` | darken what an agent cannot see: `"auto"` (default, the first predator with limited vision), `None` (show the whole board) or an agent id |

```python
import gridhunt

env = gridhunt.make("GridHunt-Specialists-v0", render_mode="rgb_array",
                    gui_config={"vision_mask": "bruiser_0"})   # show what the bruiser sees
```

For interactive use with buttons, keyboard control, logging and questionnaires, use
[`HumanSession`](../hat/sessions.md), which manages the window and its events.
