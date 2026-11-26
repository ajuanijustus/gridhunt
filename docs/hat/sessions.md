# Sessions

`HumanSession` owns the study window: it draws the board and the information bar, reads the
keyboard and buttons, asks the AI teammates for their actions, steps the environment and writes the
log. Create the environment without a render mode; the session opens the window.

## Who controls which agent

Every agent must be controlled by exactly one of:

- `humans`: `{agent_id: key_map}`. A key map is `{pygame_key: action}`; `ARROW_KEYS` (Return stays
  put) and `WASD_KEYS` (Space stays put) are provided. Several participants can share a keyboard
  with different key maps.
- `teammates`: `{agent_id: policy}`, AI agents that are always in control.
- `teams`: `{name: {agent_id: policy}}`, alternative sets of AI teammates. Each team gets a button
  that puts it in control; `initial_team` (default: the first) starts.

A policy is called as `policy(observation, agent_id, env)` and returns an action. If it has a
`reset()` method, it is called at the start of every episode and when its team takes control.

## Pacing

- **Turn-based** (`step_interval=None`, the default when there are humans): every step waits until
  each human has pressed a move key. The reaction time runs from the frame appearing to the key
  press. With `turn_timeout=seconds`, a human who has not chosen in time stays still and the step
  is logged with the source `"timeout"`.
- **Timed** (`step_interval=seconds`, the default 0.2 s when there are no humans): the game advances
  on a timer. A human's latest key press within the interval is used, and a human who pressed
  nothing stays still (source `"none"`).

Pausing (the `P` key or the Pause button) stops the game; reaction times exclude the pause.
`Escape` or closing the window ends the session (`session.quit_requested` becomes true).

Other keys can be bound with `keys={pygame_key: callback}`; `callback(session)` runs when the key is
pressed, also while paused. Keys already used for moving, pausing or quitting are refused.

## Supervising: teams and buttons

```python
import gridhunt
from gridhunt.hat import Button, HumanSession
from gridhunt.policies import GreedyPolicy

agents = ["predator_0", "predator_1", "predator_2"]
env = gridhunt.make("GridHunt-ToxicStag-v0")
session = HumanSession(
    env,
    teams={
        "Stag team": {a: GreedyPolicy(prey_types=("stag",), avoid_harmful=False) for a in agents},
        "Hare team": {a: GreedyPolicy(prey_types=("hare",)) for a in agents},
    },
    buttons=[Button("Skip round", lambda s: s.end_episode("skipped"))],
    step_interval=0.15,
)
```

A `Button(label, on_press)` calls `on_press(session)` when clicked. From a callback (or between
episodes) a session can:

| Method | Effect |
|---|---|
| `set_team(name)` | put a team in control from the next step |
| `pause()`, `resume()`, `toggle_pause()` | hold or continue the game (buttons and keys keep working) |
| `set_button_enabled(label, enabled)` | grey out or restore a button; presses of a disabled one are ignored |
| `reset_board()` | place agents and prey again without starting a new episode (step count and score carry on) |
| `end_episode(reason)` | end the current episode |
| `quit()` | end the episode and the session |
| `log_event(name, **data)` | write a custom event to the log |

Callbacks and the `info` text function can read `session.episode`, `session.step`,
`session.team_return`, `session.captures`, `session.active_team`, `session.paused`,
`session.last_rewards` and `session.last_infos` (the last step's rewards and infos).

`stop_when(session)` runs after every step, so it is also the place for per-step bookkeeping, such
as updating a score that carries across episodes.

## Episodes

```python
result = session.run_episode(seed=0, max_steps=300, stop_when=lambda s: s.captures >= 50,
                             condition="berries")
```

- `max_steps` ends the episode early (the environment's own limit still applies);
- `stop_when(session)` is checked after every step;
- `start_paused=True` shows the board but holds it until `resume()` is called, for example from a
  button the participant uses to rate the situation first;
- extra keyword arguments (here `condition`) are stored in the episode's log records.

It returns an `EpisodeResult` with the steps, team return, captures, how it ended
(`"done"`, `"max_steps"`, `"stop_when"`, `"skipped"`, `"quit"`) and its duration.

## The information bar

`info` sets the bar's text: a string, or `info(session) -> str` called every frame. The default
shows the episode, step, score and the team in control. Reserve more lines with `info_lines`.

## What the participant sees

`gui_config` passes renderer settings ([rendering](../guides/rendering.md)). By default a
supervisor sees the whole board, and a single participant playing an agent with limited vision sees
only what that agent sees; set `gui_config={"vision_mask": None}` or an agent id to choose.

## Testing a study without a person

`headless=True` draws to a hidden window, and `event_source` replaces the keyboard and mouse with a
function returning pygame events for each frame. `session.ask(questionnaire, responder=...)`
answers questionnaires automatically. The examples' `--dry-run` flag uses all three, so a study
script can run in a test suite or on a server.
