# Session logs

```python
from gridhunt.hat import SessionLog

log = SessionLog("hat_logs", participant="P01", condition="berries",
                 metadata={"experimenter": "E1"}, log_state=True)
```

A session writes two files to the folder:

- `P01_berries_<date-time>.jsonl`: one JSON record per line, written and flushed as it happens, so
  a crash loses at most the record being written. An existing file is never overwritten.
- `P01_berries_<date-time>_meta.json`: the session's metadata, rewritten when it ends.

Pass `on_record=callable` to receive every record as it is written, for example to send event
markers to an eye tracker or a lab streaming system.

## Record types

Every record has `type`, `t` (seconds since the session started, from a monotonic clock: use it for
durations and for aligning with other recordings) and `wall_time` (Unix time).

| `type` | Fields |
|---|---|
| `session_start` | `env` (id, class, grid size, agents, observation and reward mode, episode length), `humans`, `teammates`, `teams`, `buttons`, `step_interval`, `turn_timeout` |
| `episode_start` | `episode`, `seed`, `team` in control, `entities` (each human-only cue's position and image), and any fields passed to `run_episode` |
| `step` | `episode`, `step`, `actions`, `sources`, `reaction_ms`, `team`, `rewards`, `team_reward`, `captures`, `terminations`, `truncations`, `positions` of every agent and prey, and `state` (`env.state()`, unless `log_state=False`) |
| `event` | `name`, `episode`, `step`, `episode_t` (seconds since the episode started) and its data: `key` (`agent`, `key`, `action`, `accepted`, plus `early: true` for a key pressed before the frame appeared, or just `key` for a `keys` binding), `button` (`label`; `accepted: false` when disabled), `team_switch` (`team`, `previous`), `pause`, `resume`, `reset_board` (`positions`, `entities`), `quit`, or your own from `log_event` |
| `questionnaire` | `id`, `episode`, `answers`, `response_ms` per item |
| `episode_end` | `episode`, `seed`, `steps`, `team_return`, `captures`, `end_reason`, `duration_s` |
| `session_end` | `episodes`, `quit` |

`sources` says who chose each agent's action: `"human"`, `"timeout"` (turn-based, no key in time),
`"none"` (timed, no key this step), the name of the team in control, or the teammate policy's name.
`reaction_ms` holds each human's time from the frame appearing to their first key press on that
step (`null` without a press).

## Metadata

The metadata file holds the participant and condition, the start and end times, every episode's
seed and extra fields, the session settings above, and the versions of Python, GridHunt, NumPy,
PettingZoo, Gymnasium, pygame-ce and pygame-gui. Anything passed as `metadata=` is added.

## Reading a log

```python
from gridhunt.hat import read_log

records = read_log("hat_logs/P01_berries_20261113-153456.jsonl")
steps = [r for r in records if r["type"] == "step"]
switches = [r for r in records if r["type"] == "event" and r["name"] == "team_switch"]
```

With pandas, `pandas.json_normalize(steps)` gives one row per step with columns such as
`actions.predator_0` and `reaction_ms.predator_0`.
