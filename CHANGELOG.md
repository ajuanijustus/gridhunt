# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[semantic versioning](https://semver.org/). Environment IDs carry their own version suffix
(`-v0`), which increases whenever an environment's dynamics, rewards or observations change.

## [1.0.0] - Unreleased

First public release.

### Added
- Installable package (`pip install gridhunt`) with optional `render`, `hat`, `benchmark`, `dev`
  and `docs` extras; `import gridhunt` no longer requires pygame.
- Benchmark: IPPO, MAPPO and HAPPO over 20 seeds with a fixed protocol, sampled-action evaluation,
  IQM with stratified bootstrap CIs, and per-scenario budgets from a 20M-step pilot.
- Optional explicit communication channel (`gridhunt.make(..., communication=k)`).
- Scripted prey motion (`prey_motion="flee"`) and a team reward mode (`reward_mode="team"`).
- `gridhunt.list_envs()`, and registration with PettingZoo's environment registry
  (`pettingzoo.make("parallel", env_id)`, PettingZoo 1.27 or later).
- Tests that replay fixed-action trajectories of every environment and compare them with
  recorded references, so dynamics cannot change without a version bump.
- `gridhunt.hat`, a human-agent teaming toolkit: `HumanSession` (keyboard control with reaction
  times, AI teammates, team-switching and custom buttons, pausing, timers or turn-based steps),
  `SessionLog` (JSON Lines per participant session plus metadata) and `Questionnaire` (Likert,
  slider, choice, free text and instruction screens). Examples for a human-in-the-loop game and a
  human-on-the-loop policy delegation study.
- `HumanSession`: episodes that start paused until a button resumes them (`start_paused`),
  `pause()`/`resume()`, extra key bindings (`keys`), `reset_board()` to place everyone again
  mid-episode, `set_button_enabled()` and `last_rewards`. Every logged event carries `episode_t`, the time since
  its episode started. `Scale` questionnaire items for 0-100 style ratings.
- `gridhunt.policies`: the scripted policies, usable as teammates; `GreedyPolicy` can be limited
  to some prey types and told to ignore hidden information (`prey_types`, `avoid_harmful`).
- `gridhunt.zoo`: the benchmark's trained IPPO, MAPPO and HAPPO teams (median and best seed per
  scenario) as numpy-only policies, downloaded on first use, checked against SHA-256 hashes and
  cached. A team plays on its own (`team.act(observations)`) or as teammates in a `HumanSession`,
  and `team.make_env()` rebuilds the exact scenario it was trained in.
- Continuous integration on Linux and macOS with Python 3.10 to 3.13, the oldest supported
  dependency versions and a base install without rendering, and a release workflow that publishes
  to PyPI with trusted publishing.
- Documentation site: scenario pages generated from the registered configurations, guides,
  the human-agent teaming toolkit and its log format, benchmark results and the API reference.
- Benchmark reproduction: `benchmark/requirements.txt` with the exact library versions behind
  the published results, `package_runs.py` to check and package the runs for archiving, and the
  interdependence ablation in the committed summary.
- `env.render()` and `env.close()`, a working `render_mode="rgb_array"`, and
  `gui_config["vision_mask"]` to choose whose view the renderer shows.
- `env.set_entities()` and `env.set_prey()` to change human-only cues and prey settings between
  episodes.

### Changed
- The SLURM scripts work on any cluster: its account, log folder and environment set-up live in
  `benchmark/slurm/cluster.env`, and scripts are submitted through `benchmark/slurm/submit.sh`.
- The benchmark documentation is now `benchmark/README.md`.
- Rendering draws every board at a cell size that fills about 640 pixels, with the same margin on
  every side and grid lines of one width, instead of scaling 32-pixel cells by whole numbers.
- Sprites are stored at 128 × 128 pixels, which shrinks the package from 4 MB to 0.4 MB without
  visibly changing the rendering; an unused sprite was removed.
- `GridHunt-PIG-v0` is now `GridHunt-PigChase-v0`. Its dynamics are unchanged.
- `gridhunt.make()` rejects unknown settings instead of silently ignoring them, and invalid
  observation modes raise an error.
- GridHunt no longer uses the global `random` or `numpy.random` state, not even when the
  scenarios are built at import time.
- Human-only entities are placed with their own random generator and keep a position they are
  given, so adding them never changes what the agents experience. No registered environment has
  entities, so their dynamics are unchanged.

### Fixed
- Moves of more than one cell checked only the landing cell, so fast agents could jump walls; they
  now move one cell at a time. Slow terrain can be entered (it ends the move) instead of blocking
  agents of speed 1. No registered scenario is affected.
- Action masks ignored an agent's own terrain restrictions, terrain immunity and speed, and the
  terrain setting `restricted_to` affected only masks, not movement. Movement and masks now
  follow one set of rules, and `env.action_mask(agent_id)` returns an agent's mask.
- `CommunicationWrapper.unwrapped` returned the wrapper instead of the environment underneath.
- `env.render()` raised `NotImplementedError`, and `render_mode="rgb_array"` was accepted but
  drew nothing.
- Pig chase: observations returned right after a capture showed the capture positions and the
  pig's random respawn cell instead of the reset start positions.
