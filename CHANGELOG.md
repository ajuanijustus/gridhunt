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
  choice, free text and instruction screens). Examples for a human-in-the-loop game and a
  human-on-the-loop policy delegation study.
- `gridhunt.policies`: the scripted policies, usable as teammates; `GreedyPolicy` can be limited
  to some prey types and told to ignore hidden information (`prey_types`, `avoid_harmful`).
- `gridhunt.zoo`: the benchmark's trained IPPO, MAPPO and HAPPO teams (median and best seed per
  scenario) as numpy-only policies, downloaded on first use, checked against SHA-256 hashes and
  cached. A team plays on its own (`team.act(observations)`) or as teammates in a `HumanSession`,
  and `team.make_env()` rebuilds the exact scenario it was trained in.
- `env.render()` and `env.close()`, a working `render_mode="rgb_array"`, and
  `gui_config["vision_mask"]` to choose whose view the renderer shows.
- `env.set_entities()` and `env.set_prey()` to change human-only cues and prey settings between
  episodes.

### Changed
- `GridHunt-PIG-v0` is now `GridHunt-PigChase-v0`. Its dynamics are unchanged.
- `gridhunt.make()` rejects unknown settings instead of silently ignoring them, and invalid
  observation modes raise an error.
- GridHunt no longer uses the global `random` or `numpy.random` state, not even when the
  scenarios are built at import time.
- Human-only entities are placed with their own random generator and keep a position they are
  given, so adding them never changes what the agents experience. No registered environment has
  entities, so their dynamics are unchanged.

### Fixed
- `env.render()` raised `NotImplementedError`, and `render_mode="rgb_array"` was accepted but
  drew nothing.
- Pig chase: observations returned right after a capture showed the capture positions and the
  pig's random respawn cell instead of the reset start positions.
