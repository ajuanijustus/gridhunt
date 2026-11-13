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

### Changed
- `GridHunt-PIG-v0` is now `GridHunt-PigChase-v0`. Its dynamics are unchanged.
- `gridhunt.make()` rejects unknown settings instead of silently ignoring them, and invalid
  observation modes raise an error.
- GridHunt no longer uses the global `random` or `numpy.random` state, not even when the
  scenarios are built at import time.

### Fixed
- Pig chase: observations returned right after a capture showed the capture positions and the
  pig's random respawn cell instead of the reset start positions.
