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

### Fixed
- Pig chase: observations returned right after a capture showed the capture positions and the
  pig's random respawn cell instead of the reset start positions.
