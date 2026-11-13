# Contributing to GridHunt

Thanks for your interest! Bug reports, new scenarios, documentation and examples are all welcome.

## Development setup

```bash
git clone https://github.com/ajuanijustus/gridhunt.git
cd gridhunt
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest tests
```

Benchmark code lives in `benchmark/` and needs the extra dependencies:

```bash
pip install -e ".[dev,benchmark]"
pytest benchmark/tests
```

Render tests run headless with `SDL_VIDEODRIVER=dummy`.

## Guidelines

- **Environment dynamics are versioned.** Any change to an environment's transitions, rewards or
  observations must bump its ID suffix (for example `-v0` to `-v1`) and be noted in
  `CHANGELOG.md`, so published results stay comparable. `tests/test_regression.py` replays
  recorded trajectories of every environment and fails on any such change; record references
  for a new version with `tests/regression/record_reference.py`, never to make the test pass.
- Use the environment's own seeded RNG (`self._np_rng`), never the global `random` or
  `np.random` state.
- New environments should pass `pettingzoo.test.parallel_api_test` and `parallel_seed_test`;
  add them to the tests.
- Format and lint with `ruff`.
- Open an issue first for large changes, so we can agree on the design.

## Reporting bugs

Please include the GridHunt, PettingZoo and Python versions, the environment ID and the seed, and a
minimal script that reproduces the problem.
