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

## Releasing

Releases are built and published by `.github/workflows/release.yml` with PyPI trusted publishing,
so no tokens are stored anywhere.

**Once:**

1. On [PyPI](https://pypi.org/manage/account/publishing/) and on
   [TestPyPI](https://test.pypi.org/manage/account/publishing/), add a pending trusted publisher for
   the project `gridhunt`: owner `ajuanijustus`, repository `gridhunt`, workflow `release.yml`, and
   environment `pypi` (on PyPI) or `testpypi` (on TestPyPI).
2. In the repository settings, create the environments `pypi` and `testpypi` (Settings →
   Environments), and set Settings → Pages → Source to GitHub Actions.
3. On [Zenodo](https://zenodo.org/account/settings/github/), sign in with GitHub and switch the
   repository on, so every published GitHub release is archived with a DOI.

**Each release:**

1. Set `__version__` in `src/gridhunt/__init__.py` and give the `CHANGELOG.md` section its date,
   for example `## [1.0.0] - 2026-11-20`.
2. Dry run: run the Release workflow by hand (Actions → Release → Run workflow), which publishes
   to TestPyPI, then install it in a fresh environment:
   `pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ gridhunt==<version>`.
3. Commit, then tag and push: `git tag v1.0.0 && git push origin v1.0.0`. The workflow checks
   that the tag matches `__version__` (and, for a final release, that the zoo registry lists
   models), publishes to PyPI and creates the GitHub release, which Zenodo archives.
4. Add the DOI that Zenodo mints to `CITATION.cff` and the README, and set `__version__` to the
   next development version (for example `1.0.1.dev0`).

Tags containing `rc` publish a release candidate to PyPI without a GitHub release or DOI.
