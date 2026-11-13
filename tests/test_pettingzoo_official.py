"""
PettingZoo's own compliance checks, run on every registered GridHunt environment,
plus a cross-process determinism check.

parallel_seed_test only compares two envs inside one process, so it cannot catch
randomness drawn at import time or from global RNGs; the cross-process test runs
the same seeded rollout in fresh interpreters and compares the results.
"""

import hashlib
import subprocess
import sys
import warnings
from pathlib import Path

import pytest
from pettingzoo.test import parallel_api_test, parallel_seed_test

import gridhunt
from gridhunt import _BENCHMARKS

ENV_NAMES = sorted(_BENCHMARKS)
REPO_ROOT = Path(__file__).parent.parent


def _cycles(name):
    """Enough steps to run past the first truncation."""
    return gridhunt.make(name).max_steps + 50


@pytest.mark.parametrize("name", ENV_NAMES)
def test_parallel_api(name):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        parallel_api_test(gridhunt.make(name), num_cycles=_cycles(name))


@pytest.mark.parametrize("name", ENV_NAMES)
def test_parallel_seed(name):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        parallel_seed_test(lambda: gridhunt.make(name), num_cycles=_cycles(name))


ROLLOUT = """
import hashlib, random, sys
import numpy as np
import gridhunt
env = gridhunt.make(sys.argv[1])
obs, _ = env.reset(seed=123)
actions_rng = random.Random(7)
digest = hashlib.sha256()
for _ in range(300):
    actions = {a: actions_rng.randrange(5) for a in env.possible_agents}
    obs, rewards, terminations, truncations, infos = env.step(actions)
    for agent in sorted(obs):
        digest.update(np.asarray(obs[agent], dtype=np.float32).tobytes())
        digest.update(repr(float(rewards[agent])).encode())
    if not env.agents:
        obs, _ = env.reset()
print(digest.hexdigest())
"""


def _rollout_digest(name):
    result = subprocess.run(
        [sys.executable, "-c", ROLLOUT, name],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip().splitlines()[-1]


@pytest.mark.parametrize("name", ENV_NAMES)
def test_same_seed_same_trajectory_across_processes(name):
    digests = {_rollout_digest(name) for _ in range(3)}
    assert len(digests) == 1, f"{name}: seeded rollouts differ across processes"
