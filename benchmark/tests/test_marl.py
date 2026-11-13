"""Unit tests for the baseline implementations in benchmark/marl."""

import dataclasses
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "benchmark"))

from marl.ppo import ALGORITHMS, MultiAgentPPO, PPOConfig, ValueNorm, collect_rollout, compute_gae, evaluate  # noqa: E402
from marl.vec_env import VecEnv  # noqa: E402
from protocol import make_env  # noqa: E402

SMALL = PPOConfig(num_envs=2, rollout_steps=16, epochs=2, minibatches=2)


def test_gae_matches_hand_computation():
    gamma, lam = 0.9, 0.5
    rewards = np.array([[1.0], [0.0], [2.0]], dtype=np.float32)
    values = np.array([[0.5], [1.0], [1.5]], dtype=np.float32)
    next_values = np.array([[1.0], [1.5], [3.0]], dtype=np.float32)
    dones = np.zeros((3, 1), dtype=np.float32)
    adv, ret = compute_gae(rewards, values, next_values, dones, gamma, lam)
    d2 = 2.0 + 0.9 * 3.0 - 1.5
    d1 = 0.0 + 0.9 * 1.5 - 1.0
    d0 = 1.0 + 0.9 * 1.0 - 0.5
    expected = [d0 + 0.45 * (d1 + 0.45 * d2), d1 + 0.45 * d2, d2]
    np.testing.assert_allclose(adv[:, 0], expected, rtol=1e-6)
    np.testing.assert_allclose(ret, adv + values)


def test_gae_cuts_trace_at_time_limit_but_bootstraps():
    rewards = np.array([[1.0], [1.0]], dtype=np.float32)
    values = np.array([[0.0], [0.0]], dtype=np.float32)
    # Step 0 hits the time limit; its final-state value (10) is bootstrapped
    next_values = np.array([[10.0], [0.0]], dtype=np.float32)
    dones = np.array([[1.0], [0.0]], dtype=np.float32)
    adv, _ = compute_gae(rewards, values, next_values, dones, gamma=0.5, gae_lambda=1.0)
    assert adv[0, 0] == pytest.approx(1.0 + 0.5 * 10.0)  # no leak from step 1
    assert adv[1, 0] == pytest.approx(1.0)


def test_value_norm_round_trip():
    norm = ValueNorm()
    data = np.random.default_rng(0).normal(5.0, 3.0, size=10_000)
    norm.update(data[:5000])
    norm.update(data[5000:])
    assert norm.mean == pytest.approx(data.mean(), rel=1e-3)
    assert norm.var == pytest.approx(data.var(), rel=1e-2)
    np.testing.assert_allclose(norm.denormalize(norm.normalize(data)), data, rtol=1e-6)


def _learner(algo, scenario="toxic_stag", seed=0, config=SMALL):
    env = make_env(scenario)
    agents = env.possible_agents
    obs_dims = {a: env.observation_space(a).shape[0] for a in agents}
    action_dims = {a: env.action_space(a).n for a in agents}
    return MultiAgentPPO(algo, agents, obs_dims, action_dims, env.state_space.shape[0], config, seed=seed)


@pytest.mark.parametrize("algo", ALGORITHMS)
def test_architecture(algo):
    learner = _learner(algo)
    actors = {id(a) for a in learner.actors.values()}
    assert len(actors) == (1 if algo == "mappo" else len(learner.agents))
    assert list(learner.critics) == (["central"] if algo in ("mappo", "happo") else learner.agents)


@pytest.mark.parametrize("algo", ALGORITHMS)
def test_update_runs_and_is_deterministic(algo):
    def run():
        torch.manual_seed(0)
        learner = _learner(algo)
        envs = VecEnv(lambda: make_env("toxic_stag"), SMALL.num_envs, seed=0)
        obs, state = envs.reset()
        batch, *_ = collect_rollout(learner, envs, obs, state)
        stats = learner.update(batch)
        return stats, [p.detach().clone() for p in next(iter(learner.actors.values())).parameters()]

    stats_a, params_a = run()
    stats_b, params_b = run()
    assert stats_a == stats_b
    assert all(torch.equal(a, b) for a, b in zip(params_a, params_b))
    assert all(np.isfinite(v) for v in stats_a.values())


def test_sampled_evaluation_is_reproducible_and_leaves_training_rng_alone():
    learner = _learner("mappo", scenario="classic_stag")
    env_fn = lambda: make_env("classic_stag")  # noqa: E731
    torch.manual_seed(123)
    rng_before = torch.get_rng_state()
    first = evaluate(learner, env_fn, episodes=3, seed_offset=1000)
    assert torch.equal(torch.get_rng_state(), rng_before)
    assert evaluate(learner, env_fn, episodes=3, seed_offset=1000) == first
    greedy = evaluate(learner, env_fn, episodes=3, seed_offset=1000, greedy=True)
    assert len(greedy[0]) == 3 and all(np.isfinite(greedy[0]))


def test_happo_factor_starts_at_one_and_tracks_policy_change(monkeypatch):
    """The first updated agent sees the plain advantage; later ones see it reweighted."""
    learner = _learner("happo")
    envs = VecEnv(lambda: make_env("toxic_stag"), SMALL.num_envs, seed=0)
    obs, state = envs.reset()
    batch, *_ = collect_rollout(learner, envs, obs, state)
    seen = []
    original = learner._train_actor

    def record(key, agents, obs_, actions, old_logp, advantages, stats):
        seen.append(advantages[agents[0]].copy())
        return original(key, agents, obs_, actions, old_logp, advantages, stats)

    monkeypatch.setattr(learner, "_train_actor", record)
    learner.update(batch)
    adv = batch["advantages"]["central"]
    normalized = (adv - adv.mean()) / (adv.std() + 1e-8)
    np.testing.assert_allclose(seen[0], normalized, rtol=1e-5)
    assert not np.allclose(seen[1], normalized)  # reweighted by the first agent's update


def test_training_resumes_from_checkpoint(tmp_path):
    """Kill a run after its first checkpoint; rerunning the same command resumes and finishes."""
    cmd = [sys.executable, str(REPO_ROOT / "benchmark/train_marl.py"), "--scenario", "classic_stag",
           "--algo", "mappo", "--seed", "1", "--num-envs", "2", "--rollout-steps", "16",
           "--budget", "20000", "--eval-every", "10000", "--eval-episodes", "1",
           "--final-eval-episodes", "1", "--checkpoint-every", "5", "--out", str(tmp_path)]
    job = subprocess.Popen(cmd, cwd=REPO_ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    deadline = time.time() + 120
    while not (tmp_path / "checkpoint.pt").exists() and time.time() < deadline:
        time.sleep(0.05)
    job.kill()
    job.wait()
    assert (tmp_path / "checkpoint.pt").exists()
    assert not (tmp_path / "final.json").exists()

    result = subprocess.run(cmd, check=True, capture_output=True, text=True, cwd=REPO_ROOT)
    assert "Resuming from iteration" in result.stdout
    assert (tmp_path / "final.json").exists() and not (tmp_path / "checkpoint.pt").exists()
    final = json.loads((tmp_path / "final.json").read_text())
    assert 0.0 <= final["adv_floor_active_frac"] <= 1.0
    assert "adv_floor_active" in (tmp_path / "progress.csv").read_text().splitlines()[0]


def test_advantage_std_floor_only_binds_when_the_batch_has_almost_no_spread():
    learner = _learner("mappo")
    learner.value_norms["central"].var = 100.0  # return std 10 -> floor 0.05 * 10 = 0.5
    rng = np.random.default_rng(0)
    stats = {"adv_floor_active": []}
    noise = learner._standardize(rng.normal(0, 0.01, 4096), "central", stats)
    assert noise.std() == pytest.approx(0.01 / 0.5, rel=0.1)  # critic noise stays small
    spread = learner._standardize(rng.normal(0, 5.0, 4096), "central", stats)
    assert spread.std() == pytest.approx(1.0, rel=0.01)  # ordinary batches: plain standardization
    assert stats["adv_floor_active"] == [1.0, 0.0]


def test_make_jobs_refuses_runs_trained_with_other_settings(tmp_path):
    run = tmp_path / "raid" / "ippo" / "seed_0"
    run.mkdir(parents=True)
    old_config = {**dataclasses.asdict(PPOConfig()), "adv_std_floor": 0.0}
    (run / "final.json").write_text(json.dumps({"budget": 7_000_000, "config": old_config}))
    cmd = [sys.executable, str(REPO_ROOT / "benchmark/slurm/make_jobs.py"), "--scenarios", "raid",
           "--algos", "ippo", "--seeds", "0-1", "--results", str(tmp_path), "--out", str(tmp_path / "jobs.txt")]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO_ROOT)
    assert result.returncode != 0 and "adv_std_floor" in result.stderr

    (run / "final.json").write_text(json.dumps({"budget": 7_000_000, "config": dataclasses.asdict(PPOConfig())}))
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO_ROOT, check=True)
    assert "Wrote 1 jobs" in result.stdout and "1 already finished" in result.stdout
