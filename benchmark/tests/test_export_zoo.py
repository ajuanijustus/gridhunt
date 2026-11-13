"""Exported zoo models must act exactly like the PyTorch learners they came from."""

import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).parent.parent))

from export_zoo import export_learner, make_kwargs, predator_changes  # noqa: E402
from marl.ppo import MultiAgentPPO, PPOConfig  # noqa: E402
from protocol import SCENARIOS, make_env  # noqa: E402

from gridhunt import zoo  # noqa: E402


def _learner(scenario, algo, seed=0):
    env = make_env(scenario)
    agents = env.possible_agents
    learner = MultiAgentPPO(
        algo, agents, {a: env.observation_space(a).shape[0] for a in agents},
        {a: env.action_space(a).n for a in agents}, env.state_space.shape[0], PPOConfig(), seed=seed,
    )
    # Training would move the weights away from the initialisation; perturb them so the test
    # does not depend on orthogonal-init structure
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for actor in {id(a): a for a in learner.actors.values()}.values():
            for parameter in actor.parameters():
                parameter.add_(torch.randn(parameter.shape, generator=generator) * 0.3)
    return learner, env


def _export(learner, scenario, path):
    obs_dims = {a: learner.actors[a][0].in_features - (len(learner.agents) if learner.shared_actor else 0)
                for a in learner.agents}
    return export_learner(learner, path, {
        "scenario": scenario, "seed": 0, "rank": "median", "env_id": SCENARIOS[scenario]["env_id"],
        "make_kwargs": make_kwargs(scenario), "predator_changes": predator_changes(scenario),
        "obs_dims": obs_dims, "action_dims": {a: learner.actors[a][-1].out_features for a in learner.agents},
    })


@pytest.mark.parametrize(("scenario", "algo"), [
    ("classic_stag", "ippo"), ("toxic_stag", "mappo"), ("military", "happo"), ("specialists_comm", "mappo"),
    ("swarm", "ippo"),
])
def test_exported_logits_and_greedy_actions_match_torch(tmp_path, scenario, algo):
    learner, env = _learner(scenario, algo)
    _export(learner, scenario, tmp_path / "model.npz")
    team = zoo.load_file(tmp_path / "model.npz")

    observations, _ = env.reset(seed=0)
    for _ in range(40):
        if not env.agents:
            break
        batched = {a: observations[a][None].astype(np.float32) for a in learner.agents}
        torch_greedy, _ = learner.act(batched, greedy=True)
        for agent in learner.agents:
            with torch.no_grad():
                expected = learner._distribution(agent, batched[agent]).logits[0].numpy()
            # Categorical normalises logits by a constant; compare after the same shift. NumPy and
            # PyTorch sum float32 products in different orders, so allow rounding-sized differences
            got = team.logits(agent, observations[agent])
            np.testing.assert_allclose(got - got.max(), expected - expected.max(), rtol=1e-5, atol=1e-4)
            assert team.act_agent(agent, observations[agent], greedy=True) == int(torch_greedy[agent][0])
        observations, *_ = env.step(team.act(observations, greedy=True))


@pytest.mark.parametrize("scenario", list(SCENARIOS))
def test_zoo_environment_is_the_benchmark_environment(tmp_path, scenario):
    learner, _ = _learner(scenario, "ippo")
    _export(learner, scenario, tmp_path / "model.npz")
    zoo_env = zoo.load_file(tmp_path / "model.npz").make_env()
    bench_env = make_env(scenario)
    zoo_obs, _ = zoo_env.reset(seed=11)
    bench_obs, _ = bench_env.reset(seed=11)
    rng = np.random.default_rng(0)
    for _ in range(30):
        for agent in bench_env.possible_agents:
            np.testing.assert_array_equal(zoo_obs[agent], bench_obs[agent])
        if not bench_env.agents:
            break
        actions = {a: int(rng.integers(bench_env.action_space(a).n)) for a in bench_env.agents}
        zoo_obs, zoo_rewards, *_ = zoo_env.step(actions)
        bench_obs, bench_rewards, *_ = bench_env.step(actions)
        assert zoo_rewards == bench_rewards


def test_variant_predator_changes():
    assert predator_changes("classic_stag") == {}
    assert predator_changes("specialists_full_vision") == {"bruiser_0": {"limited_vision_radius": 10}}
    blind = predator_changes("toxic_stag_blind")
    assert set(blind) == {"predator_0"} and "toxicity_aware" not in blind["predator_0"]["status"]
