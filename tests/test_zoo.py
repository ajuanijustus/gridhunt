"""gridhunt.zoo: the numpy model format, inference, downloads and caching (numpy only, no torch)."""

import hashlib
import json

import numpy as np
import pytest

import gridhunt
from gridhunt import zoo


def _write_model(path, algo="ippo", env_id="GridHunt-ClassicStag-v0", make_kwargs=None, seed=0,
                 predator_changes=None, format_version=zoo.FORMAT_VERSION, hidden=(8, 8)):
    """A random two-hidden-layer team in the export format; returns (path, weights)."""
    make_kwargs = {"max_steps": 200, "reward_mode": "team"} if make_kwargs is None else make_kwargs
    env = gridhunt.make(env_id, **make_kwargs)
    agents = list(env.possible_agents)
    obs_dims = {a: int(env.observation_space(a).shape[0]) for a in agents}
    action_dims = {a: int(env.action_space(a).n) for a in agents}
    shared = algo == "mappo"
    rng = np.random.default_rng(seed)
    arrays, weights = {}, {}
    keys = {a: "shared" if shared else a for a in agents}
    for key in set(keys.values()):
        agent = agents[0] if shared else key
        sizes = [obs_dims[agent] + (len(agents) if shared else 0), *hidden, action_dims[agent]]
        layers = []
        for i, (n_in, n_out) in enumerate(zip(sizes[:-1], sizes[1:], strict=True)):
            w = rng.normal(0, 0.5, (n_out, n_in)).astype(np.float32)
            b = rng.normal(0, 0.1, n_out).astype(np.float32)
            arrays[f"actor/{key}/{i}/weight"], arrays[f"actor/{key}/{i}/bias"] = w, b
            layers.append((w, b))
        weights[key] = layers
    meta = {
        "format_version": format_version, "scenario": "classic_stag", "algo": algo, "seed": seed, "rank": "median",
        "agents": agents, "actor_keys": keys, "shared_actor": shared, "activation": "tanh",
        "env_id": env_id, "make_kwargs": make_kwargs, "predator_changes": predator_changes or {},
        "obs_dims": obs_dims, "action_dims": action_dims, "evaluation": {"sampled_return_mean": 1.0},
    }
    np.savez_compressed(path, meta=np.array(json.dumps(meta)), **arrays)
    return path, weights


def _reference_logits(weights, x):
    for w, b in weights[:-1]:
        x = np.tanh(x @ w.T + b)
    w, b = weights[-1]
    return x @ w.T + b


@pytest.mark.parametrize("algo", ["ippo", "mappo"])
def test_logits_match_the_stored_network(tmp_path, algo):
    path, weights = _write_model(tmp_path / "m.npz", algo=algo)
    team = zoo.load_file(path)
    env = team.make_env()
    observations, _ = env.reset(seed=0)
    for index, agent in enumerate(team.agents):
        x = observations[agent].astype(np.float32)
        if algo == "mappo":
            expected = _reference_logits(weights["shared"], np.concatenate([x, np.eye(len(team.agents))[index]]))
        else:
            expected = _reference_logits(weights[agent], x)
        np.testing.assert_allclose(team.logits(agent, x), expected, rtol=1e-5, atol=1e-5)
        batch = np.stack([x, x])
        assert team.logits(agent, batch).shape == (2, env.action_space(agent).n)


def test_greedy_is_deterministic_and_sampling_is_seeded(tmp_path):
    path, _ = _write_model(tmp_path / "m.npz")
    env = gridhunt.make("GridHunt-ClassicStag-v0", max_steps=200, reward_mode="team")
    observations, _ = env.reset(seed=1)

    def actions(rng, greedy):
        team = zoo.load_file(path, rng=rng)
        return [team.act(observations, greedy=greedy) for _ in range(20)]

    assert actions(0, True) == actions(1, True)
    assert all(a == actions(0, True)[0] for a in actions(0, True))
    assert actions(5, False) == actions(5, False)
    assert actions(5, False) != actions(6, False)


def test_a_team_plays_episodes_and_works_as_a_teammate(tmp_path):
    path, _ = _write_model(tmp_path / "m.npz")
    team = zoo.load_file(path, rng=0)
    env = team.make_env(max_steps=30)
    observations, _ = env.reset(seed=0)
    steps = 0
    while env.agents:
        actions = {a: team(observations[a], a, env) for a in env.agents}
        assert all(env.action_space(a).contains(actions[a]) for a in actions)
        observations, *_ = env.step(actions)
        steps += 1
    assert steps == 30
    assert "ippo:classic_stag" in repr(team)


def test_make_env_applies_variant_predator_changes_and_rejects_other_observations(tmp_path):
    path, _ = _write_model(tmp_path / "m.npz", env_id="GridHunt-Specialists-v0",
                           predator_changes={"bruiser_0": {"limited_vision_radius": 10}})
    team = zoo.load_file(path)
    assert team.make_env().unwrapped.predators["bruiser_0"].limited_vision_radius == 10
    with pytest.raises(ValueError, match="observation"):
        team.make_env(observation_mode="flat_coords")


def test_files_with_another_format_or_pickled_objects_are_refused(tmp_path):
    path, _ = _write_model(tmp_path / "old.npz", format_version=99)
    with pytest.raises(ValueError, match="format 99"):
        zoo.load_file(path)
    np.savez(tmp_path / "pickled.npz", meta=np.array([{"a": 1}], dtype=object))
    with pytest.raises(ValueError):
        zoo.load_file(tmp_path / "pickled.npz")


def _serve(tmp_path, monkeypatch, sha=None):
    """Publish one model at a file:// URL and point the registry and cache at it."""
    source = tmp_path / "release"
    source.mkdir()
    path, _ = _write_model(source / "classic_stag_ippo_seed3.npz", seed=3)
    entry = {"scenario": "classic_stag", "algo": "ippo", "rank": "median", "seed": 3,
             "env_id": "GridHunt-ClassicStag-v0", "file": path.name,
             "sha256": sha or hashlib.sha256(path.read_bytes()).hexdigest()}
    monkeypatch.setattr(zoo, "_registry", lambda: [entry])
    monkeypatch.setenv("GRIDHUNT_ZOO_URL", source.as_uri())
    monkeypatch.setenv("GRIDHUNT_ZOO_DIR", str(tmp_path / "cache"))
    return path


def test_load_downloads_verifies_and_caches(tmp_path, monkeypatch):
    source = _serve(tmp_path, monkeypatch)
    team = zoo.load("classic_stag", "ippo")
    assert team.meta["seed"] == 3
    cached = tmp_path / "cache" / source.name
    assert cached.exists()
    source.unlink()  # the cached copy is used from now on
    assert zoo.load("classic_stag", "ippo", seed=3).meta["seed"] == 3
    assert zoo.list_models(algo="ippo") and not zoo.list_models(algo="happo")


def test_a_download_with_the_wrong_hash_is_rejected_and_not_kept(tmp_path, monkeypatch):
    _serve(tmp_path, monkeypatch, sha="0" * 64)
    with pytest.raises(OSError, match="SHA-256"):
        zoo.load("classic_stag", "ippo")
    assert not any((tmp_path / "cache").iterdir())


def test_unknown_models_name_what_is_published(tmp_path, monkeypatch):
    _serve(tmp_path, monkeypatch)
    with pytest.raises(LookupError, match="classic_stag"):
        zoo.load("swarm", "mappo")


def test_shipped_registry_is_well_formed():
    for model in zoo.list_models():
        assert model["env_id"] in gridhunt.list_envs()
        assert model["rank"] in (*zoo.RANKS, None)
        assert len(model["sha256"]) == 64 and model["file"].endswith(".npz")
