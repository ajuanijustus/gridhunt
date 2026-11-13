"""The public package API: make, list_envs, the PettingZoo registry and setting validation."""

import pytest

import gridhunt


def test_list_envs_matches_the_registry():
    assert gridhunt.list_envs() == list(gridhunt._BENCHMARKS)
    assert "GridHunt-PigChase-v0" in gridhunt.list_envs()


def test_every_listed_environment_can_be_made():
    for env_id in gridhunt.list_envs():
        env = gridhunt.make(env_id)
        observations, _ = env.reset(seed=0)
        assert set(observations) == set(env.agents)


def test_unknown_environment_names_the_registered_ones():
    with pytest.raises(ValueError, match="GridHunt-ClassicStag-v0"):
        gridhunt.make("GridHunt-PIG-v0")


def test_misspelled_setting_is_rejected_instead_of_ignored():
    with pytest.raises(TypeError, match="max_step"):
        gridhunt.make("GridHunt-ClassicStag-v0", max_step=10)


def test_invalid_observation_mode_is_rejected():
    with pytest.raises(ValueError, match="observation_mode"):
        gridhunt.make("GridHunt-ClassicStag-v0", observation_mode="toxic")
    gridhunt.make("GridHunt-ToxicStag-v0", observation_mode="toxic")


def test_environments_are_registered_with_pettingzoo():
    pettingzoo = pytest.importorskip("pettingzoo")
    if not hasattr(pettingzoo, "parallel_registry"):
        pytest.skip("PettingZoo < 1.27 has no environment registry")
    for env_id in gridhunt.list_envs():
        env = pettingzoo.make("parallel", env_id, max_steps=20)
        assert env.max_steps == 20
        env.reset(seed=0)


def test_public_names_are_exported():
    for name in gridhunt.__all__:
        assert hasattr(gridhunt, name), name
