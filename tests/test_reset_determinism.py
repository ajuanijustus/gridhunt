import pytest
import gridhunt


@pytest.fixture
def env():
    return gridhunt.make("GridHunt-ClassicStag-v0")


def _positions(environment):
    return {
        **{agent_id: pred.position for agent_id, pred in environment.predators.items()},
        **{prey_id: prey.position for prey_id, prey in environment.prey.items()},
    }


def test_seeded_reset_produces_same_positions(env):
    env.reset(seed=42)
    first = _positions(env)
    env.reset(seed=42)
    second = _positions(env)
    assert first == second


def test_different_seeds_produce_different_positions(env):
    env.reset(seed=42)
    first = _positions(env)
    env.reset(seed=43)
    second = _positions(env)
    assert first != second


def test_fixed_position_parameters_are_preserved():
    env = gridhunt.make("GridHunt-PIG-v0")
    env.reset(seed=42)
    first = _positions(env)
    env.reset(seed=99)
    second = _positions(env)
    assert first["pig"] == second["pig"] == (4, 4)
    assert first["predator_0"] == second["predator_0"] == (2, 2)
    assert first["predator_1"] == second["predator_1"] == (6, 6)


def test_reset_restores_removed_prey():
    env = gridhunt.make("GridHunt-Raid-v0")
    env.reset()
    initial_count = len(env.prey)
    barn_pos = env.prey["barn_0"].position
    env.predators["raider_0"].position = barn_pos
    env.step({"raider_0": 0, "raider_1": 0, "raider_2": 0})
    assert len(env.prey) < initial_count
    env.reset()
    assert len(env.prey) == initial_count
    assert "barn_0" in env.prey


def test_reset_with_positions_override():
    env = gridhunt.make("GridHunt-ClassicStag-v0")
    options = {
        "predator_positions": {"predator_0": (0, 0), "predator_1": (1, 1)},
        "prey_positions": {"stag_0": (2, 2), "hare_0": (3, 3), "hare_1": (4, 4)},
    }
    env.reset(options=options)
    assert env.predators["predator_0"].position == (0, 0)
    assert env.predators["predator_1"].position == (1, 1)
    assert env.prey["stag_0"].position == (2, 2)
    assert env.prey["hare_0"].position == (3, 3)
    assert env.prey["hare_1"].position == (4, 4)


def test_reset_counters():
    env = gridhunt.make("GridHunt-ClassicStag-v0")
    env.reset()
    for _ in range(5):
        env.step({"predator_0": 0, "predator_1": 0})
    assert env.step_count == 5
    env.score = 123
    env.reset()
    assert env.step_count == 0
    assert env.score == 0
