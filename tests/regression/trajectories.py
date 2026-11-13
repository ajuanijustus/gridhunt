"""Fixed-action trajectories used to check that environment dynamics never change silently.

Each case steps one environment configuration with a fixed seed and a fixed, seeded
sequence of random actions, and reduces every step (observations, rewards, terminations,
truncations, infos and the global state) to a short digest. The reference digests in
``reference_trajectories.json.gz`` were recorded with the code that produced the benchmark
results; ``tests/test_regression.py`` replays every case and compares.

A mismatch means an environment's dynamics, rewards or observations changed. That is only
allowed together with a new environment version suffix (``-v0`` to ``-v1``), a CHANGELOG
entry and newly recorded references for the new version.
"""

import copy
import hashlib
import random
import struct

import numpy as np

STEPS = 200
SEEDS = (0, 1)
# Short episodes so every case also covers truncation and unseeded resets
MAX_STEPS = 60
DIGEST_CHARS = 16


def _full_vision(predators, grid_size):
    for predator in predators:
        if predator.status.get("limited_vision"):
            predator.limited_vision_radius = max(grid_size)
    return predators


def _blind(predators, grid_size):
    for predator in predators:
        predator.status.pop("toxicity_aware", None)
    return predators


PREDATOR_TRANSFORMS = {"full_vision": _full_vision, "blind": _blind}

ENV_IDS = (
    "GridHunt-ClassicStag-v0",
    "GridHunt-ToxicStag-v0",
    "GridHunt-Specialists-v0",
    "GridHunt-Military-v0",
    "GridHunt-Raid-v0",
    "GridHunt-PigChase-v0",
    "GridHunt-LandAndSea-v0",
    "GridHunt-Swarm-v0",
)
_COORDINATE_OBS = {"GridHunt-Raid-v0", "GridHunt-Swarm-v0"}


def _cases():
    cases = {}
    for env_id in ENV_IDS:
        short = env_id.split("-")[1]
        # Registered defaults
        cases[f"{short}/default"] = {"env_id": env_id, "kwargs": {}}
        # The configuration used by the benchmark protocol
        protocol = {"reward_mode": "team"}
        if env_id in _COORDINATE_OBS:
            protocol["observation_mode"] = "flat_coords"
        cases[f"{short}/protocol"] = {"env_id": env_id, "kwargs": protocol}
    # Benchmark variants
    cases["Specialists/communication"] = {
        "env_id": "GridHunt-Specialists-v0", "kwargs": {"reward_mode": "team", "communication": 4}}
    cases["Specialists/full_vision"] = {
        "env_id": "GridHunt-Specialists-v0", "kwargs": {"reward_mode": "team"}, "predators": "full_vision"}
    cases["ToxicStag/blind"] = {
        "env_id": "GridHunt-ToxicStag-v0", "kwargs": {"reward_mode": "team"}, "predators": "blind"}
    cases["Military/full_vision"] = {
        "env_id": "GridHunt-Military-v0", "kwargs": {"reward_mode": "team"}, "predators": "full_vision"}
    # Every observation mode, with and without terrain and limited vision
    for mode in ("coords", "flat_coords", "one_hot"):
        cases[f"ClassicStag/{mode}"] = {"env_id": "GridHunt-ClassicStag-v0", "kwargs": {"observation_mode": mode}}
        cases[f"Military/{mode}"] = {"env_id": "GridHunt-Military-v0", "kwargs": {"observation_mode": mode}}
    cases["ToxicStag/one_hot"] = {"env_id": "GridHunt-ToxicStag-v0", "kwargs": {"observation_mode": "one_hot"}}
    # Observation encoders and action masking
    cases["Specialists/encoders"] = {
        "env_id": "GridHunt-Specialists-v0",
        "kwargs": {"observation_encoders": ["agent_centric", "action_mask"], "use_action_mask": True},
    }
    cases["Military/encoders_coords"] = {
        "env_id": "GridHunt-Military-v0",
        "kwargs": {"observation_mode": "coords", "observation_encoders": ["agent_centric", "action_mask"]},
    }
    return cases


CASES = _cases()


def _feed(hasher, value):
    """Feed a value into the hasher in a canonical, type-tagged form.

    Integers and floats are hashed as float64, so a reward changing from 1 to 1.0 does not
    count as a change; arrays keep their dtype and shape, so an observation dtype change does.
    """
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        hasher.update(b"A" + array.dtype.str.encode() + repr(array.shape).encode())
        hasher.update(array.tobytes())
    elif isinstance(value, np.generic):
        _feed(hasher, value.item())
    elif isinstance(value, bool):
        hasher.update(b"B1" if value else b"B0")
    elif isinstance(value, (int, float)):
        hasher.update(b"F" + struct.pack("<d", float(value)))
    elif isinstance(value, str):
        hasher.update(b"S" + value.encode() + b"\0")
    elif value is None:
        hasher.update(b"N")
    elif isinstance(value, dict):
        hasher.update(b"D%d" % len(value))
        for key in sorted(value, key=str):
            _feed(hasher, str(key))
            _feed(hasher, value[key])
    elif isinstance(value, (list, tuple)):
        hasher.update(b"L%d" % len(value))
        for item in value:
            _feed(hasher, item)
    elif isinstance(value, (set, frozenset)):
        _feed(hasher, sorted(value, key=repr))
    else:
        raise TypeError(f"cannot hash {type(value).__name__} in a trajectory")


def _digest(*values):
    hasher = hashlib.sha256()
    for value in values:
        _feed(hasher, value)
    return hasher.hexdigest()[:DIGEST_CHARS]


def make_case_env(gridhunt, case, env_id=None):
    """Build the case's environment with the given gridhunt module."""
    env_id = env_id or case["env_id"]
    kwargs = copy.deepcopy(case["kwargs"])
    kwargs["max_steps"] = MAX_STEPS
    if "predators" in case:
        spec = gridhunt._BENCHMARKS[env_id]
        transform = PREDATOR_TRANSFORMS[case["predators"]]
        kwargs["predators"] = transform(copy.deepcopy(spec["predators"]), spec["grid_size"])
    return gridhunt.make(env_id, **kwargs)


def run_case(env, seed):
    """Step env for STEPS steps; return one digest per reset and per step.

    The first reset is seeded; later resets are not, so the env's own RNG carries over
    between episodes, as it does during training.
    """
    action_rng = random.Random(seed)
    observations, infos = env.reset(seed=seed)
    digests = [_digest("reset", observations, infos, env.state())]
    for _ in range(STEPS):
        if not env.agents:
            observations, infos = env.reset()
            digests.append(_digest("reset", observations, infos, env.state()))
        actions = {agent: action_rng.randrange(int(env.action_space(agent).n)) for agent in env.agents}
        observations, rewards, terminations, truncations, infos = env.step(actions)
        digests.append(
            _digest("step", actions, observations, rewards, terminations, truncations, infos, env.state())
        )
    return digests
