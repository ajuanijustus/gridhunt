"""Pretrained teams from the GridHunt benchmark, runnable with numpy alone.

Each model is one trained team (IPPO, MAPPO or HAPPO) for one benchmark scenario, exported
from the benchmark runs. Two runs per scenario and algorithm are published: the seed with
the median final score (representative of the algorithm) and the best seed (the strongest
teammates). Models are downloaded on first use, checked against their SHA-256 hash and cached.

    from gridhunt import zoo

    zoo.list_models(scenario="toxic_stag")
    team = zoo.load("toxic_stag", "mappo", rank="best")
    env = team.make_env()                       # the scenario exactly as it was trained
    observations, _ = env.reset(seed=0)
    while env.agents:
        observations, *_ = env.step(team.act(observations))

A team is also a teammate for ``gridhunt.hat.HumanSession``: it is called as
``team(observation, agent_id, env)``, so ``teammates={"predator_1": team}`` lets a trained
agent play alongside a participant.

Downloads go to ``$GRIDHUNT_ZOO_DIR`` or ``~/.cache/gridhunt/zoo``; set ``$GRIDHUNT_ZOO_URL`` to
use a mirror.
"""

import copy
import hashlib
import json
import os
import tempfile
import urllib.request
from importlib import resources
from pathlib import Path

import numpy as np

FORMAT_VERSION = 1
DEFAULT_URL = "https://github.com/ajuanijustus/gridhunt/releases/download/zoo-v1/"
RANKS = ("median", "best")

__all__ = ["FORMAT_VERSION", "ZooTeam", "cache_dir", "list_models", "load", "load_file"]


# ------------------------------------------------------------------ registry


def _registry():
    text = resources.files(__package__).joinpath("registry.json").read_text(encoding="utf-8")
    return json.loads(text)["models"]


def list_models(scenario=None, algo=None, rank=None):
    """Published models, optionally filtered: a list of dicts with ``scenario``, ``algo``,
    ``rank``, ``seed``, ``env_id``, ``file``, ``sha256`` and the final evaluation score."""
    models = _registry()
    return [
        m for m in models
        if (scenario is None or m["scenario"] == scenario)
        and (algo is None or m["algo"] == algo)
        and (rank is None or m["rank"] == rank)
    ]


def cache_dir():
    """Where downloaded models are kept."""
    if os.environ.get("GRIDHUNT_ZOO_DIR"):
        return Path(os.environ["GRIDHUNT_ZOO_DIR"])
    base = os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache"
    return Path(base) / "gridhunt" / "zoo"


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fetch(entry):
    """Path to a verified local copy of a registry entry, downloading it if needed."""
    target = cache_dir() / entry["file"]
    if target.exists() and _sha256(target) == entry["sha256"]:
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    url = os.environ.get("GRIDHUNT_ZOO_URL", DEFAULT_URL).rstrip("/") + "/" + entry["file"]
    with tempfile.NamedTemporaryFile(dir=target.parent, delete=False, suffix=".part") as tmp:
        tmp_path = Path(tmp.name)
    try:
        if url.startswith(("http://", "https://", "file://")):
            with urllib.request.urlopen(url, timeout=60) as response, open(tmp_path, "wb") as out:  # noqa: S310
                while chunk := response.read(1 << 20):
                    out.write(chunk)
        else:
            raise ValueError(f"unsupported model URL {url!r}")
        actual = _sha256(tmp_path)
        if actual != entry["sha256"]:
            raise OSError(f"{entry['file']}: SHA-256 {actual} does not match the registry ({entry['sha256']})")
        tmp_path.replace(target)
    finally:
        tmp_path.unlink(missing_ok=True)
    return target


def load(scenario, algo, rank="median", seed=None, rng=None):
    """Load a published team.

    Args:
        scenario: a benchmark scenario name, for example ``"toxic_stag"``; see ``list_models``.
        algo: ``"ippo"``, ``"mappo"`` or ``"happo"``.
        rank: ``"median"`` (the run with the median final score) or ``"best"``.
        seed: pick a specific training seed instead of a rank (only if that run is published).
        rng: seed or ``numpy.random.Generator`` for sampling actions.
    """
    matches = [m for m in list_models(scenario, algo) if (m["seed"] == seed if seed is not None else m["rank"] == rank)]
    if not matches:
        available = sorted({(m["scenario"], m["algo"], m["rank"], m["seed"]) for m in list_models()})
        if not available:
            raise LookupError("no pretrained models are published in this version of gridhunt")
        raise LookupError(f"no model for scenario={scenario!r}, algo={algo!r}, rank={rank!r}, seed={seed!r}; "
                          f"published: {available}")
    return load_file(_fetch(matches[0]), rng=rng)


def load_file(path, rng=None):
    """Load a team from a local ``.npz`` file written by the benchmark's export script."""
    with np.load(path, allow_pickle=False) as data:
        meta = json.loads(str(data["meta"]))
        arrays = {key: data[key] for key in data.files if key != "meta"}
    if meta.get("format_version") != FORMAT_VERSION:
        raise ValueError(f"{path}: model format {meta.get('format_version')} is not supported "
                         f"(this gridhunt reads format {FORMAT_VERSION})")
    return ZooTeam(meta, arrays, rng=rng)


# ---------------------------------------------------------------------- team


class ZooTeam:
    """A trained team: one actor per agent (IPPO, HAPPO) or one shared actor (MAPPO)."""

    def __init__(self, meta, arrays, rng=None):
        self.meta = meta
        self.scenario = meta["scenario"]
        self.algo = meta["algo"]
        self.agents = list(meta["agents"])
        self.name = f"{self.algo}:{self.scenario}:{meta.get('rank') or 'seed'}{meta['seed']}"
        self.rng = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)
        self._layers = {}
        for key in set(meta["actor_keys"].values()):
            layers, index = [], 0
            while f"actor/{key}/{index}/weight" in arrays:
                weight = arrays[f"actor/{key}/{index}/weight"].astype(np.float32)
                bias = arrays[f"actor/{key}/{index}/bias"].astype(np.float32)
                layers.append((weight.T.copy(), bias))
                index += 1
            if not layers:
                raise ValueError(f"model has no weights for actor {key!r}")
            self._layers[key] = layers
        self._onehot = {a: np.eye(len(self.agents), dtype=np.float32)[i] for i, a in enumerate(self.agents)}

    def __repr__(self):
        score = self.meta.get("evaluation", {}).get("sampled_return_mean")
        return f"ZooTeam({self.name}, agents={self.agents}, final score={score})"

    # -- inference

    def logits(self, agent, observation):
        """Action logits for one agent; ``observation`` may be one vector or a batch."""
        x = np.asarray(observation, dtype=np.float32)
        if self.meta["shared_actor"]:
            onehot = np.broadcast_to(self._onehot[agent], x.shape[:-1] + (len(self.agents),))
            x = np.concatenate([x, onehot], axis=-1)
        layers = self._layers[self.meta["actor_keys"][agent]]
        for weight, bias in layers[:-1]:
            x = np.tanh(x @ weight + bias)
        weight, bias = layers[-1]
        return x @ weight + bias

    def probabilities(self, agent, observation):
        logits = self.logits(agent, observation).astype(np.float64)
        logits -= logits.max(axis=-1, keepdims=True)
        exp = np.exp(logits)
        return exp / exp.sum(axis=-1, keepdims=True)

    def act_agent(self, agent, observation, greedy=False):
        """One agent's action. Sampled from the policy by default (as the benchmark evaluates)."""
        if greedy:
            return int(np.argmax(self.logits(agent, observation)))
        probs = self.probabilities(agent, observation)
        return int(self.rng.choice(len(probs), p=probs))

    def act(self, observations, greedy=False):
        """Actions for every agent in ``observations`` (a dict from ``env.reset``/``env.step``)."""
        return {agent: self.act_agent(agent, obs, greedy) for agent, obs in observations.items()}

    def __call__(self, observation, agent_id, env=None):
        """Teammate interface for ``gridhunt.hat.HumanSession``."""
        return self.act_agent(agent_id, observation)

    def reset(self):
        """Nothing to reset: the policies have no memory."""

    # -- environment

    def make_env(self, **overrides):
        """The environment the team was trained in (benchmark settings), with optional overrides.

        Observation settings must stay as trained; overriding ``max_steps`` or ``render_mode``
        is fine.
        """
        import gridhunt

        env_id = self.meta["env_id"]
        kwargs = copy.deepcopy(self.meta["make_kwargs"])
        changes = self.meta.get("predator_changes") or {}
        if changes:
            predators = copy.deepcopy(gridhunt._BENCHMARKS[env_id]["predators"])
            for predator in predators:
                for attribute, value in changes.get(predator.predator_id, {}).items():
                    setattr(predator, attribute, copy.deepcopy(value))
            kwargs["predators"] = predators
        kwargs.update(overrides)
        env = gridhunt.make(env_id, **kwargs)
        for agent in self.agents:
            size = env.observation_space(agent).shape[0]
            if size != self.meta["obs_dims"][agent]:
                raise ValueError(f"{agent}: the environment gives {size} observation values but the team was "
                                 f"trained on {self.meta['obs_dims'][agent]}; do not override observation settings")
        return env
