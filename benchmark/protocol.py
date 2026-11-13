"""
GridHunt benchmark protocol: how every scenario is configured for the baseline tables.

The registered environments (gridhunt.make("GridHunt-...-v0")) keep their defaults;
the benchmark only overrides what is listed here, identically for every algorithm:

- reward_mode = "team": every predator receives the team reward, so cooperative
  methods (MAPPO, HAPPO) are used as intended and zero-capture-power roles (e.g.
  Military's UAV) still get a learning signal. The primary metric is the team
  return: the sum of infos["__common__"]["team_reward"] over an episode.
- shorter episodes (max_steps): long horizons only add advantage-estimation noise.
- prey are not trainable; they move only where the scenario sets prey_motion
  (Specialists: prey flee from the nearest predator).

Training budgets (env steps) come from a 20M-step pilot (3 seeds, every algorithm):
1.5x the step where the slowest algorithm's curve levels off, rounded up to a whole
million and capped at the pilot's 20M. Variants use their main scenario's budget, so
each hypothesis test compares runs trained for equally long.
Exception: Military was still rising steadily at 20M (about half the scripted score),
so it and its variant get 50M; its 20-seed curves show whether that is enough.
Still improving at 20M, and reported as not converged: pig_chase (seeds jump between
0 and 50) and swarm (HAPPO only).
"""

import copy

import gridhunt

REWARD_MODE = "team"

# name -> registered env id, episode length used by the benchmark, budget (env steps).
# "main" scenarios form the benchmark tables; "variant" scenarios change one thing
# about a main scenario to test a hypothesis (env_kwargs / full_vision / blind).
SCENARIOS = {
    "classic_stag": {"env_id": "GridHunt-ClassicStag-v0", "max_steps": 200, "budget": 3_000_000, "group": "main"},
    "toxic_stag": {"env_id": "GridHunt-ToxicStag-v0", "max_steps": 200, "budget": 20_000_000, "group": "main"},
    "specialists": {"env_id": "GridHunt-Specialists-v0", "max_steps": 200, "budget": 20_000_000, "group": "main"},
    "military": {"env_id": "GridHunt-Military-v0", "max_steps": 200, "budget": 50_000_000, "group": "main"},
    # Raid and Swarm use coordinate observations: their default one-hot grids (870 and 9,528
    # inputs per agent) left every baseline at or below the random team in the 5M-step pilot.
    "raid": {"env_id": "GridHunt-Raid-v0", "max_steps": 200, "budget": 7_000_000, "group": "main",
             "env_kwargs": {"observation_mode": "flat_coords"}},
    "pig_chase": {"env_id": "GridHunt-PigChase-v0", "max_steps": 200, "budget": 20_000_000, "group": "main"},
    "land_and_sea": {"env_id": "GridHunt-LandAndSea-v0", "max_steps": 200, "budget": 7_000_000, "group": "main"},
    "swarm": {"env_id": "GridHunt-Swarm-v0", "max_steps": 300, "budget": 20_000_000, "group": "main",
              "env_kwargs": {"observation_mode": "flat_coords"}},
    # Does an explicit message channel help the Scout guide the Bruiser to the stag?
    "specialists_comm": {"env_id": "GridHunt-Specialists-v0", "max_steps": 200, "budget": 20_000_000,
                         "group": "variant", "env_kwargs": {"communication": 4}},
    # Information ablation: the Bruiser sees the whole board, same observation encoding
    "specialists_full_vision": {"env_id": "GridHunt-Specialists-v0", "max_steps": 200, "budget": 20_000_000,
                                "group": "variant", "full_vision": True},
    # Information ablation: no predator knows whether the stag is toxic
    "toxic_stag_blind": {"env_id": "GridHunt-ToxicStag-v0", "max_steps": 200, "budget": 20_000_000,
                         "group": "variant", "blind": True},
    # Information ablation: the ground vehicle sees the whole board (is the UAV's scouting needed?)
    "military_full_vision": {"env_id": "GridHunt-Military-v0", "max_steps": 200, "budget": 50_000_000,
                             "group": "variant", "full_vision": True},
}
MAIN_SCENARIOS = [name for name, spec in SCENARIOS.items() if spec["group"] == "main"]

# Evaluation: fixed seeds kept apart from training seeds (training uses 0..999)
EVAL_SEED_OFFSET = 1_000_000
EVAL_EPISODES = 32


def _variant_predators(env_id, full_vision=False, blind=False):
    """The scenario's predators with information removed or added.

    full_vision: limited-vision radii widen to the whole board; the flag is kept so
    observations keep the same egocentric encoding and size (only information changes).
    blind: no predator is told whether a stag is toxic.
    """
    spec = gridhunt._BENCHMARKS[env_id]
    predators = copy.deepcopy(spec["predators"])
    for predator in predators:
        if full_vision and predator.status.get("limited_vision"):
            predator.limited_vision_radius = max(spec["grid_size"])
        if blind:
            predator.status.pop("toxicity_aware", None)
    return predators


def make_env(scenario, reward_mode=REWARD_MODE, render_mode=None, **overrides):
    """Create a scenario configured by the benchmark protocol."""
    spec = SCENARIOS[scenario]
    kwargs = {"max_steps": spec["max_steps"], "reward_mode": reward_mode, **spec.get("env_kwargs", {})}
    if spec.get("full_vision") or spec.get("blind"):
        kwargs["predators"] = _variant_predators(spec["env_id"], spec.get("full_vision", False), spec.get("blind", False))
    kwargs.update(overrides)
    return gridhunt.make(spec["env_id"], render_mode=render_mode, **kwargs)


def team_return(infos_per_step):
    """Episode team return from the per-step infos returned by env.step()."""
    return sum(infos["__common__"]["team_reward"] for infos in infos_per_step)
