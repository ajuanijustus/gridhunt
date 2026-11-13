"""Export trained benchmark teams to the numpy model format read by gridhunt.zoo.

For every (scenario, algorithm) under --runs it exports the run with the median final score and
the best run (or every run with --all-seeds), then checks each exported file against the
trained PyTorch model:

1. logits: numpy and torch logits agree (to float32 rounding) on observations from real episodes;
2. greedy replay: greedy numpy actions reproduce the greedy evaluation returns stored in
   final.json, episode by episode (same evaluation seeds);
3. sampled score: the mean sampled return over the evaluation seeds agrees with final.json
   within 4 standard errors (the action draws differ from torch's, so only the mean is compared).

It writes the .npz files, registry.json (copy it to src/gridhunt/zoo/registry.json) and
VERIFICATION.md to --out. Upload the .npz files as assets of the "zoo-v1" GitHub release.

    python benchmark/export_zoo.py --runs benchmark/results/marl --out zoo_export
"""

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))

from marl.ppo import load_learner  # noqa: E402
from protocol import EVAL_SEED_OFFSET, REWARD_MODE, SCENARIOS, _variant_predators  # noqa: E402

import gridhunt  # noqa: E402
from gridhunt import zoo  # noqa: E402

LOGIT_TOLERANCE = 1e-4
GREEDY_MATCH_REQUIRED = 0.95
# Sampled means may differ by chance; 4 standard errors keeps false alarms below 1% over ~70 models
SAMPLED_SE_LIMIT = 4.0


def make_kwargs(scenario):
    """gridhunt.make() settings of a benchmark scenario (without predator changes)."""
    spec = SCENARIOS[scenario]
    return {"max_steps": spec["max_steps"], "reward_mode": REWARD_MODE, **spec.get("env_kwargs", {})}


def predator_changes(scenario):
    """Predator attributes a variant changes: {agent: {attribute: value}}."""
    spec = SCENARIOS[scenario]
    if not (spec.get("full_vision") or spec.get("blind")):
        return {}
    defaults = {p.predator_id: p for p in gridhunt._BENCHMARKS[spec["env_id"]]["predators"]}
    changes = {}
    for predator in _variant_predators(spec["env_id"], spec.get("full_vision", False), spec.get("blind", False)):
        default = defaults[predator.predator_id]
        changed = {
            attribute: copy.deepcopy(getattr(predator, attribute))
            for attribute in ("limited_vision_radius", "status")
            if getattr(predator, attribute) != getattr(default, attribute)
        }
        if changed:
            changes[predator.predator_id] = changed
    return changes


def _plain(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"cannot store {type(value).__name__} in model metadata")


def export_learner(learner, path, meta):
    """Write a MultiAgentPPO's actors and metadata to an .npz file; returns the full metadata."""
    actor_keys = {agent: "shared" if learner.shared_actor else agent for agent in learner.agents}
    modules = {"shared": learner.actors[learner.agents[0]]} if learner.shared_actor else learner.actors
    arrays = {}
    for key, module in modules.items():
        linears = [layer for layer in module if isinstance(layer, torch.nn.Linear)]
        for index, layer in enumerate(linears):
            arrays[f"actor/{key}/{index}/weight"] = layer.weight.detach().cpu().numpy().astype(np.float32)
            arrays[f"actor/{key}/{index}/bias"] = layer.bias.detach().cpu().numpy().astype(np.float32)
    full_meta = {
        "format_version": zoo.FORMAT_VERSION,
        "algo": learner.algo,
        "agents": list(learner.agents),
        "actor_keys": actor_keys,
        "shared_actor": learner.shared_actor,
        "activation": "tanh",
        "gridhunt_version": gridhunt.__version__,
        **meta,
    }
    full_meta = json.loads(json.dumps(full_meta, default=_plain))  # numpy numbers -> JSON numbers
    np.savez_compressed(path, meta=np.array(json.dumps(full_meta)), **arrays)
    return full_meta


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify(path, learner, final, episodes):
    """The three checks described in the module docstring; returns a dict of results."""
    team = zoo.load_file(path, rng=0)
    env = team.make_env()

    # 1. Logits on observations from greedy episodes
    worst = 0.0
    for episode in range(3):
        observations, _ = env.reset(seed=EVAL_SEED_OFFSET + episode)
        for _ in range(50):
            if not env.agents:
                break
            for agent in learner.agents:
                obs = observations[agent].astype(np.float32)
                with torch.no_grad():
                    expected = learner.actors[agent](torch.as_tensor(learner._actor_input(agent, obs[None])))[0].numpy()
                # Relative to the logits' size: float32 sums differ in rounding between libraries
                difference = np.abs(team.logits(agent, obs) - expected) / (1.0 + np.abs(expected))
                worst = max(worst, float(difference.max()))
            observations, *_ = env.step(team.act(observations, greedy=True))

    def play(greedy, seed):
        observations, _ = env.reset(seed=seed)
        total = 0.0
        while env.agents:
            observations, _, _, _, infos = env.step(team.act(observations, greedy=greedy))
            total += infos["__common__"]["team_reward"]
        return total

    # 2. Greedy replay of the evaluation episodes
    recorded = final["final_greedy_returns"][:episodes]
    replayed = [play(True, EVAL_SEED_OFFSET + i) for i in range(len(recorded))]
    matches = sum(abs(a - b) < 1e-6 for a, b in zip(replayed, recorded, strict=True))

    # 3. Sampled mean against final.json
    sampled = [play(False, EVAL_SEED_OFFSET + i) for i in range(episodes)]
    reference = final["final_eval_returns"]
    se = float(np.sqrt(np.var(sampled, ddof=1) / len(sampled) + np.var(reference, ddof=1) / len(reference)))
    difference = float(np.mean(sampled) - np.mean(reference))
    env.close()
    return {
        "max_logit_difference": worst,  # relative: |numpy - torch| / (1 + |torch|)
        "greedy_episodes": len(recorded),
        "greedy_matches": matches,
        "sampled_episodes": episodes,
        "sampled_return_mean": float(np.mean(sampled)),
        "reference_return_mean": float(np.mean(reference)),
        "difference_in_standard_errors": difference / se if se > 0 else 0.0,
        "passed": worst <= LOGIT_TOLERANCE and matches >= GREEDY_MATCH_REQUIRED * len(recorded) and (
            se == 0 and difference == 0 or abs(difference) <= SAMPLED_SE_LIMIT * se),
    }


def select_runs(run_dirs, all_seeds):
    """[(rank, run_dir, final)]: the median and best runs by final sampled score."""
    finals = []
    for run_dir in run_dirs:
        final = json.loads((run_dir / "final.json").read_text())
        finals.append((final["final_eval_return_mean"], final["seed"], run_dir, final))
    finals.sort(key=lambda item: (item[0], item[1]))
    if all_seeds:
        return [(None, run_dir, final) for _, _, run_dir, final in finals]
    median = finals[(len(finals) - 1) // 2]
    return [("median", median[2], median[3]), ("best", finals[-1][2], finals[-1][3])]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs", type=Path, default=Path("benchmark/results/marl"))
    parser.add_argument("--out", type=Path, default=Path("zoo_export"))
    parser.add_argument("--scenarios", nargs="*", default=list(SCENARIOS))
    parser.add_argument("--algos", nargs="*", default=["ippo", "mappo", "happo"])
    parser.add_argument("--all-seeds", action="store_true", help="export every run, not just median and best")
    parser.add_argument("--episodes", type=int, default=100, help="evaluation episodes for the checks")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)

    entries, exported = [], {}
    for scenario in args.scenarios:
        for algo in args.algos:
            run_dirs = sorted(p.parent for p in (args.runs / scenario / algo).glob("seed_*/final.json"))
            if not run_dirs:
                print(f"skip {scenario}/{algo}: no finished runs under {args.runs}")
                continue
            for rank, run_dir, final in select_runs(run_dirs, args.all_seeds):
                seed = final["seed"]
                file_name = f"{scenario}_{algo}_seed{seed}.npz"
                path = args.out / file_name
                if file_name not in exported:
                    learner = load_learner(run_dir / "model.pt")
                    one_hot = len(learner.agents) if learner.shared_actor else 0  # MAPPO's agent id input
                    meta = export_learner(learner, path, {
                        "scenario": scenario,
                        "seed": seed,
                        "env_id": SCENARIOS[scenario]["env_id"],
                        "make_kwargs": make_kwargs(scenario),
                        "predator_changes": predator_changes(scenario),
                        "obs_dims": {a: int(learner.actors[a][0].in_features) - one_hot for a in learner.agents},
                        "action_dims": {a: int(learner.actors[a][-1].out_features) for a in learner.agents},
                        "training": {
                            "env_steps": final["env_steps"], "budget": final["budget"], "config": final["config"],
                            "git_commit": final.get("git_commit"), "slurm_job": final.get("slurm_job"),
                        },
                        "evaluation": {
                            "episodes": len(final["final_eval_returns"]), "seed_offset": EVAL_SEED_OFFSET,
                            "sampled_return_mean": final["final_eval_return_mean"],
                            "greedy_return_mean": float(np.mean(final["final_greedy_returns"])),
                        },
                    })
                    check = verify(path, learner, final, args.episodes)
                    exported[file_name] = (meta, check)
                    status = "ok" if check["passed"] else "FAILED"
                    print(f"{status:6s} {file_name}: logits {check['max_logit_difference']:.1e}, greedy "
                          f"{check['greedy_matches']}/{check['greedy_episodes']}, sampled "
                          f"{check['sampled_return_mean']:.2f} vs {check['reference_return_mean']:.2f}", flush=True)
                meta, check = exported[file_name]
                entries.append({
                    "scenario": scenario, "algo": algo, "rank": rank, "seed": seed, "env_id": meta["env_id"],
                    "file": file_name, "sha256": _sha256(path), "size_bytes": path.stat().st_size,
                    "sampled_return_mean": meta["evaluation"]["sampled_return_mean"],
                    "greedy_return_mean": meta["evaluation"]["greedy_return_mean"],
                    "verification": check,
                })

    registry = {"format_version": zoo.FORMAT_VERSION, "models": entries}
    (args.out / "registry.json").write_text(json.dumps(registry, indent=2) + "\n")
    lines = ["| Scenario | Algorithm | Rank | Seed | Score | Logit diff | Greedy replay | Sampled vs recorded | Passed |",
             "|---|---|---|---|---|---|---|---|---|"]
    for e in entries:
        v = e["verification"]
        lines.append(f"| {e['scenario']} | {e['algo']} | {e['rank'] or '-'} | {e['seed']} | "
                     f"{e['sampled_return_mean']:.2f} | {v['max_logit_difference']:.1e} | "
                     f"{v['greedy_matches']}/{v['greedy_episodes']} | {v['sampled_return_mean']:.2f} vs "
                     f"{v['reference_return_mean']:.2f} ({v['difference_in_standard_errors']:+.1f} SE) | "
                     f"{'yes' if v['passed'] else 'NO'} |")
    (args.out / "VERIFICATION.md").write_text("\n".join(lines) + "\n")
    failed = [e["file"] for e in entries if not e["verification"]["passed"]]
    total = sum(p.stat().st_size for p in args.out.glob("*.npz"))
    print(f"\n{len(exported)} files ({total / 1e6:.1f} MB), {len(entries)} registry entries -> {args.out}")
    if failed:
        print(f"Checks failed for: {sorted(set(failed))}")
        sys.exit(1)


if __name__ == "__main__":
    main()
