"""
Train one baseline (IPPO, MAPPO or HAPPO) on one GridHunt scenario with one seed.

Every run writes to its own directory (default benchmark/results/marl/<scenario>/<algo>/seed_<seed>):
  progress.csv   one row per evaluation: env steps, evaluation returns (sampled actions;
                 greedy also recorded), training stats
  evals.jsonl    every evaluation's per-episode team returns and capture counts
  checkpoint.pt  latest training state; rerunning the same command resumes from it
  model.pt       final policy and critic weights
  final.json     final evaluation (more episodes), config, versions and git commit

Usage:
    python benchmark/train_marl.py --scenario toxic_stag --algo happo --seed 0 --budget 2000000
"""

import argparse
import csv
import dataclasses
import json
import os
import platform
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from marl.ppo import ALGORITHMS, MultiAgentPPO, PPOConfig, collect_rollout, evaluate  # noqa: E402
from marl.vec_env import VecEnv  # noqa: E402
from protocol import EVAL_EPISODES, EVAL_SEED_OFFSET, SCENARIOS, make_env  # noqa: E402

PROGRESS_FIELDS = [
    "env_steps", "iteration", "wall_time_s", "eval_return_mean", "eval_return_std",
    "eval_return_median", "greedy_return_mean", "train_return_mean", "policy_loss", "value_loss", "entropy",
    "approx_kl", "clip_frac", "adv_floor_active", "steps_per_s",
]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scenario", required=True, choices=list(SCENARIOS))
    parser.add_argument("--algo", required=True, choices=ALGORITHMS)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--budget", type=int, default=None,
                        help="training env steps (default: the protocol budget for the scenario)")
    parser.add_argument("--eval-every", type=int, default=None, help="env steps between evaluations (default: budget/100)")
    parser.add_argument("--eval-episodes", type=int, default=EVAL_EPISODES)
    parser.add_argument("--final-eval-episodes", type=int, default=100)
    parser.add_argument("--checkpoint-every", type=int, default=25, help="iterations between checkpoints")
    parser.add_argument("--threads", type=int, default=1, help="torch CPU threads")
    parser.add_argument("--out", type=Path, default=None)
    defaults = PPOConfig()
    for field in dataclasses.fields(PPOConfig):
        if field.name != "hidden":
            parser.add_argument(f"--{field.name.replace('_', '-')}", type=field.type, default=getattr(defaults, field.name))
    return parser.parse_args()


def git_commit():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                              cwd=Path(__file__).parent, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main():
    args = parse_args()
    budget = args.budget or SCENARIOS[args.scenario]["budget"]
    if not budget:
        raise SystemExit(f"No budget set for {args.scenario}; pass --budget")
    eval_every = args.eval_every or max(budget // 100, 1)
    out = args.out or Path("benchmark/results/marl") / args.scenario / args.algo / f"seed_{args.seed}"
    out.mkdir(parents=True, exist_ok=True)
    if (out / "final.json").exists():
        print(f"Already finished: {out / 'final.json'}")
        return
    torch.set_num_threads(args.threads)

    config = PPOConfig(**{f.name: getattr(args, f.name) for f in dataclasses.fields(PPOConfig) if f.name != "hidden"})
    steps_per_iteration = config.num_envs * config.rollout_steps
    seed_everything(args.seed)

    def env_fn():
        return make_env(args.scenario)

    probe = env_fn()
    agents = probe.possible_agents
    obs_dims = {a: probe.observation_space(a).shape[0] for a in agents}
    action_dims = {a: probe.action_space(a).n for a in agents}
    learner = MultiAgentPPO(args.algo, agents, obs_dims, action_dims, probe.state_space.shape[0], config, seed=args.seed)

    # Resume if a checkpoint exists
    iteration, env_steps, wall_offset, next_eval = 0, 0, 0.0, 0
    floor_updates = 0.0  # updates where the advantage std floor was active (fraction per update, summed)
    checkpoint_path = out / "checkpoint.pt"
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, weights_only=False)
        learner.load_state_dict(checkpoint["learner"])
        iteration, env_steps = checkpoint["iteration"], checkpoint["env_steps"]
        wall_offset, next_eval = checkpoint["wall_time_s"], checkpoint["next_eval"]
        floor_updates = checkpoint.get("floor_updates", 0.0)
        random.setstate(checkpoint["python_rng"])
        np.random.set_state(checkpoint["numpy_rng"])
        torch.set_rng_state(checkpoint["torch_rng"])
        print(f"Resuming from iteration {iteration} ({env_steps} env steps)")
    elif (out / "progress.csv").exists():
        (out / "progress.csv").unlink()
        (out / "evals.jsonl").unlink(missing_ok=True)

    # A resumed run reseeds its environments from (seed, iteration)
    envs = VecEnv(env_fn, config.num_envs, seed=args.seed * 1000 + iteration)
    obs, state = envs.reset()
    progress_new = not (out / "progress.csv").exists()
    progress_file = open(out / "progress.csv", "a", newline="")
    progress = csv.DictWriter(progress_file, fieldnames=PROGRESS_FIELDS)
    if progress_new:
        progress.writeheader()

    start = time.time() - wall_offset
    recent_returns, stats, interval_start, interval_steps = [], {}, time.time(), 0
    while True:
        if env_steps >= next_eval or env_steps >= budget:
            returns, captures = evaluate(learner, env_fn, args.eval_episodes, EVAL_SEED_OFFSET)
            greedy_returns, greedy_captures = evaluate(learner, env_fn, args.eval_episodes, EVAL_SEED_OFFSET, greedy=True)
            elapsed = time.time() - interval_start
            row = {
                "env_steps": env_steps, "iteration": iteration, "wall_time_s": round(time.time() - start, 1),
                "eval_return_mean": float(np.mean(returns)), "eval_return_std": float(np.std(returns)),
                "eval_return_median": float(np.median(returns)),
                "greedy_return_mean": float(np.mean(greedy_returns)),
                "train_return_mean": float(np.mean(recent_returns)) if recent_returns else "",
                "steps_per_s": round(interval_steps / elapsed, 1) if interval_steps else "",
                **{k: stats.get(k, "") for k in ("policy_loss", "value_loss", "entropy", "approx_kl", "clip_frac",
                                                 "adv_floor_active")},
            }
            progress.writerow(row)
            progress_file.flush()
            with open(out / "evals.jsonl", "a") as f:
                f.write(json.dumps({"env_steps": env_steps, "returns": returns, "captures": captures,
                                    "greedy_returns": greedy_returns, "greedy_captures": greedy_captures}) + "\n")
            print(f"[{args.scenario}/{args.algo}/seed {args.seed}] {env_steps:>10d} steps  "
                  f"eval {row['eval_return_mean']:8.2f} ± {row['eval_return_std']:6.2f}  "
                  f"greedy {row['greedy_return_mean']:8.2f}  {row['steps_per_s']} steps/s", flush=True)
            recent_returns, interval_start, interval_steps = [], time.time(), 0
            next_eval += eval_every
        if env_steps >= budget:
            break

        batch, obs, state, finished = collect_rollout(learner, envs, obs, state)
        stats = learner.update(batch)
        floor_updates += stats.get("adv_floor_active", 0.0)
        recent_returns.extend(finished)
        iteration += 1
        env_steps += steps_per_iteration
        interval_steps += steps_per_iteration

        if iteration % args.checkpoint_every == 0:
            torch.save({
                "learner": learner.state_dict(), "iteration": iteration, "env_steps": env_steps,
                "wall_time_s": time.time() - start, "next_eval": next_eval, "floor_updates": floor_updates,
                "python_rng": random.getstate(), "numpy_rng": np.random.get_state(),
                "torch_rng": torch.get_rng_state(),
            }, checkpoint_path)
    progress_file.close()

    torch.save({"learner": learner.state_dict(), "agents": agents, "obs_dims": obs_dims, "action_dims": action_dims,
                "state_dim": probe.state_space.shape[0],
                "config": dataclasses.asdict(config)}, out / "model.pt")
    returns, captures = evaluate(learner, env_fn, args.final_eval_episodes, EVAL_SEED_OFFSET)
    greedy_returns, greedy_captures = evaluate(learner, env_fn, args.final_eval_episodes, EVAL_SEED_OFFSET, greedy=True)
    final = {
        "scenario": args.scenario, "algo": args.algo, "seed": args.seed, "env_steps": env_steps,
        "budget": budget, "protocol": SCENARIOS[args.scenario], "config": dataclasses.asdict(config),
        "final_eval_returns": returns, "final_eval_captures": captures,
        "final_eval_return_mean": float(np.mean(returns)), "eval_policy": "sampled",
        "adv_floor_active_frac": floor_updates / max(iteration, 1),
        "final_greedy_returns": greedy_returns, "final_greedy_captures": greedy_captures, "wall_time_s": round(time.time() - start, 1),
        "git_commit": git_commit(), "python": platform.python_version(), "torch": torch.__version__,
        "numpy": np.__version__, "host": platform.node(), "slurm_job": os.environ.get("SLURM_JOB_ID"),
    }
    (out / "final.json").write_text(json.dumps(final, indent=2))
    checkpoint_path.unlink(missing_ok=True)
    print(f"Done: final eval {final['final_eval_return_mean']:.2f} over {len(returns)} episodes -> {out}")


if __name__ == "__main__":
    main()
