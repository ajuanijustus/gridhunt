"""
Reference scores for every protocol scenario: a uniform-random team and the scripted
GreedyPolicy team, evaluated on the same held-out seeds as the trained baselines.

aggregate.py uses them to report normalized scores, (R - random) / (scripted - random):
0 is a random team, 1 matches the scripted team. The scripted team reads the true
state, so it is a reference point, not an upper bound (learned teams can exceed 1).

Usage:
    python benchmark/reference_scores.py                      # all scenarios, 100 episodes
    python benchmark/reference_scores.py --scenarios specialists --episodes 50
"""

import argparse
import csv
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from protocol import EVAL_SEED_OFFSET, SCENARIOS, make_env
from reference_policies import GreedyPolicy, RandomPolicy, run_episode

OUTPUT = Path("benchmark/results/reference_scores.csv")


def score(scenario, episodes):
    env = make_env(scenario)
    agents = env.possible_agents
    greedy = GreedyPolicy()
    random_returns, scripted_returns = [], []
    for episode in range(episodes):
        seed = EVAL_SEED_OFFSET + episode
        random_team = {a: RandomPolicy(seed=episode * 100 + i) for i, a in enumerate(agents)}
        random_returns.append(run_episode(env, random_team, seed))
        scripted_returns.append(run_episode(env, {a: greedy for a in agents}, seed))
    return {
        "scenario": scenario, "episodes": episodes,
        "random_mean": float(np.mean(random_returns)), "random_std": float(np.std(random_returns)),
        "scripted_mean": float(np.mean(scripted_returns)), "scripted_std": float(np.std(scripted_returns)),
    }


def load(path=OUTPUT):
    """{scenario: {"random": mean, "scripted": mean}} from a saved CSV (empty if missing)."""
    if not Path(path).exists():
        return {}
    with open(path) as f:
        return {r["scenario"]: {"random": float(r["random_mean"]), "scripted": float(r["scripted_mean"])}
                for r in csv.DictReader(f)}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scenarios", nargs="+", default=list(SCENARIOS), choices=list(SCENARIOS))
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()

    with ProcessPoolExecutor(max_workers=min(len(args.scenarios), os.cpu_count() or 1)) as pool:
        rows = list(pool.map(score, args.scenarios, [args.episodes] * len(args.scenarios)))
    # Keep scores of scenarios not recomputed this time
    existing = {}
    if args.output.exists():
        with open(args.output) as f:
            existing = {r["scenario"]: r for r in csv.DictReader(f)}
    existing.update({r["scenario"]: r for r in rows})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = ["scenario", "episodes", "random_mean", "random_std", "scripted_mean", "scripted_std"]
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for name in SCENARIOS:
            if name in existing:
                writer.writerow({k: existing[name][k] for k in fields})
    for r in rows:
        print(f"{r['scenario']:24s} random {float(r['random_mean']):8.2f}   scripted {float(r['scripted_mean']):8.2f}")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
