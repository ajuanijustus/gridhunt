"""
Interdependence ablation: does each scenario actually need every agent?

For each scenario, the scripted GreedyPolicy team plays the benchmark protocol on
fixed evaluation seeds. Then each agent in turn is removed (it stays still) while
the remaining agents re-plan for the team they actually have (they ignore prey the
remaining capture power cannot take). The drop in team return versus the full team
is that agent's interdependence: a large drop means the scenario needs that agent,
no drop means it can be solved without it.

The scripted team reads the true state, so this measures capture and role
interdependence (who is needed to capture what), not information interdependence
(who needs to see what). That is measured again with trained policies later.

Usage:
    python benchmark/interdependence.py                 # all scenarios
    python benchmark/interdependence.py --episodes 100 --scenarios toxic_stag raid
"""

import argparse
import csv
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from protocol import EVAL_SEED_OFFSET, MAIN_SCENARIOS, SCENARIOS, make_env
from reference_policies import GreedyPolicy, IdlePolicy, RandomPolicy, run_episode

OUTPUT_DIR = Path("benchmark/results/interdependence")


def mean_ci(values):
    """Mean and 95% confidence half-width (normal approximation)."""
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        return float(values.mean()), 0.0
    return float(values.mean()), float(1.96 * values.std(ddof=1) / np.sqrt(len(values)))


def evaluate(scenario, make_policies, episodes):
    env = make_env(scenario)
    returns = []
    for episode in range(episodes):
        returns.append(run_episode(env, make_policies(env, episode), seed=EVAL_SEED_OFFSET + episode))
    return returns


def ablate(scenario, episodes):
    """Rows of (condition, removed agent, mean, ci, drop vs full team)."""
    agents = make_env(scenario).possible_agents

    def team(removed=None):
        active = [agent for agent in agents if agent != removed]
        greedy = GreedyPolicy(active_agents=active)
        return lambda env, episode: {
            agent: (IdlePolicy() if agent == removed else greedy) for agent in agents
        }

    full_mean, full_ci = mean_ci(evaluate(scenario, team(), episodes))

    def drop(mean):
        return (full_mean - mean) / abs(full_mean) if full_mean else float("nan")

    rows = [("greedy team", "-", full_mean, full_ci, 0.0)]
    for agent in agents:
        mean, ci = mean_ci(evaluate(scenario, team(removed=agent), episodes))
        rows.append((f"without {agent}", agent, mean, ci, drop(mean)))
    random_returns = evaluate(
        scenario,
        lambda env, episode: {agent: RandomPolicy(seed=episode * 100 + i) for i, agent in enumerate(agents)},
        episodes,
    )
    mean, ci = mean_ci(random_returns)
    rows.append(("all random", "-", mean, ci, drop(mean)))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--scenarios", nargs="+", default=MAIN_SCENARIOS, choices=list(SCENARIOS))
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT_DIR / "interdependence.csv"
    # One process per scenario
    with ProcessPoolExecutor(max_workers=min(len(args.scenarios), os.cpu_count() or 1)) as pool:
        results = dict(zip(args.scenarios, pool.map(ablate, args.scenarios, [args.episodes] * len(args.scenarios))))
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["scenario", "condition", "removed_agent", "team_return_mean", "ci95", "drop_vs_full"])
        for scenario in args.scenarios:
            print(f"\n## {scenario} ({args.episodes} episodes, max_steps={SCENARIOS[scenario]['max_steps']})")
            print(f"{'condition':34s} {'team return':>18s} {'drop':>8s}")
            for condition, agent, mean, ci, drop in results[scenario]:
                writer.writerow([scenario, condition, agent, f"{mean:.3f}", f"{ci:.3f}", f"{drop:.4f}"])
                print(f"{condition:34s} {mean:9.2f} ± {ci:6.2f} {drop:8.1%}")
    print(f"\nSaved {csv_path}")


if __name__ == "__main__":
    main()
