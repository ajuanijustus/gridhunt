"""
Write one training command per (scenario, algorithm, seed) for a SLURM array job.

Runs that already have a final.json are skipped, so regenerating the list after a
partial array (timeouts, preemption) only resubmits what is missing; unfinished runs
resume from their checkpoint.

Usage (from the repository root):
    # Pilot: 3 seeds, 5M steps, to find where curves plateau
    python benchmark/slurm/make_jobs.py --scenarios all --seeds 0-2 --budget 5000000 --out pilot_jobs.txt
    # Full benchmark: budgets from protocol.py, 20 seeds
    python benchmark/slurm/make_jobs.py --scenarios main --seeds 0-19 --out jobs.txt
    sbatch --array=1-$(wc -l < jobs.txt)%100 benchmark/slurm/run_array.sh jobs.txt
"""

import argparse
import dataclasses
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from marl.ppo import PPOConfig  # noqa: E402
from protocol import MAIN_SCENARIOS, SCENARIOS  # noqa: E402

# Training settings as final.json records them (JSON turns tuples into lists)
CURRENT_CONFIG = json.loads(json.dumps(dataclasses.asdict(PPOConfig())))

ALGORITHMS = ("ippo", "mappo", "happo")


def parse_seeds(text):
    """'0-19' or '0,3,5' -> list of ints."""
    seeds = []
    for part in text.split(","):
        if "-" in part:
            start, end = part.split("-")
            seeds.extend(range(int(start), int(end) + 1))
        else:
            seeds.append(int(part))
    return seeds


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scenarios", nargs="+", default=["main"],
                        help="scenario names, or 'main', 'variants' or 'all'")
    parser.add_argument("--algos", nargs="+", default=list(ALGORITHMS), choices=ALGORITHMS)
    parser.add_argument("--seeds", default="0-19")
    parser.add_argument("--budget", type=int, default=None, help="override every scenario's budget (e.g. for the pilot)")
    parser.add_argument("--results", type=Path, default=Path("benchmark/results/marl"))
    parser.add_argument("--out", type=Path, default=Path("jobs.txt"))
    parser.add_argument("--extra", default="", help="extra arguments passed to train_marl.py")
    args = parser.parse_args()

    groups = {"main": MAIN_SCENARIOS, "variants": [s for s in SCENARIOS if s not in MAIN_SCENARIOS], "all": list(SCENARIOS)}
    scenarios = []
    for name in args.scenarios:
        scenarios.extend(groups.get(name, [name]))
    unknown = [s for s in scenarios if s not in SCENARIOS]
    if unknown:
        raise SystemExit(f"Unknown scenarios: {unknown}")

    lines, skipped = [], 0
    for scenario in scenarios:
        budget = args.budget or SCENARIOS[scenario]["budget"]
        if not budget:
            raise SystemExit(f"No budget for {scenario}: set it in protocol.py or pass --budget")
        for algo in args.algos:
            for seed in parse_seeds(args.seeds):
                out = args.results / scenario / algo / f"seed_{seed}"
                if (out / "final.json").exists():
                    finished = json.loads((out / "final.json").read_text())
                    if finished.get("budget") != budget:
                        raise SystemExit(
                            f"{out} already holds a finished run with budget {finished.get('budget')}, not {budget}. "
                            f"Move earlier results (e.g. the pilot) aside or pass a different --results.")
                    if not args.extra and finished.get("config") != CURRENT_CONFIG:
                        changed = sorted(k for k in CURRENT_CONFIG
                                         if finished.get("config", {}).get(k) != CURRENT_CONFIG[k])
                        raise SystemExit(
                            f"{out} already holds a finished run trained with different settings ({', '.join(changed)}). "
                            f"Move earlier results aside or pass a different --results.")
                    skipped += 1
                    continue
                lines.append(f"python benchmark/train_marl.py --scenario {scenario} --algo {algo} --seed {seed} "
                             f"--budget {budget} --out {out} {args.extra}".strip())
    args.out.write_text("\n".join(lines) + ("\n" if lines else ""))
    print(f"Wrote {len(lines)} jobs to {args.out} ({skipped} already finished)")
    if lines:
        print(f"Submit with: sbatch --array=1-{len(lines)}%100 benchmark/slurm/run_array.sh {args.out}")


if __name__ == "__main__":
    main()
