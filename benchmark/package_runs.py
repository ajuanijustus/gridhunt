"""Package the benchmark runs (and optionally the exported zoo) for a Zenodo deposit.

Checks that every (scenario, algorithm, seed) of the protocol finished with the protocol's
budget and the current training settings, then writes to --out:

- gridhunt-benchmark-runs.tar.gz: runs/<scenario>/<algo>/seed_<n>/ with final.json,
  progress.csv, evals.jsonl and (unless --no-models) model.pt, plus the reference scores and
  the summary tables and figures;
- the zoo files from --zoo (the .npz models, registry.json and VERIFICATION.md), uploaded as
  separate files so each model has its own download link;
- README.md, a description of the deposit for its Zenodo page;
- SHA256SUMS for every file above.

    python benchmark/package_runs.py --runs benchmark/results/marl --zoo zoo_export --out zenodo_upload
"""

import argparse
import dataclasses
import hashlib
import json
import shutil
import sys
import tarfile
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from protocol import SCENARIOS  # noqa: E402

ALGORITHMS = ("ippo", "mappo", "happo")
RUN_FILES = ("final.json", "progress.csv", "evals.jsonl")
ARCHIVE = "gridhunt-benchmark-runs.tar.gz"


def current_config():
    from marl.ppo import PPOConfig

    return json.loads(json.dumps(dataclasses.asdict(PPOConfig())))


def check_runs(runs, seeds, config):
    """Return (finished run directories, problems)."""
    found, problems = [], []
    for scenario, spec in SCENARIOS.items():
        for algo in ALGORITHMS:
            for seed in seeds:
                run = runs / scenario / algo / f"seed_{seed}"
                final_path = run / "final.json"
                if not final_path.exists():
                    problems.append(f"missing: {run}")
                    continue
                final = json.loads(final_path.read_text())
                if final.get("budget") != spec["budget"]:
                    problems.append(f"{run}: budget {final.get('budget')}, protocol says {spec['budget']}")
                if config is not None and final.get("config") != config:
                    problems.append(f"{run}: trained with different settings than the current PPOConfig")
                for name in RUN_FILES:
                    if not (run / name).exists():
                        problems.append(f"{run}: no {name}")
                found.append(run)
    return found, problems


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def describe(runs_found, models, zoo_files, finals):
    jobs = sorted({f.get("slurm_job") for f in finals if f.get("slurm_job")})
    commits = sorted({f.get("git_commit") for f in finals if f.get("git_commit")})
    versions = Counter((f.get("python"), f.get("torch"), f.get("numpy")) for f in finals)
    steps = sum(f.get("env_steps", 0) for f in finals)
    seeds = sorted({f["seed"] for f in finals})
    lines = [
        "# GridHunt benchmark runs",
        "",
        "Training runs behind the GridHunt benchmark results: IPPO, MAPPO and HAPPO on every "
        f"benchmark scenario, {len(seeds)} seeds each ({len(runs_found)} runs, "
        f"{steps / 1e9:.2f} billion environment steps in total).",
        "",
        "## Contents",
        "",
        f"- `{ARCHIVE}`: `runs/<scenario>/<algo>/seed_<n>/` with",
        "  - `final.json`: final evaluation (100 episodes, sampled and greedy actions), training settings,",
        "    library versions, git commit and SLURM job id;",
        "  - `progress.csv`: evaluation every 1% of the training budget and training statistics;",
        "  - `evals.jsonl`: every evaluation episode's return and captures;",
    ]
    if models:
        lines.append("  - `model.pt`: final PyTorch actor and critic weights (load with `benchmark/marl/ppo.py: load_learner`);")
    lines += [
        "  - plus `reference_scores.csv` (random and scripted teams) and `summary/` (the published tables",
        "    and figures).",
    ]
    if zoo_files:
        lines += [
            "- The pretrained teams read by `gridhunt.zoo` (numpy `.npz` files, one per exported run),",
            "  `registry.json` with their SHA-256 hashes and `VERIFICATION.md`, the checks each passed",
            "  against its PyTorch model.",
        ]
    lines += [
        "- `SHA256SUMS`: checksums of every file.",
        "",
        "## Reproducing the tables",
        "",
        "```bash",
        "git clone https://github.com/ajuanijustus/gridhunt.git && cd gridhunt",
        'pip install -e ".[benchmark]"',
        f"tar -xzf {ARCHIVE}",
        "python benchmark/aggregate.py --results runs --out summary_check",
        "```",
        "",
        "## Provenance",
        "",
        f"- SLURM jobs: {', '.join(jobs) if jobs else 'not recorded'}",
        f"- Git commits: {', '.join(c[:10] for c in commits) if commits else 'not recorded'}",
        "- Python / PyTorch / NumPy: " + "; ".join(f"{p} / {t} / {n} ({count} runs)" for (p, t, n), count in
                                                   sorted(versions.items(), key=str)),
        "",
        "## License",
        "",
        "CC BY 4.0. Please cite the GridHunt paper and this dataset.",
        "",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs", type=Path, default=Path("benchmark/results/marl"))
    parser.add_argument("--zoo", type=Path, default=None, help="output folder of export_zoo.py")
    parser.add_argument("--out", type=Path, default=Path("zenodo_upload"))
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--no-models", action="store_true", help="leave out the model.pt files")
    parser.add_argument("--allow-incomplete", action="store_true", help="package even if runs are missing")
    args = parser.parse_args()

    runs_found, problems = check_runs(args.runs, range(args.seeds), current_config())
    if problems:
        print("\n".join(problems[:30]) + (f"\n... and {len(problems) - 30} more" if len(problems) > 30 else ""))
        if not args.allow_incomplete:
            sys.exit(f"{len(problems)} problems; fix them or pass --allow-incomplete")
    args.out.mkdir(parents=True, exist_ok=True)
    benchmark = Path(__file__).parent

    archive = args.out / ARCHIVE
    finals = []
    with tarfile.open(archive, "w:gz") as tar:
        for run in sorted(runs_found):
            relative = Path("runs") / run.relative_to(args.runs)
            names = list(RUN_FILES) + ([] if args.no_models else ["model.pt"])
            for name in names:
                if (run / name).exists():
                    tar.add(run / name, arcname=str(relative / name))
            finals.append(json.loads((run / "final.json").read_text()))
        references = benchmark / "results" / "reference_scores.csv"
        if references.exists():
            tar.add(references, arcname="reference_scores.csv")
        summary = benchmark / "results" / "summary"
        if summary.exists():
            tar.add(summary, arcname="summary", filter=lambda info: None if info.name.endswith(".DS_Store") else info)

    zoo_files = []
    if args.zoo is not None:
        for path in sorted(args.zoo.glob("*.npz")) + [args.zoo / "registry.json", args.zoo / "VERIFICATION.md"]:
            if path.exists():
                shutil.copy2(path, args.out / path.name)
                zoo_files.append(path.name)

    (args.out / "README.md").write_text(describe(runs_found, not args.no_models, zoo_files, finals))
    files = [ARCHIVE, *zoo_files, "README.md"]
    (args.out / "SHA256SUMS").write_text("".join(f"{sha256(args.out / name)}  {name}\n" for name in files))
    total = sum((args.out / name).stat().st_size for name in files)
    print(f"{len(runs_found)} runs, {len(zoo_files)} zoo files, {total / 1e6:.0f} MB -> {args.out}")


if __name__ == "__main__":
    main()
