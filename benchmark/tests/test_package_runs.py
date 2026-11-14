"""package_runs.py: completeness checks, archive contents and checksums."""

import dataclasses
import hashlib
import json
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

pytest.importorskip("torch")
BENCHMARK = Path(__file__).parent.parent
sys.path.insert(0, str(BENCHMARK))

from marl.ppo import PPOConfig  # noqa: E402
from protocol import SCENARIOS  # noqa: E402

CONFIG = json.loads(json.dumps(dataclasses.asdict(PPOConfig())))


def _fake_runs(root, seeds=1):
    for scenario, spec in SCENARIOS.items():
        for algo in ("ippo", "mappo", "happo"):
            for seed in range(seeds):
                run = root / scenario / algo / f"seed_{seed}"
                run.mkdir(parents=True)
                (run / "final.json").write_text(json.dumps({
                    "scenario": scenario, "algo": algo, "seed": seed, "budget": spec["budget"], "config": CONFIG,
                    "env_steps": spec["budget"], "slurm_job": "123", "git_commit": "abcdef1234567",
                    "python": "3.12", "torch": "2.5.1", "numpy": "2.4.4",
                }))
                (run / "progress.csv").write_text("env_steps\n0\n")
                (run / "evals.jsonl").write_text("{}\n")
                (run / "model.pt").write_bytes(b"weights")
                (run / "checkpoint.pt").write_bytes(b"not packaged")


def _package(runs, out, *extra):
    return subprocess.run([sys.executable, str(BENCHMARK / "package_runs.py"), "--runs", str(runs), "--out", str(out),
                           "--seeds", "1", *extra], capture_output=True, text=True)


def test_complete_runs_are_packaged_with_checksums(tmp_path):
    runs, out = tmp_path / "runs", tmp_path / "out"
    _fake_runs(runs)
    zoo = tmp_path / "zoo"
    zoo.mkdir()
    (zoo / "classic_stag_ippo_seed0.npz").write_bytes(b"npz")
    (zoo / "registry.json").write_text("{}")
    result = _package(runs, out, "--zoo", str(zoo))
    assert result.returncode == 0, result.stdout + result.stderr

    with tarfile.open(out / "gridhunt-benchmark-runs.tar.gz") as tar:
        names = tar.getnames()
    assert "runs/classic_stag/ippo/seed_0/model.pt" in names
    assert "runs/swarm/happo/seed_0/final.json" in names
    assert not any(name.endswith("checkpoint.pt") for name in names)
    assert "reference_scores.csv" in names and any(name.startswith("summary/") for name in names)
    assert len([n for n in names if n.endswith("final.json")]) == len(SCENARIOS) * 3

    sums = dict(line.split("  ")[::-1] for line in (out / "SHA256SUMS").read_text().splitlines())
    assert set(sums) == {"gridhunt-benchmark-runs.tar.gz", "classic_stag_ippo_seed0.npz", "registry.json", "README.md"}
    for name, digest in sums.items():
        assert hashlib.sha256((out / name).read_bytes()).hexdigest() == digest
    readme = (out / "README.md").read_text()
    assert f"{len(SCENARIOS) * 3} runs" in readme and "123" in readme


def test_missing_runs_or_other_budgets_stop_packaging(tmp_path):
    runs = tmp_path / "runs"
    _fake_runs(runs)
    (runs / "raid" / "mappo" / "seed_0" / "final.json").unlink()
    final = runs / "swarm" / "ippo" / "seed_0" / "final.json"
    final.write_text(json.dumps({**json.loads(final.read_text()), "budget": 1}))
    result = _package(runs, tmp_path / "out")
    assert result.returncode != 0
    assert "missing" in result.stdout and "raid/mappo/seed_0" in result.stdout
    assert "budget 1" in result.stdout
    assert _package(runs, tmp_path / "out2", "--allow-incomplete").returncode == 0
