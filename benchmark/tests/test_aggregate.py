"""Statistics and outputs of benchmark/aggregate.py."""

import csv
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
import aggregate  # noqa: E402


def test_iqm_trims_a_quarter_from_each_end():
    assert aggregate.iqm([1, 2, 3, 4, 100, -100, 5, 6]) == pytest.approx(np.mean([2, 3, 4, 5]))
    np.testing.assert_allclose(aggregate.iqm(np.array([[1, 2, 3, 4], [5, 6, 7, 80]]), axis=1), [2.5, 6.5])


def test_bootstrap_ci_brackets_the_estimate():
    rng = np.random.default_rng(0)
    data = rng.normal(10, 1, size=50)
    lo, hi = aggregate.bootstrap_ci([data], np.mean, reps=500, rng=rng)
    assert lo < data.mean() < hi and hi - lo < 1.0


def test_probability_of_improvement():
    assert aggregate.probability_of_improvement([2, 3], [0, 1]) == 1.0
    assert aggregate.probability_of_improvement([1], [1]) == 0.5
    assert aggregate.probability_of_improvement([0, 2], [1]) == 0.5


def _curve_run(seed, curve):
    return {"seed": seed, "steps": np.arange(len(curve)) * 1000, "curve": np.asarray(curve, dtype=float)}


def test_budget_rows_survives_noisy_end_and_no_gain():
    rising = list(np.minimum(np.arange(100) / 20, 1.0) * 100)
    noisy_end = rising[:-1] + [50.0]           # last evaluation dips below 95% of the gain
    declining = list(np.linspace(10, 0, 100))  # never improves on its first evaluation
    runs = {("classic_stag", "ippo"): [_curve_run(s, noisy_end) for s in range(3)],
            ("classic_stag", "mappo"): [_curve_run(s, declining) for s in range(3)]}
    row = aggregate.budget_rows(runs)[0]
    assert row["plateau_ippo"] < 30000 and row["plateaued"]
    assert row["plateau_mappo"] == "" and row["no_gain"] == "mappo"

    flat = {("classic_stag", "happo"): [_curve_run(s, declining) for s in range(3)]}
    row = aggregate.budget_rows(flat)[0]
    assert row["suggested_budget"] == "" and not row["plateaued"]

    late = list(np.minimum(np.arange(100) / 90, 1.0) * 100)  # levels off at 90% of training
    row = aggregate.budget_rows({("classic_stag", "ippo"): [_curve_run(s, late) for s in range(3)]})[0]
    assert row["suggested_budget"] == row["trained_to"] == 99000 and not row["plateaued"]


def _fake_run(root, scenario, algo, seed, final_score):
    out = root / scenario / algo / f"seed_{seed}"
    out.mkdir(parents=True)
    with open(out / "progress.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["env_steps", "eval_return_mean"])
        for i, step in enumerate(range(0, 11000, 1000)):
            writer.writerow([step, final_score * min(1.0, i / 5)])
    (out / "final.json").write_text(json.dumps({
        "scenario": scenario, "algo": algo, "seed": seed,
        "final_eval_returns": [final_score] * 4, "final_eval_captures": [{"hare": 2}] * 4,
        "final_greedy_returns": [final_score / 2] * 4, "final_greedy_captures": [{"hare": 1}] * 4,
    }))


def test_end_to_end_on_synthetic_runs(tmp_path, monkeypatch):
    results, out = tmp_path / "marl", tmp_path / "summary"
    for algo, base in (("ippo", 10.0), ("mappo", 20.0), ("happo", 30.0)):
        for seed in range(4):
            _fake_run(results, "classic_stag", algo, seed, base + seed)
    refs = tmp_path / "refs.csv"
    refs.write_text("scenario,episodes,random_mean,random_std,scripted_mean,scripted_std\n"
                    "classic_stag,10,0,0,40,0\n")
    monkeypatch.setattr(sys, "argv", ["aggregate.py", "--results", str(results), "--out", str(out),
                                      "--references", str(refs), "--bootstrap", "200"])
    aggregate.main()

    rows = {r["algo"]: r for r in csv.DictReader(open(out / "final_scores.csv"))}
    assert float(rows["happo"]["iqm"]) == pytest.approx(np.mean([31, 32]))
    assert float(rows["ippo"]["norm_iqm"]) == pytest.approx(np.mean([11, 12]) / 40)
    assert float(rows["happo"]["greedy_iqm"]) == pytest.approx(np.mean([31, 32]) / 2)
    budgets = list(csv.DictReader(open(out / "budgets.csv")))[0]
    assert budgets["plateau_ippo"] == "5000" and budgets["plateaued"] == "True"
    for name in ("final_scores.md", "final_scores.tex", "behaviour.csv", "curves/classic_stag.png", "curves/classic_stag.pdf"):
        assert (out / name).exists()
