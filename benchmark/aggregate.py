"""
Aggregate benchmark runs into tables and figures.

Reads every finished run in benchmark/results/marl/<scenario>/<algo>/seed_<n>/
(final.json and progress.csv) and writes to benchmark/results/summary/:

  final_scores.csv/.md/.tex  per scenario x algorithm: seeds, mean ± std, IQM [95% CI],
                             normalized IQM [95% CI]
  aggregate.csv/.md          across the main scenarios: normalized IQM [95% CI] per
                             algorithm and the probability that one beats another
  behaviour.csv              captures per episode by prey type
  budgets.csv                where each learning curve plateaus (for choosing budgets)
  curves/<scenario>.png/.pdf IQM learning curves with 95% CI bands
  aggregate.png/.pdf         normalized IQM with 95% CI per algorithm

Statistics follow Agarwal et al. (2021): the interquartile mean (IQM, the mean of the
middle 50% of runs) with 95% stratified bootstrap confidence intervals over seeds.
A run's score is the mean team return of its final greedy evaluation. Normalized
score = (score - random team) / (scripted team - random team), from reference_scores.py.

Usage:
    python benchmark/aggregate.py
    python benchmark/aggregate.py --results /rds/.../marl --out /rds/.../summary
"""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import reference_scores  # noqa: E402
from protocol import MAIN_SCENARIOS, SCENARIOS  # noqa: E402

ALGORITHMS = ("ippo", "mappo", "happo")
LABELS = {"ippo": "IPPO", "mappo": "MAPPO", "happo": "HAPPO"}
# Fixed categorical order (validated: CVD-safe on the light surface)
COLORS = {"ippo": "#2a78d6", "mappo": "#eb6834", "happo": "#1baf7a"}
SURFACE, TEXT, TEXT_SECONDARY, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def iqm(values, axis=-1):
    """Interquartile mean: mean of the middle 50% (25% trimmed from each end)."""
    values = np.sort(np.asarray(values, dtype=float), axis=axis)
    n = values.shape[axis]
    cut = int(np.floor(0.25 * n))
    return np.take(values, np.arange(cut, n - cut), axis=axis).mean(axis=axis)


def bootstrap_ci(groups, statistic, reps, rng, level=0.95):
    """Stratified bootstrap: resample runs within each group, pool, apply statistic."""
    groups = [np.asarray(g, dtype=float) for g in groups if len(g)]
    stats = np.empty(reps)
    for r in range(reps):
        stats[r] = statistic(np.concatenate([g[rng.integers(0, len(g), len(g))] for g in groups]))
    alpha = (1 - level) / 2
    return float(np.quantile(stats, alpha)), float(np.quantile(stats, 1 - alpha))


def probability_of_improvement(x, y):
    """P(a random run of X scores higher than a random run of Y); ties count half."""
    x, y = np.asarray(x, dtype=float)[:, None], np.asarray(y, dtype=float)[None, :]
    return float(((x > y) + 0.5 * (x == y)).mean())


def normalize(score, reference):
    if reference is None:
        return float("nan")
    span = reference["scripted"] - reference["random"]
    return (np.asarray(score, dtype=float) - reference["random"]) / span if span else float("nan")


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_runs(root):
    """{(scenario, algo): [run]} for finished runs; run = seed, score, captures, curve."""
    runs = defaultdict(list)
    for final_path in sorted(Path(root).glob("*/*/seed_*/final.json")):
        final = json.loads(final_path.read_text())
        steps, curve = [], []
        progress = final_path.parent / "progress.csv"
        if progress.exists():
            with open(progress) as f:
                for row in csv.DictReader(f):
                    steps.append(int(row["env_steps"]))
                    curve.append(float(row["eval_return_mean"]))
        runs[(final["scenario"], final["algo"])].append({
            "seed": final["seed"],
            "score": float(np.mean(final["final_eval_returns"])),
            # Greedy (argmax) evaluation, recorded since sampled evaluation became the headline
            "greedy_score": float(np.mean(final["final_greedy_returns"])) if "final_greedy_returns" in final else float("nan"),
            "captures": final["final_eval_captures"],
            "steps": np.array(steps), "curve": np.array(curve),
        })
    return runs


def aligned_curves(runs):
    """Steps shared by every run and a [runs, steps] matrix of evaluation returns."""
    common = set(runs[0]["steps"].tolist())
    for run in runs[1:]:
        common &= set(run["steps"].tolist())
    steps = np.array(sorted(common))
    matrix = np.array([[run["curve"][run["steps"] == s][0] for s in steps] for run in runs])
    return steps, matrix


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

def fmt_ci(value, lo, hi, digits=1):
    if not np.isfinite(value):
        return "–"
    return f"{value:.{digits}f} [{lo:.{digits}f}, {hi:.{digits}f}]"


def final_score_rows(runs, references, reps, rng):
    rows = []
    for scenario in SCENARIOS:
        for algo in ALGORITHMS:
            group = runs.get((scenario, algo))
            if not group:
                continue
            scores = np.array([r["score"] for r in group])
            greedy = np.array([r.get("greedy_score", float("nan")) for r in group])
            ref = references.get(scenario)
            value, (lo, hi) = iqm(scores), bootstrap_ci([scores], iqm, reps, rng)
            norm = normalize(scores, ref)
            if ref is not None:
                norm_value, (norm_lo, norm_hi) = iqm(norm), bootstrap_ci([norm], iqm, reps, rng)
            else:
                norm_value = norm_lo = norm_hi = float("nan")
            rows.append({
                "scenario": scenario, "algo": algo, "seeds": len(scores),
                "mean": scores.mean(), "std": scores.std(ddof=1) if len(scores) > 1 else 0.0,
                "iqm": value, "iqm_lo": lo, "iqm_hi": hi,
                "norm_iqm": norm_value, "norm_lo": norm_lo, "norm_hi": norm_hi,
                "greedy_iqm": iqm(greedy) if not np.isnan(greedy).any() else float("nan"),
                "random": ref["random"] if ref else float("nan"),
                "scripted": ref["scripted"] if ref else float("nan"),
            })
    return rows


def aggregate_rows(runs, references, reps, rng):
    """Normalized IQM over the main scenarios, plus pairwise probability of improvement."""
    scenarios = [s for s in MAIN_SCENARIOS if s in references and all((s, a) in runs for a in ALGORITHMS)]
    per_algo = {
        algo: [normalize([r["score"] for r in runs[(s, algo)]], references[s]) for s in scenarios]
        for algo in ALGORITHMS
    }
    rows = []
    for algo, groups in per_algo.items():
        if not groups:
            continue
        pooled = np.concatenate(groups)
        lo, hi = bootstrap_ci(groups, iqm, reps, rng)
        rows.append({"algo": algo, "scenarios": len(groups), "norm_iqm": iqm(pooled), "lo": lo, "hi": hi})
    pairs = []
    for i, a in enumerate(ALGORITHMS):
        for b in ALGORITHMS[i + 1:]:
            if not scenarios:
                continue
            value = np.mean([probability_of_improvement(x, y) for x, y in zip(per_algo[a], per_algo[b])])
            boot = []
            for _ in range(reps // 4):
                boot.append(np.mean([
                    probability_of_improvement(x[rng.integers(0, len(x), len(x))], y[rng.integers(0, len(y), len(y))])
                    for x, y in zip(per_algo[a], per_algo[b])
                ]))
            pairs.append({"x": a, "y": b, "p": value, "lo": float(np.quantile(boot, 0.025)),
                          "hi": float(np.quantile(boot, 0.975))})
    return rows, pairs, scenarios


def behaviour_rows(runs):
    rows = []
    for (scenario, algo), group in sorted(runs.items()):
        totals = defaultdict(list)
        for run in group:
            per_type = defaultdict(float)
            for episode in run["captures"]:
                for prey_type, count in episode.items():
                    per_type[prey_type] += count
            for prey_type, count in per_type.items():
                totals[prey_type].append(count / max(len(run["captures"]), 1))
        for prey_type, values in sorted(totals.items()):
            values = values + [0.0] * (len(group) - len(values))  # runs that never caught it
            rows.append({"scenario": scenario, "algo": algo, "prey": prey_type,
                         "captures_per_episode": float(np.mean(values))})
    return rows


def trailing_mean(values, window):
    """Mean of each value and the window - 1 values before it."""
    sums = np.cumsum(np.insert(np.asarray(values, dtype=float), 0, 0.0))
    starts = np.maximum(np.arange(len(values)) + 1 - window, 0)
    return (sums[1:] - sums[starts]) / (np.arange(len(values)) + 1 - starts)


def budget_rows(runs, threshold=0.95):
    """First step after which an algorithm's IQM curve stays above 95% of its gain.

    The curve is smoothed over 5% of training, and "stays above" means at least 90% of the
    later evaluations are above the target, so a few noisy evaluations near the end do not
    move the plateau. An algorithm whose curve never settles above the target gets the last step
    (so the scenario is reported as not plateaued); one that does not improve on its first
    evaluation is listed under no_gain and ignored for the budget.
    """
    rows = []
    for scenario in SCENARIOS:
        plateaus, no_gain, last = {}, [], 0
        for algo in ALGORITHMS:
            group = runs.get((scenario, algo))
            if not group:
                continue
            steps, matrix = aligned_curves(group)
            if len(steps) < 3:
                continue
            last = max(last, int(steps[-1]))
            curve = iqm(matrix, axis=0)
            final = curve[-max(1, len(curve) // 10):].mean()
            if final <= curve[0]:
                no_gain.append(algo)
                continue
            smooth = trailing_mean(curve, max(1, len(curve) // 20))
            target = curve[0] + threshold * (final - curve[0])
            above = smooth >= target
            # Fraction of evaluations from each step onwards that are above the target
            frac_after = np.flip(np.cumsum(np.flip(above))) / np.arange(len(above), 0, -1)
            settled = above & (frac_after >= 0.9)
            idx = int(np.argmax(settled)) if settled.any() else len(curve) - 1
            plateaus[algo] = int(steps[idx])
        if plateaus or no_gain:
            latest = max(plateaus.values(), default=0)
            rows.append({"scenario": scenario, **{f"plateau_{a}": plateaus.get(a, "") for a in ALGORITHMS},
                         "no_gain": " ".join(no_gain), "trained_to": last,
                         "plateaued": bool(plateaus) and latest <= 0.8 * last,
                         # 1.5x the latest plateau, never more than the pilot trained for
                         "suggested_budget": min(int(np.ceil(1.5 * latest / 1e5) * 1e5), last) if plateaus else ""})
    return rows


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_final_tables(out, rows):
    write_csv(out / "final_scores.csv", rows)
    md = ["| Scenario | Algorithm | Seeds | Mean ± std | IQM [95% CI] | Normalized IQM [95% CI] | Greedy IQM |",
          "|---|---|---|---|---|---|---|"]
    tex = [r"\begin{tabular}{llrrrr}", r"\toprule",
           r"Scenario & Algorithm & Seeds & Mean $\pm$ std & IQM [95\% CI] & Norm. IQM [95\% CI] \\", r"\midrule"]
    for r in rows:
        mean_std = f"{r['mean']:.1f} ± {r['std']:.1f}"
        iqm_text = fmt_ci(r["iqm"], r["iqm_lo"], r["iqm_hi"])
        norm_text = fmt_ci(r["norm_iqm"], r["norm_lo"], r["norm_hi"], digits=2)
        greedy_text = "" if np.isnan(r["greedy_iqm"]) else f"{r['greedy_iqm']:.1f}"
        md.append(f"| {r['scenario']} | {LABELS[r['algo']]} | {r['seeds']} | {mean_std} | {iqm_text} | {norm_text} | {greedy_text} |")
        mean_std_tex = mean_std.replace("±", r"$\pm$")
        scenario_tex = r["scenario"].replace("_", " ")
        tex.append(f"{scenario_tex} & {LABELS[r['algo']]} & {r['seeds']} & {mean_std_tex} & {iqm_text} & {norm_text} " + r"\\")
    tex += [r"\bottomrule", r"\end{tabular}"]
    (out / "final_scores.md").write_text("\n".join(md) + "\n")
    (out / "final_scores.tex").write_text("\n".join(tex) + "\n")


def write_aggregate_tables(out, rows, pairs, scenarios):
    write_csv(out / "aggregate.csv", rows)
    write_csv(out / "probability_of_improvement.csv", pairs)
    md = [f"Normalized IQM over {len(scenarios)} main scenarios ({', '.join(scenarios)})", "",
          "| Algorithm | Normalized IQM [95% CI] |", "|---|---|"]
    md += [f"| {LABELS[r['algo']]} | {fmt_ci(r['norm_iqm'], r['lo'], r['hi'], 2)} |" for r in rows]
    md += ["", "| P(X > Y) | Estimate [95% CI] |", "|---|---|"]
    md += [f"| {LABELS[p['x']]} > {LABELS[p['y']]} | {fmt_ci(p['p'], p['lo'], p['hi'], 2)} |" for p in pairs]
    (out / "aggregate.md").write_text("\n".join(md) + "\n")


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def _style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _steps_formatter(value, _):
    if value >= 1e6:
        return f"{value / 1e6:g}M"
    if value >= 1e3:
        return f"{value / 1e3:g}K"
    return f"{value:g}"


def plot_curves(out, runs, references, reps, rng):
    (out / "curves").mkdir(exist_ok=True)
    for scenario in SCENARIOS:
        algos = [a for a in ALGORITHMS if runs.get((scenario, a))]
        if not algos:
            continue
        fig, ax = plt.subplots(figsize=(6.4, 3.8), facecolor=SURFACE)
        _style(ax)
        ends = []
        for algo in algos:
            steps, matrix = aligned_curves(runs[(scenario, algo)])
            if len(steps) == 0:
                continue
            centre = iqm(matrix, axis=0)
            idx = rng.integers(0, len(matrix), (reps // 4, len(matrix)))
            boot = iqm(matrix[idx], axis=1)  # [reps, steps]
            lo, hi = np.quantile(boot, 0.025, axis=0), np.quantile(boot, 0.975, axis=0)
            ax.fill_between(steps, lo, hi, color=COLORS[algo], alpha=0.15, linewidth=0)
            ax.plot(steps, centre, color=COLORS[algo], linewidth=1.6, solid_capstyle="round",
                    label=f"{LABELS[algo]} (n={len(matrix)})")
            ax.plot(steps[-1], centre[-1], "o", color=COLORS[algo], markersize=5,
                    markeredgecolor=SURFACE, markeredgewidth=1.5)
            ends.append((centre[-1], algo, steps[-1]))
        ref = references.get(scenario)
        if ref:
            for value, name in ((ref["scripted"], "scripted team"), (ref["random"], "random team")):
                ax.axhline(value, color=TEXT_SECONDARY, linewidth=0.8, zorder=0)
                ax.annotate(name, (1.0, value), xycoords=("axes fraction", "data"), xytext=(4, 0),
                            textcoords="offset points", va="center", fontsize=8, color=TEXT_SECONDARY)
        for value, _algo, step in ends:  # direct labels in text colour
            ax.annotate(f"{value:.0f}", (step, value), xytext=(6, 0), textcoords="offset points",
                        va="center", fontsize=8, color=TEXT)
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(_steps_formatter))
        ax.set_xlabel("Environment steps", color=TEXT_SECONDARY, fontsize=9)
        ax.set_ylabel("Team return (IQM, 95% CI)", color=TEXT_SECONDARY, fontsize=9)
        ax.set_title(scenario.replace("_", " ").title(), color=TEXT, fontsize=11, loc="left")
        legend = ax.legend(frameon=False, fontsize=8, loc="upper left")
        for text in legend.get_texts():
            text.set_color(TEXT)
        fig.tight_layout()
        for suffix in ("png", "pdf"):
            fig.savefig(out / "curves" / f"{scenario}.{suffix}", dpi=200, facecolor=SURFACE)
        plt.close(fig)


def plot_aggregate(out, rows, scenarios):
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(6.4, 0.6 + 0.55 * len(rows)), facecolor=SURFACE)
    _style(ax)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    for i, r in enumerate(rows):
        y = len(rows) - 1 - i
        ax.plot([r["lo"], r["hi"]], [y, y], color=COLORS[r["algo"]], linewidth=6, solid_capstyle="round", alpha=0.35)
        ax.plot(r["norm_iqm"], y, "o", color=COLORS[r["algo"]], markersize=8, markeredgecolor=SURFACE, markeredgewidth=2)
        ax.annotate(f"{r['norm_iqm']:.2f}", (r["hi"], y), xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=8, color=TEXT)
    for value, name in ((0, "random"), (1, "scripted")):
        ax.axvline(value, color=TEXT_SECONDARY, linewidth=0.8, zorder=0)
        ax.annotate(name, (value, 1.0), xycoords=("data", "axes fraction"), xytext=(0, 3),
                    textcoords="offset points", ha="center", fontsize=8, color=TEXT_SECONDARY)
    ax.set_yticks(range(len(rows)), [LABELS[r["algo"]] for r in reversed(rows)], color=TEXT)
    ax.set_xlabel(f"Normalized IQM over {len(scenarios)} scenarios (95% CI)", color=TEXT_SECONDARY, fontsize=9)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(out / f"aggregate.{suffix}", dpi=200, facecolor=SURFACE)
    plt.close(fig)


# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", type=Path, default=Path("benchmark/results/marl"))
    parser.add_argument("--out", type=Path, default=Path("benchmark/results/summary"))
    parser.add_argument("--references", type=Path, default=reference_scores.OUTPUT)
    parser.add_argument("--bootstrap", type=int, default=2000, help="bootstrap resamples")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    runs = load_runs(args.results)
    if not runs:
        raise SystemExit(f"No finished runs under {args.results}")
    references = reference_scores.load(args.references)
    args.out.mkdir(parents=True, exist_ok=True)

    final_rows = final_score_rows(runs, references, args.bootstrap, rng)
    write_final_tables(args.out, final_rows)
    agg_rows, pairs, scenarios = aggregate_rows(runs, references, args.bootstrap, rng)
    write_aggregate_tables(args.out, agg_rows, pairs, scenarios)
    write_csv(args.out / "behaviour.csv", behaviour_rows(runs))
    write_csv(args.out / "budgets.csv", budget_rows(runs))
    plot_curves(args.out, runs, references, args.bootstrap, rng)
    plot_aggregate(args.out, agg_rows, scenarios)

    print((args.out / "final_scores.md").read_text())
    if agg_rows:
        print((args.out / "aggregate.md").read_text())
    print(f"Wrote tables and figures to {args.out}")


if __name__ == "__main__":
    main()
