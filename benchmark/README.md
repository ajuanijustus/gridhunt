# GridHunt Benchmark

Baselines for every scenario: IPPO, MAPPO and HAPPO trained under one fixed protocol, 20 seeds
each, evaluated on held-out seeds and reported as IQM with 95% stratified bootstrap confidence
intervals. The environments are in [`src/gridhunt/`](../src/gridhunt); this folder holds the
protocol, the baselines, the analysis scripts and the results. It is not part of the pip package.

| File | Purpose |
|---|---|
| [`protocol.py`](protocol.py) | How each scenario is run for the benchmark (team reward, episode length, budget, variants) |
| [`marl/ppo.py`](marl/ppo.py) | IPPO, MAPPO and HAPPO in one PPO implementation |
| [`marl/vec_env.py`](marl/vec_env.py) | Batched environment copies with time-limit bootstrapping |
| [`train_marl.py`](train_marl.py) | Train one (scenario, algorithm, seed); resumable |
| [`reference_policies.py`](reference_policies.py) | Random and scripted reference teams (from `gridhunt.policies`) |
| [`reference_scores.py`](reference_scores.py) | Random and scripted scores used to normalize results |
| [`interdependence.py`](interdependence.py) | Does each scenario need every agent? (scripted-team ablation) |
| [`aggregate.py`](aggregate.py) | Tables, learning curves and cross-scenario statistics |
| [`export_zoo.py`](export_zoo.py) | Export trained teams for `gridhunt.zoo`, checked against the PyTorch models |
| [`package_runs.py`](package_runs.py) | Package the runs and the zoo for the Zenodo deposit |
| [`requirements.txt`](requirements.txt) | Exact library versions behind the published results |
| [`slurm/`](slurm/) | Job list generator and SLURM scripts for any cluster |
| [`results/`](results/) | Reference scores and the summary tables and figures |

## Results

Normalized IQM over the 8 main scenarios, `(score - random) / (scripted - random)`:

| Algorithm | Normalized IQM [95% CI] |
|---|---|
| IPPO | 0.71 [0.67, 0.75] |
| MAPPO | 0.76 [0.74, 0.77] |
| HAPPO | 0.76 [0.74, 0.78] |

Per-scenario scores are in [`results/summary/final_scores.md`](results/summary/final_scores.md)
(also as CSV and LaTeX), learning curves in [`results/summary/curves/`](results/summary/curves),
probabilities of improvement in `probability_of_improvement.csv`, prey captured per episode in
`behaviour.csv`, plateau points in `budgets.csv` and the interdependence ablation in
`interdependence.csv`.

Things to know when reading them:

- **The scripted reference is not an upper bound.** It reads the true state and walks greedily to
  the best prey, so trained teams beat it where splitting up pays (ToxicStag, Specialists,
  LandAndSea: normalized scores above 1) and fall far short where it coordinates on the stag
  (ClassicStag 0.38, Swarm 0.05, where the trained teams settle for hares).
- **Raid:** every algorithm takes the barn and the house and skips the bank, which needs all three
  raiders and is trapped half the time (expected value 20 - 0.5 × 25 = 7.5 per raid).
- **Pig chase:** every seed solves it early. Seeds solved at the end: IPPO 14/20, MAPPO 18/20,
  HAPPO 19/20; the rest collapsed and never recovered, so the IPPO interval is wide.
- **Greedy evaluation:** Military and Raid score much lower with greedy (argmax) actions than with
  sampled ones, because deterministic grid policies get stuck.

## Protocol

- **Reward:** `reward_mode="team"`, so every predator receives the team reward. The metric is the
  **team return**, the sum of `infos["__common__"]["team_reward"]` over an episode, with each capture
  counted once.
- **Episodes:** 200 steps (Swarm 300). Prey are not trainable. They are static, except in Specialists,
  where they flee the nearest predator.
- **Observations:** each scenario's default, except Raid and Swarm, which use `flat_coords`
  (entity coordinates). With their default one-hot grids (870 and 9,528 inputs per agent), every
  baseline stayed at or below the random team in a 5M-step pilot.
- **Evaluation:** actions sampled from the policy, on 32 fixed held-out seeds (`EVAL_SEED_OFFSET`)
  every 1% of the budget, and 100 episodes at the end. A run's score is its final evaluation mean.
  Sampling uses its own seeded generator per episode, so it is reproducible and does not affect
  training. Greedy (argmax) scores are recorded alongside (`greedy_return_mean`,
  `final_greedy_returns`, the `Greedy IQM` column).
- **Statistics:** IQM over seeds with 95% stratified bootstrap CIs (2,000 resamples). Results are
  also normalized, `(score - random) / (scripted - random)`, with the reference scores in
  [`results/reference_scores.csv`](results/reference_scores.csv) (100 episodes each).
- **Budgets:** set per scenario in `protocol.py` from a 20M-step, 3-seed pilot, identical for all
  algorithms: 1.5× the step where the slowest algorithm levels off, rounded up to a whole million
  and capped at 20M. Variants use their main scenario's budget, so each hypothesis test compares
  equally trained runs. ClassicStag gets 3M, Raid and LandAndSea 7M, Military and its variant 50M
  (Military was still rising at 20M), everything else 20M. In the final runs every curve levelled
  off within its budget except Pig chase (IPPO, whose collapsing seeds keep the mean moving) and
  Swarm (HAPPO, still rising at 20M), which are reported as not converged (`budgets.csv`).

| Algorithm | Actor | Critic |
|---|---|---|
| IPPO | one per agent | one per agent, on its own observation |
| MAPPO | one shared (observation + agent one-hot) | one, on the global state `env.state()` |
| HAPPO | one per agent, updated sequentially in random order | one, on the global state |

All three share rollouts, GAE (λ=0.95, γ=0.99), value normalization, clipping (0.2), 5 epochs of 4
minibatches, Adam (lr 5e-4), entropy 0.01 and a 128×128 tanh MLP (`marl/ppo.py: PPOConfig`).

Advantages are standardized per batch, as in most PPO implementations, but never divided by less
than 0.05 × the critic's running return std (`adv_std_floor`). Without the floor, once a deterministic
task is solved every trajectory in a batch is the same, the advantages are only critic noise, and
per-batch scaling blows that noise up to full size. In Pig chase this broke the solved policy in most
seeds: in a 4M-step MAPPO test, 0 of 8 seeds still scored 50 at the end without the floor, against
8 of 8 with it. The floor binds only when a batch's advantage spread is tiny: in the final runs it
was never active in 9 of the 12 scenarios, and at most 0.4% of updates in ClassicStag and 10% in
Raid. Each run logs it (`adv_floor_active` in `progress.csv`, `adv_floor_active_frac` in `final.json`).

### Scenario variants (hypothesis tests)

| Variant | Change | Question |
|---|---|---|
| `specialists_comm` | 4-token message channel (`communication=4`) | Can explicit communication replace the Bruiser's missing vision? |
| `specialists_full_vision` | Bruiser sees the whole board (same encoding) | How much is the information gap costing? |
| `toxic_stag_blind` | Nobody knows if the stag is toxic | Does predator_0's knowledge matter? |
| `military_full_vision` | Ground vehicle sees the whole board | Is the UAV's scouting needed? |

## Reproducing

All commands run from the repository root.

### Environment

```bash
pip install -e ".[benchmark]"                                   # current library versions
pip install -r benchmark/requirements.txt && pip install -e . --no-deps   # or the exact published versions
```

### One run, reference scores and analysis

```bash
# One run (writes benchmark/results/marl/<scenario>/<algo>/seed_<n>/)
python benchmark/train_marl.py --scenario specialists --algo happo --seed 0

# Random and scripted reference scores, and the interdependence ablation (scripted teams, minutes)
python benchmark/reference_scores.py
python benchmark/interdependence.py

# Tables and figures from all finished runs (benchmark/results/summary/)
python benchmark/aggregate.py
```

`train_marl.py` uses the scenario's budget from `protocol.py` unless `--budget` is given, and
resumes from its checkpoint if interrupted. A full set is 12 scenarios × 3 algorithms × 20 seeds =
720 runs on one CPU each. The 20M-step pilot runs of Military took about 4 hours on BlueBEAR,
so its 50M-step runs need more than SLURM's usual 10-hour limit (the benchmark used 36 hours).

### The full benchmark on a SLURM cluster

1. **Cluster settings (once):** copy `benchmark/slurm/cluster.env.example` to
   `benchmark/slurm/cluster.env` and fill in the account, log folder, how to load conda and where
   to create the environment. Every script is submitted through `benchmark/slurm/submit.sh`, which
   adds these settings.
2. **Environment (once):** creates the environment with the exact published versions and runs the
   test suites.
   ```bash
   benchmark/slurm/submit.sh benchmark/slurm/setup_env.sh
   ```
3. **Runs:** `make_jobs.py` writes one command per run that has not finished yet, and refuses to
   reuse a finished run whose budget or training settings (`PPOConfig`) differ from the current
   ones, so stale runs cannot slip into the results.
   ```bash
   python benchmark/slurm/make_jobs.py --scenarios all --seeds 0-19 --out jobs.txt
   benchmark/slurm/submit.sh --time=1-12:00:00 --array=1-$(wc -l < jobs.txt)%100 benchmark/slurm/run_array.sh jobs.txt
   ```
   If tasks time out, run both commands again: finished runs are skipped and unfinished ones
   resume from their checkpoints.
4. **Tables:** `python benchmark/aggregate.py` (with the environment activated).

Each run records the git commit, library versions and SLURM job id in its `final.json`.

The budgets came from a pilot run the same way, into its own folder:

```bash
python benchmark/slurm/make_jobs.py --scenarios all --seeds 0-2 --budget 20000000 \
    --results benchmark/results/pilot20m --out pilot_jobs.txt
python benchmark/aggregate.py --results benchmark/results/pilot20m --out benchmark/results/pilot20m_summary
```

`budgets.csv` then suggests 1.5× the step where each curve levels off; `plateaued=False` means a
scenario was still improving at the end of the pilot.

### From the published runs

The 720 runs (evaluations, training curves and final weights) are archived on Zenodo as
`gridhunt-benchmark-runs.tar.gz`, together with the zoo models. To rebuild the tables from them:

```bash
tar -xzf gridhunt-benchmark-runs.tar.gz
python benchmark/aggregate.py --results runs --out summary_check
```

`package_runs.py` builds that deposit from a results folder, after checking that every run
finished with the protocol's budget and the current training settings.

## Pretrained teams (gridhunt.zoo)

`export_zoo.py` turns finished runs into the numpy models that `gridhunt.zoo` downloads. For each
scenario and algorithm it exports the run with the median final score and the best run, and
checks every file against its PyTorch model: equal logits on observations from real episodes,
greedy actions that reproduce the greedy evaluation returns in `final.json` episode by episode,
and a sampled mean within 4 standard errors of the recorded score.

```bash
python benchmark/export_zoo.py --runs benchmark/results/marl --out zoo_export
benchmark/slurm/submit.sh benchmark/slurm/export_zoo.sh benchmark/results/marl zoo_export   # on a cluster
```

It writes the `.npz` files, `registry.json` (copied to `src/gridhunt/zoo/registry.json`) and
`VERIFICATION.md`.

## Earlier RLlib pipeline

The earlier preprint trained its baselines with RLlib, using scripts that are not part of this
repository. Their "MAPPO" had no centralized critic and their LSTM settings were never applied, so
their numbers are superseded by this benchmark.
