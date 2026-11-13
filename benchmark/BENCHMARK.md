# GridHunt Benchmark

Baselines for every scenario: IPPO, MAPPO and HAPPO trained under one fixed protocol,
evaluated on held-out seeds and reported as IQM with 95% bootstrap confidence intervals.
Scenario rules live in `gridhunt/`; this folder holds the protocol, the baselines and the
analysis scripts.

| File | Purpose |
|---|---|
| [`protocol.py`](protocol.py) | How each scenario is run for the benchmark (team reward, episode length, budget, variants) |
| [`marl/ppo.py`](marl/ppo.py) | IPPO, MAPPO and HAPPO in one PPO implementation |
| [`marl/vec_env.py`](marl/vec_env.py) | Batched environment copies with time-limit bootstrapping |
| [`train_marl.py`](train_marl.py) | Train one (scenario, algorithm, seed); resumable |
| [`reference_policies.py`](reference_policies.py) | Random and scripted (greedy BFS) reference teams |
| [`reference_scores.py`](reference_scores.py) | Random/scripted scores used to normalize results |
| [`interdependence.py`](interdependence.py) | Does each scenario need every agent? (scripted-team ablation) |
| [`aggregate.py`](aggregate.py) | Tables, learning curves and cross-scenario statistics |
| [`slurm/`](slurm/) | Job list generator and SLURM array script for BlueBEAR |

## Protocol

- **Reward:** `reward_mode="team"`, so every predator receives the team reward. The metric is the
  **team return**, the sum of `infos["__common__"]["team_reward"]` over an episode, with each capture
  counted once.
- **Episodes:** 200 steps (Swarm 300). Prey are not trainable. They are static, except in Specialists,
  where they flee the nearest predator.
- **Observations:** each scenario's default, except Raid and Swarm, which use `flat_coords`
  (entity coordinates). With their default one-hot grids (870 and 9,528 inputs per agent), every
  baseline stayed at or below the random team in the 5M-step pilot.
- **Evaluation:** actions sampled from the policy, on 32 fixed held-out seeds (`EVAL_SEED_OFFSET`)
  every 1% of the budget, and 100 episodes at the end. A run's score is its final evaluation mean.
  Sampling uses its own seeded generator per episode, so it is reproducible and does not affect
  training. Greedy (argmax) scores are recorded alongside (`greedy_return_mean`,
  `final_greedy_returns`, the `Greedy IQM` column). They are not the headline because a deterministic
  grid-world policy easily gets stuck: in the pilot, greedy Raid scores were up to 5× lower than sampled.
- **Statistics:** IQM over seeds with 95% stratified bootstrap CIs. Results are also normalized,
  `(score - random) / (scripted - random)`. The scripted team reads the true state, so it is a
  reference point, not an upper bound.
- **Budgets:** set per scenario in `protocol.py` from a 20M-step pilot, identical for all algorithms:
  1.5× the step where the slowest algorithm levels off, rounded up to a whole million and capped
  at 20M. Variants use their main scenario's budget, so each hypothesis test compares equally
  trained runs. ClassicStag gets 3M, Raid and LandAndSea 7M, Military and its variant 50M
  (Military was still rising steadily at 20M), everything else 20M. Pig chase and Swarm (HAPPO)
  were still improving at 20M and are reported as not converged, as is Military if its 20-seed
  curves are still rising at 50M.

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
8 of 8 with it. The floor binds only when a batch's advantage spread is tiny. Each run logs how often
it was active (`adv_floor_active` in `progress.csv`, `adv_floor_active_frac` in `final.json`).

### Scenario variants (hypothesis tests)

| Variant | Change | Question |
|---|---|---|
| `specialists_comm` | 4-token message channel (`communication=4`) | Can explicit communication replace the Bruiser's missing vision? |
| `specialists_full_vision` | Bruiser sees the whole board (same encoding) | How much is the information gap costing? |
| `toxic_stag_blind` | Nobody knows if the stag is toxic | Does predator_0's knowledge matter? |
| `military_full_vision` | Ground vehicle sees the whole board | Is the UAV's scouting needed? |

## Running

```bash
# One run (writes benchmark/results/marl/<scenario>/<algo>/seed_<n>/)
python benchmark/train_marl.py --scenario specialists --algo happo --seed 0 --budget 5000000

# Reference scores (random and scripted teams), once
python benchmark/reference_scores.py

# Tables and figures from all finished runs (benchmark/results/summary/)
python benchmark/aggregate.py
```

## BlueBEAR

The scripts are set up for the `baberc-human-agent-teaming` project. They assume the
repository lives at `/rds/projects/b/baberc-human-agent-teaming/Aju/gridhunt`. The conda env
goes in `.../Aju/${USER}_conda_envs/gridhunt`, logs go in `.../Aju/gridhunt/slurm_logs`, and every
command below runs from the repository root.

1. **Environment (once):**
   ```bash
   mkdir -p /rds/projects/b/baberc-human-agent-teaming/Aju/gridhunt/slurm_logs
   sbatch benchmark/slurm/setup_env.sh      # creates the env and runs the test suite
   ```
2. **Pilot:** 3 seeds × 20M steps for every scenario, in its own results folder, to find where the
   curves plateau. A first 5M-step pilot showed most curves still rising at 5M.
   ```bash
   python benchmark/slurm/make_jobs.py --scenarios all --seeds 0-2 --budget 20000000 \
       --results benchmark/results/pilot20m --out pilot_jobs.txt
   sbatch --time=1-12:00:00 --array=1-$(wc -l < pilot_jobs.txt)%100 benchmark/slurm/run_array.sh pilot_jobs.txt
   python benchmark/reference_scores.py   # once (the CSV is gitignored)
   python benchmark/aggregate.py --results benchmark/results/pilot20m --out benchmark/results/pilot20m_summary
   ```
   `budgets.csv` suggests 1.5× the step where each curve levels off, capped at the pilot's length.
   `plateaued=False` means a scenario was still improving at 20M; report it as not converged.
   Check the curve images too, not just the CSV. Run `make_jobs.py`, `reference_scores.py` and
   `aggregate.py` inside the env (`mamba activate` the env path above, after loading the same modules
   as `run_array.sh`), with `reference_scores.py` on a compute node (`srun ... --pty bash`).
3. Put the suggested budgets into `protocol.py`, commit, and run the full benchmark (20 seeds) into
   the default `benchmark/results/marl`. `make_jobs.py` refuses to reuse a finished run whose budget
   or training settings (`PPOConfig`) differ from the current ones, so leftover pilot runs or runs
   from an earlier version can't slip into the results.
   ```bash
   python benchmark/slurm/make_jobs.py --scenarios all --seeds 0-19 --out jobs.txt
   sbatch --time=1-12:00:00 --array=1-$(wc -l < jobs.txt)%100 benchmark/slurm/run_array.sh jobs.txt
   ```
4. If tasks time out, run `make_jobs.py` again: finished runs are skipped and unfinished ones resume
   from their checkpoints.

Each run records the git commit, library versions and SLURM job id in its `final.json`.

## Pretrained teams (gridhunt.zoo)

`export_zoo.py` turns finished runs into the numpy models that `gridhunt.zoo` downloads. For each
scenario and algorithm it exports the run with the median final score and the best run, and
checks every file against its PyTorch model: equal logits on observations from real episodes,
greedy actions that reproduce the greedy evaluation returns in `final.json` episode by episode,
and a sampled mean within 4 standard errors of the recorded score.

```bash
python benchmark/export_zoo.py --runs benchmark/results/marl --out zoo_export
sbatch benchmark/slurm/export_zoo.sh benchmark/results/marl zoo_export   # on BlueBEAR
```

It writes the `.npz` files, `registry.json` (copied to `src/gridhunt/zoo/registry.json`) and
`VERIFICATION.md`. The `.npz` files are published as assets of the `zoo-v1` GitHub release.

## Earlier RLlib pipeline

The earlier preprint trained its baselines with RLlib, using scripts that are not part of this
repository. Their "MAPPO" had no centralized critic and their LSTM settings were never applied, so
their numbers are superseded by this benchmark.
