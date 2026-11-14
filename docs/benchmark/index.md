# Benchmark

IPPO, MAPPO and HAPPO were trained on every scenario under one fixed protocol, 20 seeds each, and
evaluated on held-out seeds. Results are reported as the interquartile mean (IQM) over seeds with
95% stratified bootstrap confidence intervals. The code, the protocol and instructions to reproduce
everything are in the repository's
[`benchmark/`](https://github.com/ajuanijustus/gridhunt/tree/main/benchmark) folder, which is not
part of the pip package.

## Protocol

- **Reward:** team reward (`reward_mode="team"`). The score is the **team return**: the sum of
  `infos["__common__"]["team_reward"]` over an episode, each capture counted once.
- **Episodes:** 200 steps (Swarm 300); prey are static except in Specialists, where they flee.
- **Observations:** each scenario's default, except Raid and Swarm, which use `flat_coords`.
- **Evaluation:** actions sampled from the policy on 32 fixed held-out seeds every 1% of training,
  and 100 episodes at the end; a run's score is its final evaluation mean. Greedy (argmax) scores
  are recorded alongside.
- **Budgets:** per scenario, the same for all algorithms, set from a 20M-step pilot: ClassicStag 3M,
  Raid and LandAndSea 7M, Military 50M, all others 20M.
- **Algorithms:** one PyTorch implementation of PPO with a 128×128 tanh MLP, GAE, value
  normalization and a floor on advantage standardization:

| Algorithm | Actor | Critic |
|---|---|---|
| IPPO | one per agent | one per agent, on its own observation |
| MAPPO | one shared (observation + agent one-hot) | one, on the global state |
| HAPPO | one per agent, updated sequentially in random order | one, on the global state |

Scores are also normalized as `(score - random) / (scripted - random)`, using a random team and the
scripted `GreedyPolicy` team. The scripted team reads the true state, including hidden information,
so it is a reference point, not an upper bound: trained teams beat it in some scenarios
(normalized scores above 1) and fall far short in others.

## Results

Normalized IQM over the eight main scenarios:

--8<-- "benchmark/results/summary/aggregate.md"

![Normalized IQM per algorithm](aggregate.png)

Per scenario (team return):

--8<-- "benchmark/results/summary/final_scores.md"

The `_comm`, `_full_vision` and `_blind` rows are hypothesis tests that change one thing about a
scenario: a message channel for Specialists, full vision for the agent that lacks it, and no
knowledge of stag toxicity in Toxic stag.

## Learning curves

IQM of the evaluation score over training, with 95% bootstrap intervals.

| | |
|---|---|
| ![Classic stag](curves/classic_stag.png) | ![Toxic stag](curves/toxic_stag.png) |
| ![Specialists](curves/specialists.png) | ![Military](curves/military.png) |
| ![Raid](curves/raid.png) | ![Pig chase](curves/pig_chase.png) |
| ![Land and sea](curves/land_and_sea.png) | ![Swarm](curves/swarm.png) |

## Reproducing

The repository's
[`benchmark/README.md`](https://github.com/ajuanijustus/gridhunt/blob/main/benchmark/README.md)
explains how to train a single run, run the full sweep on a SLURM cluster, and rebuild every table
from the archived runs.
