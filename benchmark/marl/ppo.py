"""
IPPO, MAPPO and HAPPO sharing one PPO implementation.

            actor                                    critic
  IPPO      one per agent                            one per agent, on its own observation
  MAPPO     one shared by all agents                 one, on the global state (env.state())
            (input: observation + agent one-hot)
  HAPPO     one per agent                            one, on the global state

HAPPO (Kuba et al., 2022) updates the actors one at a time in a random order; each
agent's advantage is reweighted by the probability ratio of the agents updated
before it in that iteration. Everything else (rollouts, GAE, value normalization,
losses, hyperparameters) is identical across the three, so differences in results
come from the algorithms alone.

GridHunt episodes end only at the time limit, so value targets bootstrap from the
value of the final state instead of treating the time limit as terminal.
"""

import dataclasses
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

ALGORITHMS = ("ippo", "mappo", "happo")


@dataclasses.dataclass
class PPOConfig:
    num_envs: int = 16
    rollout_steps: int = 128  # samples per iteration = num_envs * rollout_steps
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip: float = 0.2
    epochs: int = 5
    minibatches: int = 4
    lr: float = 5e-4
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    # Advantages are standardized per batch, but never divided by less than this fraction
    # of the critic's running return std; 0 disables the floor (plain per-batch scaling).
    # Without it, a solved deterministic task (Pig chase) has near-zero advantages whose
    # noise per-batch scaling blows up to unit size, and the resulting update breaks the policy.
    adv_std_floor: float = 0.05
    hidden: tuple = (128, 128)


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

def layer_init(layer, gain=np.sqrt(2)):
    nn.init.orthogonal_(layer.weight, gain)
    nn.init.zeros_(layer.bias)
    return layer


def mlp(in_dim, out_dim, hidden, out_gain):
    layers, dim = [], in_dim
    for size in hidden:
        layers += [layer_init(nn.Linear(dim, size)), nn.Tanh()]
        dim = size
    layers.append(layer_init(nn.Linear(dim, out_dim), gain=out_gain))
    return nn.Sequential(*layers)


class ValueNorm:
    """Running mean/std of value targets; critics predict normalized values."""

    def __init__(self, epsilon=1e-5):
        self.mean, self.var, self.count, self.epsilon = 0.0, 1.0, epsilon, epsilon

    def update(self, x):
        x = np.asarray(x, dtype=np.float64)
        batch_mean, batch_var, batch_count = x.mean(), x.var(), x.size
        delta = batch_mean - self.mean
        total = self.count + batch_count
        self.mean += delta * batch_count / total
        self.var = (self.var * self.count + batch_var * batch_count
                    + delta ** 2 * self.count * batch_count / total) / total
        self.count = total

    def normalize(self, x):
        return (x - self.mean) / np.sqrt(self.var + self.epsilon)

    def denormalize(self, x):
        return x * np.sqrt(self.var + self.epsilon) + self.mean

    def state_dict(self):
        return {"mean": self.mean, "var": self.var, "count": self.count}

    def load_state_dict(self, state):
        self.mean, self.var, self.count = state["mean"], state["var"], state["count"]


def compute_gae(rewards, values, next_values, dones, gamma, gae_lambda):
    """Generalized advantage estimation over a [T, E] rollout.

    next_values[t] is the value of the state reached after step t; at a time limit
    (dones[t]) it is the value of that episode's final state, and the advantage
    trace is cut so it does not leak into the next episode.
    """
    advantages = np.zeros_like(rewards)
    last = np.zeros(rewards.shape[1:], dtype=rewards.dtype)
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * next_values[t] - values[t]
        last = delta + gamma * gae_lambda * (1.0 - dones[t]) * last
        advantages[t] = last
    return advantages, advantages + values


def _normalize(x, min_std=0.0):
    return (x - x.mean()) / (max(x.std(), min_std) + 1e-8)


# ---------------------------------------------------------------------------
# The three algorithms
# ---------------------------------------------------------------------------

class MultiAgentPPO:
    def __init__(self, algo, agents, obs_dims, action_dims, state_dim, config, seed):
        if algo not in ALGORITHMS:
            raise ValueError(f"algo must be one of {ALGORITHMS}")
        self.algo, self.agents, self.config = algo, list(agents), config
        self.shared_actor = algo == "mappo"
        self.central_critic = algo in ("mappo", "happo")
        self.rng = np.random.default_rng(seed)
        hidden = config.hidden

        if self.shared_actor:
            if len(set(obs_dims.values())) != 1 or len(set(action_dims.values())) != 1:
                raise ValueError("MAPPO's shared actor needs equal observation and action sizes")
            first = self.agents[0]
            actor = mlp(obs_dims[first] + len(self.agents), action_dims[first], hidden, 0.01)
            self.actors = {agent: actor for agent in self.agents}
        else:
            self.actors = {a: mlp(obs_dims[a], action_dims[a], hidden, 0.01) for a in self.agents}
        if self.central_critic:
            self.critics = {"central": mlp(state_dim, 1, hidden, 1.0)}
        else:
            self.critics = {a: mlp(obs_dims[a], 1, hidden, 1.0) for a in self.agents}
        self.value_norms = {key: ValueNorm() for key in self.critics}

        self.actor_keys = ["shared"] if self.shared_actor else self.agents
        actor_modules = {"shared": actor} if self.shared_actor else self.actors
        self.actor_optimizers = {
            k: torch.optim.Adam(actor_modules[k].parameters(), lr=config.lr, eps=1e-5)
            for k in self.actor_keys
        }
        self.critic_optimizers = {
            k: torch.optim.Adam(c.parameters(), lr=config.lr, eps=1e-5) for k, c in self.critics.items()
        }
        self._agent_onehot = {
            agent: np.eye(len(self.agents), dtype=np.float32)[i] for i, agent in enumerate(self.agents)
        }

    # -- acting -------------------------------------------------------------

    def _actor_input(self, agent, obs):
        if not self.shared_actor:
            return obs
        onehot = np.broadcast_to(self._agent_onehot[agent], obs.shape[:-1] + (len(self.agents),))
        return np.concatenate([obs, onehot], axis=-1)

    def _distribution(self, agent, obs):
        logits = self.actors[agent](torch.as_tensor(self._actor_input(agent, obs)))
        return Categorical(logits=logits)

    @torch.no_grad()
    def act(self, obs, greedy=False, generator=None):
        """obs: {agent: [E, d]} -> actions {agent: [E]}, log-probs {agent: [E]}.

        generator: optional torch.Generator for sampling, so evaluation can sample
        reproducibly without touching the global torch RNG used by training.
        """
        actions, logps = {}, {}
        for agent in self.agents:
            dist = self._distribution(agent, obs[agent])
            if greedy:
                action = dist.probs.argmax(-1)
            elif generator is not None:
                action = torch.multinomial(dist.probs, 1, generator=generator).squeeze(-1)
            else:
                action = dist.sample()
            actions[agent] = action.numpy()
            logps[agent] = dist.log_prob(action).numpy()
        return actions, logps

    @torch.no_grad()
    def values(self, obs, state):
        """Denormalized values: {critic key: [E]}."""
        out = {}
        for key, critic in self.critics.items():
            x = state if key == "central" else obs[key]
            prediction = critic(torch.as_tensor(x)).squeeze(-1).numpy()
            out[key] = self.value_norms[key].denormalize(prediction)
        return out

    # -- learning -------------------------------------------------------------

    def _minibatches(self, size):
        order = self.rng.permutation(size)
        return np.array_split(order, self.config.minibatches)

    def _policy_loss(self, agent, obs, actions, old_logp, advantages):
        dist = self._distribution(agent, obs)
        logp = dist.log_prob(torch.as_tensor(actions))
        ratio = torch.exp(logp - torch.as_tensor(old_logp))
        adv = torch.as_tensor(advantages)
        clipped = torch.clamp(ratio, 1 - self.config.clip, 1 + self.config.clip)
        policy_loss = -torch.min(ratio * adv, clipped * adv).mean()
        entropy = dist.entropy().mean()
        with torch.no_grad():
            approx_kl = ((ratio - 1) - torch.log(ratio)).mean().item()
            clip_frac = ((ratio - 1).abs() > self.config.clip).float().mean().item()
        return policy_loss - self.config.entropy_coef * entropy, {
            "policy_loss": policy_loss.item(), "entropy": entropy.item(),
            "approx_kl": approx_kl, "clip_frac": clip_frac,
        }

    def _step(self, optimizer, loss, parameters):
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(parameters, self.config.max_grad_norm)
        optimizer.step()

    def _train_actor(self, key, agents, obs, actions, old_logp, advantages, stats):
        """PPO epochs for one actor; agents lists who shares it (all, for MAPPO)."""
        x = np.concatenate([obs[a] for a in agents])
        act = np.concatenate([actions[a] for a in agents])
        logp = np.concatenate([old_logp[a] for a in agents])
        adv = np.concatenate([advantages[a] for a in agents])
        module = self.actors[agents[0]]
        for _ in range(self.config.epochs):
            for idx in self._minibatches(len(x)):
                # Each sample's agent determines the one-hot (MAPPO); recover it by block
                loss, info = self._loss_for_blocks(agents, x, act, logp, adv, idx)
                self._step(self.actor_optimizers[key], loss, module.parameters())
                for k, v in info.items():
                    stats[k].append(v)

    def _loss_for_blocks(self, agents, x, act, logp, adv, idx):
        block = len(x) // len(agents)
        losses, infos = [], []
        for i, agent in enumerate(agents):
            sel = idx[(idx >= i * block) & (idx < (i + 1) * block)]
            if len(sel) == 0:
                continue
            loss, info = self._policy_loss(agent, x[sel], act[sel], logp[sel], adv[sel])
            losses.append(loss * len(sel))
            infos.append((info, len(sel)))
        total = sum(n for _, n in infos)
        merged = {k: sum(info[k] * n for info, n in infos) / total for k in infos[0][0]}
        return sum(losses) / total, merged

    def _train_critics(self, obs, state, returns, stats):
        for key, critic in self.critics.items():
            x = state if key == "central" else obs[key]
            norm = self.value_norms[key]
            norm.update(returns[key])
            target = norm.normalize(returns[key]).astype(np.float32)
            for _ in range(self.config.epochs):
                for idx in self._minibatches(len(x)):
                    prediction = critic(torch.as_tensor(x[idx])).squeeze(-1)
                    loss = ((prediction - torch.as_tensor(target[idx])) ** 2).mean()
                    self._step(self.critic_optimizers[key], loss, critic.parameters())
                    stats["value_loss"].append(loss.item())

    def _standardize(self, adv, key, stats):
        floor = self.config.adv_std_floor * float(np.sqrt(self.value_norms[key].var))
        stats["adv_floor_active"].append(float(adv.std() < floor))
        return _normalize(adv, floor)

    def update(self, batch):
        """batch: flattened rollout (see collect_rollout). Returns mean training stats."""
        obs, state, actions, old_logp = batch["obs"], batch["state"], batch["actions"], batch["logps"]
        stats = {k: [] for k in ("policy_loss", "entropy", "approx_kl", "clip_frac", "value_loss", "adv_floor_active")}
        self._train_critics(obs, state, batch["returns"], stats)

        if self.central_critic:
            shared_adv = self._standardize(batch["advantages"]["central"], "central", stats)
            advantages = {agent: shared_adv for agent in self.agents}
        else:
            advantages = {agent: self._standardize(batch["advantages"][agent], agent, stats)
                          for agent in self.agents}

        if self.algo == "mappo":
            self._train_actor("shared", self.agents, obs, actions, old_logp, advantages, stats)
        elif self.algo == "ippo":
            for agent in self.agents:
                self._train_actor(agent, [agent], obs, actions, old_logp, advantages, stats)
        else:  # HAPPO: sequential updates with the multi-agent advantage factor
            factor = np.ones_like(shared_adv)
            for agent in self.rng.permutation(self.agents):
                weighted = {agent: (shared_adv * factor).astype(np.float32)}
                self._train_actor(agent, [agent], obs, actions, old_logp, weighted, stats)
                with torch.no_grad():
                    new_logp = self._distribution(agent, obs[agent]).log_prob(
                        torch.as_tensor(actions[agent])
                    ).numpy()
                factor = factor * np.exp(new_logp - old_logp[agent])
        return {k: float(np.mean(v)) for k, v in stats.items() if v}

    # -- checkpoints ------------------------------------------------------------

    def state_dict(self):
        actor_modules = {"shared": self.actors[self.agents[0]]} if self.shared_actor else self.actors
        return {
            "algo": self.algo,
            "actors": {k: actor_modules[k].state_dict() for k in self.actor_keys},
            "critics": {k: c.state_dict() for k, c in self.critics.items()},
            "actor_optimizers": {k: o.state_dict() for k, o in self.actor_optimizers.items()},
            "critic_optimizers": {k: o.state_dict() for k, o in self.critic_optimizers.items()},
            "value_norms": {k: n.state_dict() for k, n in self.value_norms.items()},
            "rng": self.rng.bit_generator.state,
        }

    def load_state_dict(self, state):
        actor_modules = {"shared": self.actors[self.agents[0]]} if self.shared_actor else self.actors
        for k in self.actor_keys:
            actor_modules[k].load_state_dict(state["actors"][k])
            self.actor_optimizers[k].load_state_dict(state["actor_optimizers"][k])
        for k, critic in self.critics.items():
            critic.load_state_dict(state["critics"][k])
            self.critic_optimizers[k].load_state_dict(state["critic_optimizers"][k])
            self.value_norms[k].load_state_dict(state["value_norms"][k])
        self.rng.bit_generator.state = state["rng"]


# ---------------------------------------------------------------------------
# Rollouts and evaluation
# ---------------------------------------------------------------------------

def collect_rollout(learner, envs, obs, state):
    """Collect rollout_steps steps from every env copy and compute advantages.

    Returns (batch, next_obs, next_state, finished_episode_returns).
    """
    cfg = learner.config
    T, E, agents = cfg.rollout_steps, envs.num_envs, envs.agents
    keys = list(learner.critics)
    buf = {
        "obs": {a: [] for a in agents}, "state": [], "actions": {a: [] for a in agents},
        "logps": {a: [] for a in agents}, "values": {k: [] for k in keys},
        "rewards": {k: [] for k in keys}, "dones": [], "bootstrap": {k: [] for k in keys},
    }
    finished = []
    for _ in range(T):
        actions, logps = learner.act(obs)
        values = learner.values(obs, state)
        out = envs.step(actions)
        for a in agents:
            buf["obs"][a].append(obs[a])
            buf["actions"][a].append(actions[a])
            buf["logps"][a].append(logps[a])
        buf["state"].append(state)
        buf["dones"].append(out["dones"].astype(np.float32))
        if out["dones"].any():  # only needed where an episode just hit the time limit
            final_values = learner.values(out["final_obs"], out["final_state"])
        else:
            final_values = {k: np.zeros(E, dtype=np.float32) for k in keys}
        for k in keys:
            buf["values"][k].append(values[k])
            buf["rewards"][k].append(out["team_rewards"] if k == "central" else out["rewards"][k])
            buf["bootstrap"][k].append(final_values[k])
        finished.extend(out["episode_returns"])
        obs, state = out["obs"], out["state"]

    last_values = learner.values(obs, state)
    dones = np.stack(buf["dones"])
    batch = {"advantages": {}, "returns": {}}
    for k in keys:
        values = np.stack(buf["values"][k]).astype(np.float32)
        next_values = np.concatenate([values[1:], last_values[k][None]], axis=0)
        next_values = np.where(dones > 0, np.stack(buf["bootstrap"][k]), next_values).astype(np.float32)
        rewards = np.stack(buf["rewards"][k]).astype(np.float32)
        adv, ret = compute_gae(rewards, values, next_values, dones, cfg.gamma, cfg.gae_lambda)
        batch["advantages"][k] = adv.reshape(T * E)
        batch["returns"][k] = ret.reshape(T * E)
    batch["obs"] = {a: np.stack(buf["obs"][a]).reshape(T * E, -1) for a in agents}
    batch["actions"] = {a: np.stack(buf["actions"][a]).reshape(T * E) for a in agents}
    batch["logps"] = {a: np.stack(buf["logps"][a]).reshape(T * E).astype(np.float32) for a in agents}
    batch["state"] = np.stack(buf["state"]).reshape(T * E, -1)
    return batch, obs, state, finished


def evaluate(learner, make_env, episodes, seed_offset, greedy=False):
    """Evaluate on fixed seeds. Returns per-episode team returns and capture counts.

    By default actions are sampled from the policy (the benchmark's headline metric), using
    a generator seeded per episode so results are reproducible and training's RNG is left
    untouched. greedy=True takes the most likely action instead.
    """
    env = make_env()
    returns, capture_counts = [], []
    for episode in range(episodes):
        obs, _ = env.reset(seed=seed_offset + episode)
        generator = torch.Generator().manual_seed(seed_offset + episode)
        total, counts = 0.0, Counter()
        while env.agents:
            batched = {a: obs[a][None].astype(np.float32) for a in learner.agents}
            actions, _ = learner.act(batched, greedy=greedy, generator=generator)
            obs, _, _, _, infos = env.step({a: int(actions[a][0]) for a in learner.agents})
            total += infos["__common__"]["team_reward"]
            for capture in infos["__common__"]["captures"]:
                counts[capture["type"] + ("_toxic" if capture["toxic"] else "")] += 1
        returns.append(total)
        capture_counts.append(dict(counts))
    return returns, capture_counts


def load_learner(model_path):
    """Rebuild a trained MultiAgentPPO from a model.pt written by train_marl.py."""
    saved = torch.load(model_path, weights_only=False)
    config_dict = dict(saved["config"])
    config_dict["hidden"] = tuple(config_dict["hidden"])
    learner = MultiAgentPPO(
        saved["learner"]["algo"], saved["agents"], saved["obs_dims"], saved["action_dims"],
        saved["state_dim"], PPOConfig(**config_dict), seed=0,
    )
    learner.load_state_dict(saved["learner"])
    return learner
