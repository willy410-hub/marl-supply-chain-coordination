"""
MADDPG: Multi-Agent Deep Deterministic Policy Gradient
========================================================

Implements the "Centralized Critic with Decentralized Actors" (CTDE)
architecture specified in the design doc:

  - Each agent i has its own ACTOR network: pi_i(o_i) -> a_i, using
    ONLY its local partial observation. This is what actually executes
    at deployment time -- fully decentralized.

  - Each agent i also has its own CRITIC network: Q_i(s_global, a_1..a_N)
    -> scalar, which during TRAINING sees the full joint state and all
    agents' actions. This lets agent i's critic learn how its local
    action affects whole-chain cost, solving the non-stationarity
    problem that plagues independent learners in multi-agent settings
    (from any single agent's view, the environment is non-stationary
    because other agents are also learning -- the centralized critic
    fixes this by conditioning on everyone's actions).

  - Target networks (actor_target, critic_target) + Polyak averaging
    for training stability, exactly as in DDPG.

  - Delayed policy updates (TD3-style): actor and target networks are
    updated only once every `policy_delay` critic updates. This damps
    the feedback loop where a still-noisy critic pushes the actor in a
    bad direction, which the critic then overfits to -- a major source
    of the training-curve variance seen during development on this
    project (episode-to-episode cost sometimes spiked 5-10x even after
    the policy had mostly converged). Delaying actor updates relative
    to the critic gives the critic more time to settle between policy
    updates that would otherwise chase a moving target.

  - Bullwhip-shaping term: an additional reward penalty proportional to
    the *increase* in order-variance relative to the immediately
    upstream neighbor, added directly to the training reward signal.
    This directly discourages bullwhip-inducing behavior rather than
    relying on it falling out of cost-minimization alone, per the
    design doc's reward architecture.

Reference: Lowe et al., "Multi-Agent Actor-Critic for Mixed
Cooperative-Competitive Environments" (2017).
"""

from __future__ import annotations
import numpy as np
from collections import deque
from typing import List, Dict, Tuple, Optional

from agents.networks import MLP, Adam, RunningNormalizer


# --------------------------------------------------------------------------- #
# Replay buffer (shared across agents -- stores joint transitions so the
# centralized critics can be trained with everyone's actions together)
# --------------------------------------------------------------------------- #

class ReplayBuffer:
    def __init__(self, capacity: int, n_agents: int, seed: Optional[int] = None):
        self.capacity = capacity
        self.n_agents = n_agents
        self.buffer = deque(maxlen=capacity)
        self.rng = np.random.default_rng(seed)

    def push(self, obs, actions, rewards, next_obs, dones, global_state, next_global_state):
        self.buffer.append((obs, actions, rewards, next_obs, dones, global_state, next_global_state))

    def sample(self, batch_size: int):
        idxs = self.rng.choice(len(self.buffer), size=min(batch_size, len(self.buffer)), replace=False)
        batch = [self.buffer[i] for i in idxs]
        return batch

    def __len__(self):
        return len(self.buffer)


# --------------------------------------------------------------------------- #
# Ornstein-Uhlenbeck exploration noise (standard for DDPG-family continuous
# control -- gives temporally-correlated exploration, better suited to
# inventory ordering than i.i.d. Gaussian noise since real order noise is
# autocorrelated too).
# --------------------------------------------------------------------------- #

class OUNoise:
    def __init__(self, dim: int, mu: float = 0.0, theta: float = 0.15,
                 sigma: float = 0.3, seed: Optional[int] = None):
        self.dim = dim
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.rng = np.random.default_rng(seed)
        self.reset()

    def reset(self):
        self.state = np.ones(self.dim) * self.mu

    def sample(self) -> np.ndarray:
        dx = self.theta * (self.mu - self.state) + self.sigma * self.rng.normal(size=self.dim)
        self.state += dx
        return self.state.copy()

    def decay_sigma(self, factor: float, min_sigma: float = 0.02):
        self.sigma = max(self.sigma * factor, min_sigma)


# --------------------------------------------------------------------------- #
# Single agent's actor + critic pair
# --------------------------------------------------------------------------- #

class MADDPGAgent:
    def __init__(
        self,
        agent_id: int,
        obs_dim: int,
        global_state_dim: int,
        n_agents: int,
        max_action: float,
        hidden_sizes: Tuple[int, int] = (64, 64),
        actor_lr: float = 1e-3,
        critic_lr: float = 2e-3,
        seed: Optional[int] = None,
    ):
        self.agent_id = agent_id
        self.obs_dim = obs_dim
        self.max_action = max_action
        self.n_agents = n_agents

        rng_seed = (seed or 0) + agent_id * 17

        # Actor: obs -> action in [-1, 1] (tanh), later rescaled to [0, max_action]
        self.actor = MLP([obs_dim, *hidden_sizes, 1], out_activation="tanh", seed=rng_seed)
        self.actor_target = self.actor.copy()
        self.actor_opt = Adam(self.actor.params, lr=actor_lr)

        # Critic: [global_state, all agents' actions] -> Q value (linear output)
        critic_in_dim = global_state_dim + n_agents
        self.critic = MLP([critic_in_dim, *hidden_sizes, 1], out_activation="linear", seed=rng_seed + 1000)
        self.critic_target = self.critic.copy()
        self.critic_opt = Adam(self.critic.params, lr=critic_lr)

        self.noise = OUNoise(dim=1, seed=rng_seed + 2000)

    def act(self, obs: np.ndarray, explore: bool = True) -> float:
        obs_b = obs.reshape(1, -1)
        raw_action = self.actor.forward(obs_b)[0, 0]  # in [-1, 1]
        action = (raw_action + 1.0) / 2.0 * self.max_action  # rescale to [0, max_action]
        if explore:
            action += self.noise.sample()[0] * self.max_action * 0.1
        return float(np.clip(action, 0.0, self.max_action))

    def act_batch_target(self, obs_batch: np.ndarray) -> np.ndarray:
        """Target-actor action for a batch of observations, rescaled to [0, max_action]."""
        raw = self.actor_target.forward(obs_batch)  # (batch, 1), in [-1, 1]
        return (raw + 1.0) / 2.0 * self.max_action


# --------------------------------------------------------------------------- #
# The MADDPG trainer orchestrating all agents
# --------------------------------------------------------------------------- #

class MADDPGTrainer:
    def __init__(
        self,
        obs_dims: List[int],
        global_state_dim: int,
        max_actions: List[float],
        agent_names: List[str],
        hidden_sizes: Tuple[int, int] = (64, 64),
        actor_lr: float = 1e-3,
        critic_lr: float = 2e-3,
        gamma: float = 0.95,
        tau: float = 0.01,
        buffer_capacity: int = 50_000,
        batch_size: int = 128,
        bullwhip_penalty_coef: float = 0.05,
        reward_scale: float = 0.01,
        grad_clip: float = 5.0,
        policy_delay: int = 2,
        seed: Optional[int] = None,
    ):
        self.n_agents = len(obs_dims)
        self.agent_names = agent_names
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.bullwhip_penalty_coef = bullwhip_penalty_coef
        # Raw per-step costs can be in the hundreds; over a long horizon
        # the undiscounted-sum-style Q targets would grow into the
        # thousands, which destabilizes a small critic MLP (exploding
        # TD targets -> divergence). Scaling rewards down keeps Q-values
        # in a numerically well-behaved range; effective returns are
        # unaffected up to this constant multiplier.
        self.reward_scale = reward_scale
        self.grad_clip = grad_clip
        # TD3-style delayed policy update: only update actor + targets
        # every `policy_delay` calls to update(). Critic updates happen
        # every call. See class docstring for why this matters here.
        self.policy_delay = policy_delay
        self._update_count = 0

        self.agents = [
            MADDPGAgent(
                agent_id=i,
                obs_dim=obs_dims[i],
                global_state_dim=global_state_dim,
                n_agents=self.n_agents,
                max_action=max_actions[i],
                hidden_sizes=hidden_sizes,
                actor_lr=actor_lr,
                critic_lr=critic_lr,
                seed=seed,
            )
            for i in range(self.n_agents)
        ]

        self.buffer = ReplayBuffer(buffer_capacity, self.n_agents, seed=seed)

        # Observation/state normalizers -- see RunningNormalizer docstring
        # for why this is necessary (prevents tanh-actor saturation from
        # raw, unnormalized inventory/backlog magnitudes).
        self.obs_normalizers = [RunningNormalizer(obs_dims[i]) for i in range(self.n_agents)]
        self.state_normalizer = RunningNormalizer(global_state_dim)

        # Rolling recent-order history per agent, used to compute the
        # bullwhip-amplification shaping penalty relative to the
        # immediately upstream neighbor.
        self._recent_orders = [deque(maxlen=10) for _ in range(self.n_agents)]

    # ------------------------------------------------------------------ #
    def act(self, obs: Dict[int, np.ndarray], explore: bool = True, update_stats: bool = True) -> Dict[int, float]:
        actions = {}
        for i in range(self.n_agents):
            if update_stats:
                self.obs_normalizers[i].update(obs[i])
            normed_obs = self.obs_normalizers[i].normalize(obs[i])
            actions[i] = self.agents[i].act(normed_obs, explore=explore)
        return actions

    def reset_noise(self):
        for agent in self.agents:
            agent.noise.reset()

    def decay_noise(self, factor: float = 0.995):
        for agent in self.agents:
            agent.noise.decay_sigma(factor)

    # ------------------------------------------------------------------ #
    def shape_rewards(self, raw_rewards: Dict[int, float], orders: Dict[int, float]) -> Dict[int, float]:
        """
        Adds a bullwhip-amplification shaping penalty: for each agent i > 0,
        compare the rolling variance of its own placed orders against the
        rolling variance of the orders placed by its immediate downstream
        neighbor (i-1), which is effectively the demand signal agent i is
        reacting to. If agent i's order variance is *higher* than its
        downstream neighbor's, it is amplifying variance upstream --
        penalize proportionally.
        """
        for i in range(self.n_agents):
            self._recent_orders[i].append(orders[i])

        shaped = dict(raw_rewards)
        for i in range(1, self.n_agents):
            if len(self._recent_orders[i]) >= 5 and len(self._recent_orders[i - 1]) >= 5:
                var_self = np.var(list(self._recent_orders[i]))
                var_upstream_signal = np.var(list(self._recent_orders[i - 1]))
                amplification = max(0.0, var_self - var_upstream_signal)
                shaped[i] -= self.bullwhip_penalty_coef * amplification
        return shaped

    # ------------------------------------------------------------------ #
    def store(self, obs, actions, rewards, next_obs, dones, global_state, next_global_state):
        scaled_rewards = {i: r * self.reward_scale for i, r in rewards.items()}
        self.state_normalizer.update(global_state)
        # Raw (unnormalized) obs/state are stored in the buffer; normalization
        # is applied at sample-time in update() using the normalizer's
        # *current* running statistics, matching standard practice (e.g.
        # OpenAI's running-mean-std obs normalization in PPO/DDPG baselines).
        self.buffer.push(obs, actions, scaled_rewards, next_obs, dones, global_state, next_global_state)

    # ------------------------------------------------------------------ #
    def _build_critic_input(self, global_states: np.ndarray, actions_all: np.ndarray) -> np.ndarray:
        """global_states: (B, state_dim), actions_all: (B, n_agents) -> (B, state_dim+n_agents)"""
        return np.concatenate([global_states, actions_all], axis=1)

    def update(self) -> Optional[Dict[str, float]]:
        if len(self.buffer) < self.batch_size:
            return None

        batch = self.buffer.sample(self.batch_size)
        B = len(batch)

        obs_batch_raw = [np.stack([b[0][i] for b in batch]) for i in range(self.n_agents)]
        next_obs_batch_raw = [np.stack([b[3][i] for b in batch]) for i in range(self.n_agents)]
        actions_batch = np.stack([[b[1][i] for i in range(self.n_agents)] for b in batch])  # (B, n_agents)
        rewards_batch = np.stack([[b[2][i] for i in range(self.n_agents)] for b in batch])  # (B, n_agents)
        dones_batch = np.stack([[float(b[4][i]) for i in range(self.n_agents)] for b in batch])  # (B, n_agents)
        global_state_batch_raw = np.stack([b[5] for b in batch])       # (B, state_dim)
        next_global_state_batch_raw = np.stack([b[6] for b in batch])  # (B, state_dim)

        # Apply current running-stat normalization (no stat update here --
        # stats are only updated once per real env step in act()/store()).
        obs_batch = [self.obs_normalizers[i].normalize(obs_batch_raw[i]) for i in range(self.n_agents)]
        next_obs_batch = [self.obs_normalizers[i].normalize(next_obs_batch_raw[i]) for i in range(self.n_agents)]
        global_state_batch = self.state_normalizer.normalize(global_state_batch_raw)
        next_global_state_batch = self.state_normalizer.normalize(next_global_state_batch_raw)

        # Target actions for all agents at next state, using each agent's
        # target actor and its own next-obs slice.
        next_actions_all = np.stack(
            [self.agents[i].act_batch_target(next_obs_batch[i])[:, 0] for i in range(self.n_agents)],
            axis=1,
        )  # (B, n_agents)

        metrics = {"critic_loss": 0.0, "actor_loss": 0.0}
        self._update_count += 1
        do_policy_update = (self._update_count % self.policy_delay == 0)

        for i, agent in enumerate(self.agents):
            # -------------------- Critic update (every call) -------------------- #
            critic_next_input = self._build_critic_input(next_global_state_batch, next_actions_all)
            q_next = agent.critic_target.forward(critic_next_input)[:, 0]  # (B,)
            td_target = rewards_batch[:, i] + self.gamma * (1 - dones_batch[:, i]) * q_next

            critic_input = self._build_critic_input(global_state_batch, actions_batch)
            cache = {}
            q_pred = agent.critic.forward(critic_input, cache=cache)[:, 0]

            td_error = q_pred - td_target  # (B,)
            critic_loss = float(np.mean(td_error ** 2))
            d_out = (2.0 * td_error / B).reshape(-1, 1)
            grads = agent.critic.backward(cache, d_out)
            agent.critic_opt.step(agent.critic.params, grads, grad_clip=self.grad_clip)
            metrics["critic_loss"] += critic_loss / self.n_agents

            if not do_policy_update:
                continue

            # -------------------- Actor update (delayed, every policy_delay calls) -------------------- #
            # Deterministic policy gradient: maximize Q_i(s, a_1,...,pi_i(o_i),...,a_N)
            # w.r.t. agent i's own actor params. Other agents' actions are
            # held fixed at the batch's actual (behavioral) actions -- this
            # is the standard MADDPG approximation.
            actor_cache = {}
            raw_action_i = agent.actor.forward(obs_batch[i], cache=actor_cache)  # (B, 1), in [-1, 1]
            action_i = (raw_action_i + 1.0) / 2.0 * agent.max_action  # (B, 1)

            actions_for_critic = actions_batch.copy()
            actions_for_critic[:, i] = action_i[:, 0]
            critic_input_pg = self._build_critic_input(global_state_batch, actions_for_critic)

            critic_cache_pg = {}
            q_for_pg = agent.critic.forward(critic_input_pg, cache=critic_cache_pg)  # (B, 1)

            # We want to ASCEND q_for_pg w.r.t. action_i -> minimize -mean(Q)
            d_q = -np.ones_like(q_for_pg) / B
            critic_grads_pg = agent.critic.backward(critic_cache_pg, d_q)
            # Gradient of Q w.r.t. its input, sliced to the action_i column
            # (which sits right after the global_state block).
            d_input = critic_grads_pg["dX"]  # (B, state_dim+n_agents)
            state_dim = global_state_batch.shape[1]
            d_action_i = d_input[:, state_dim + i: state_dim + i + 1]  # (B, 1)

            # Chain rule through the rescale (raw in [-1,1] -> [0,max_action])
            d_raw_action_i = d_action_i * (agent.max_action / 2.0)

            actor_grads = agent.actor.backward(actor_cache, d_raw_action_i)
            actor_loss = float(-np.mean(q_for_pg))
            agent.actor_opt.step(agent.actor.params, actor_grads, grad_clip=self.grad_clip)
            metrics["actor_loss"] += actor_loss / self.n_agents

            # -------------------- Target network soft update (delayed too) -------------------- #
            agent.actor_target.polyak_update(agent.actor, self.tau)
            agent.critic_target.polyak_update(agent.critic, self.tau)

        return metrics
