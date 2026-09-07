"""
Training entry point: trains MADDPG on the multi-echelon supply chain env.

Usage:
    python -m training.train --episodes 500 --episode_length 104 --seed 0

Produces:
    results/training_curve.npz     -- per-episode cost / bullwhip / service logs
    results/maddpg_checkpoint.npz  -- final trained actor weights (all agents)
"""

from __future__ import annotations
import sys
import os
import time
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.supply_chain_env import SupplyChainEnv, DemandGenerator
from agents.maddpg import MADDPGTrainer


def save_checkpoint(trainer: MADDPGTrainer, path: str):
    save_dict = {}
    for i, agent in enumerate(trainer.agents):
        for k, v in agent.actor.get_flat_params().items():
            save_dict[f"agent{i}_actor_{k}"] = v
        # Persist each agent's observation-normalizer running statistics.
        # Without this, a freshly loaded trainer starts with a normalizer
        # at its default (mean=0, var=1) identity mapping, so raw
        # unnormalized observations (inventory ~0-100+) flow straight into
        # the actor's tanh layers and saturate them -- the policy then
        # behaves nothing like it did at the end of training. This was
        # discovered during evaluation: loaded checkpoints performed far
        # worse than classical baselines until this was fixed.
        save_dict[f"agent{i}_obsnorm_mean"] = trainer.obs_normalizers[i].mean
        save_dict[f"agent{i}_obsnorm_var"] = trainer.obs_normalizers[i].var
        save_dict[f"agent{i}_obsnorm_count"] = np.array([trainer.obs_normalizers[i].count])
    save_dict["state_norm_mean"] = trainer.state_normalizer.mean
    save_dict["state_norm_var"] = trainer.state_normalizer.var
    save_dict["state_norm_count"] = np.array([trainer.state_normalizer.count])
    np.savez(path, **save_dict)


def load_checkpoint(trainer: MADDPGTrainer, path: str):
    data = np.load(path)
    for i, agent in enumerate(trainer.agents):
        params = {}
        prefix = f"agent{i}_actor_"
        for key in data.files:
            if key.startswith(prefix):
                params[key[len(prefix):]] = data[key]
        agent.actor.set_flat_params(params)
        agent.actor_target.set_flat_params(params)

        mean_key, var_key, count_key = f"agent{i}_obsnorm_mean", f"agent{i}_obsnorm_var", f"agent{i}_obsnorm_count"
        if mean_key in data.files:
            trainer.obs_normalizers[i].mean = data[mean_key]
            trainer.obs_normalizers[i].var = data[var_key]
            trainer.obs_normalizers[i].count = float(data[count_key][0])

    if "state_norm_mean" in data.files:
        trainer.state_normalizer.mean = data["state_norm_mean"]
        trainer.state_normalizer.var = data["state_norm_var"]
        trainer.state_normalizer.count = float(data["state_norm_count"][0])


def train(
    episodes: int = 500,
    episode_length: int = 104,
    warmup_episodes: int = 20,
    updates_per_step: int = 1,
    demand_mode: str = "seasonal",
    seed: int = 0,
    log_every: int = 10,
    results_dir: str = "results",
    verbose: bool = True,
):
    os.makedirs(results_dir, exist_ok=True)

    demand_gen = DemandGenerator(mode=demand_mode, base_mean=20.0, base_std=4.0, seed=seed)
    env = SupplyChainEnv(episode_length=episode_length, demand_generator=demand_gen, seed=seed)

    obs_dims = [env.obs_dim(i) for i in range(env.n_agents)]
    global_state_dim = env.global_state_dim()
    max_actions = env.max_order

    trainer = MADDPGTrainer(
        obs_dims=obs_dims,
        global_state_dim=global_state_dim,
        max_actions=max_actions,
        agent_names=env.agent_names,
        hidden_sizes=(64, 64),
        actor_lr=1e-3,
        critic_lr=1e-3,
        gamma=0.95,
        tau=0.01,
        buffer_capacity=50_000,
        batch_size=128,
        bullwhip_penalty_coef=0.05,
        reward_scale=0.01,
        grad_clip=5.0,
        policy_delay=2,
        seed=seed,
    )

    log = {
        "episode": [],
        "total_cost": [],
        "avg_bullwhip": [],
        "avg_service_level": [],
        "gini": [],
        "critic_loss": [],
        "actor_loss": [],
    }

    start_time = time.time()
    best_rolling_cost = float("inf")
    for ep in range(episodes):
        obs = env.reset(seed=seed + ep)
        trainer.reset_noise()
        explore = ep < episodes  # always explore during training; eval script uses explore=False

        ep_critic_losses, ep_actor_losses = [], []

        for t in range(episode_length):
            global_state = env.global_state()  # state BEFORE taking the action
            actions_cont = trainer.act(obs, explore=(ep >= 0))
            obs_next, rewards, dones, infos = env.step(actions_cont)
            next_global_state = env.global_state()  # state AFTER the transition

            shaped_rewards = trainer.shape_rewards(rewards, actions_cont)

            trainer.store(
                obs=obs,
                actions=actions_cont,
                rewards=shaped_rewards,
                next_obs=obs_next,
                dones=dones,
                global_state=global_state,
                next_global_state=next_global_state,
            )

            obs = obs_next

            if ep >= warmup_episodes:
                for _ in range(updates_per_step):
                    metrics = trainer.update()
                    if metrics is not None:
                        ep_critic_losses.append(metrics["critic_loss"])
                        ep_actor_losses.append(metrics["actor_loss"])

        trainer.decay_noise(factor=0.99)

        total_cost = env.total_cost()
        bullwhip = env.bullwhip_ratios()
        service = env.service_levels()
        gini = env.cost_fairness_gini()

        log["episode"].append(ep)
        log["total_cost"].append(total_cost)
        log["avg_bullwhip"].append(float(np.nanmean(list(bullwhip.values()))))
        log["avg_service_level"].append(float(np.mean(list(service.values()))))
        log["gini"].append(gini)
        log["critic_loss"].append(float(np.mean(ep_critic_losses)) if ep_critic_losses else np.nan)
        log["actor_loss"].append(float(np.mean(ep_actor_losses)) if ep_actor_losses else np.nan)

        if verbose and (ep % log_every == 0 or ep == episodes - 1):
            elapsed = time.time() - start_time
            print(
                f"[ep {ep:4d}/{episodes}] cost={total_cost:9.1f}  "
                f"bullwhip={log['avg_bullwhip'][-1]:.2f}  "
                f"service={log['avg_service_level'][-1]:.3f}  "
                f"gini={gini:.3f}  "
                f"critic_loss={log['critic_loss'][-1]:.3f}  "
                f"elapsed={elapsed:.1f}s"
            )

        # Track the best-performing checkpoint seen so far (by a rolling
        # average of recent episode cost, not a single noisy episode) --
        # DDPG-family training on this environment has real episode-to-
        # episode variance even late in training, so saving only the
        # FINAL episode's weights risks saving a temporarily-bad policy.
        # This is standard RL practice (analogous to early stopping /
        # best-checkpoint selection).
        recent_window = 10
        if ep >= warmup_episodes + recent_window:
            rolling_cost = float(np.mean(log["total_cost"][-recent_window:]))
            if rolling_cost < best_rolling_cost:
                best_rolling_cost = rolling_cost
                save_checkpoint(trainer, os.path.join(results_dir, "maddpg_checkpoint_best.npz"))

    np.savez(os.path.join(results_dir, "training_curve.npz"), **{k: np.array(v) for k, v in log.items()})
    save_checkpoint(trainer, os.path.join(results_dir, "maddpg_checkpoint.npz"))

    if verbose:
        print(f"\nTraining complete in {time.time() - start_time:.1f}s.")
        print(f"Saved: {results_dir}/training_curve.npz")
        print(f"Saved: {results_dir}/maddpg_checkpoint.npz (final-episode weights)")
        print(f"Saved: {results_dir}/maddpg_checkpoint_best.npz (best rolling-10-episode-avg-cost weights, rolling_cost={best_rolling_cost:.1f})")

    return trainer, log


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--episode_length", type=int, default=104)
    parser.add_argument("--warmup_episodes", type=int, default=20)
    parser.add_argument("--updates_per_step", type=int, default=1)
    parser.add_argument("--demand_mode", type=str, default="seasonal", choices=["stationary", "seasonal", "shock"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--results_dir", type=str, default="results")
    args = parser.parse_args()

    train(
        episodes=args.episodes,
        episode_length=args.episode_length,
        warmup_episodes=args.warmup_episodes,
        updates_per_step=args.updates_per_step,
        demand_mode=args.demand_mode,
        seed=args.seed,
        results_dir=args.results_dir,
    )
