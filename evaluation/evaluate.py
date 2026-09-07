"""
Evaluation harness: compares the trained MADDPG policy against classical
OR baselines (base-stock, (s,S), naive pass-through) on the metrics
specified in the design doc:

  1. Total chain cost (vs. centralized-information-style base-stock optimum)
  2. Bullwhip ratio per echelon (variance amplification)
  3. Service level (fill rate) per echelon
  4. Cost-distribution fairness across echelons (Gini coefficient)

Also runs a stress test under non-stationary demand shocks and a
supply-capacity-disruption scenario, since the design doc explicitly
calls out that classical fixed policies should struggle to adapt there
while a learned policy plausibly can -- this is the key claim this
project needs evidence for, not just "MARL matches OR baselines under
easy stationary demand."

Usage:
    python -m evaluation.evaluate --checkpoint results/maddpg_checkpoint.npz
"""

from __future__ import annotations
import sys
import os
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.supply_chain_env import SupplyChainEnv, DemandGenerator
from agents.maddpg import MADDPGTrainer
from baselines.classical_policies import BaseStockPolicy, SSPolicy, NaivePassThrough, run_policy_episode
from training.train import load_checkpoint


N_EVAL_EPISODES = 20


def make_env(demand_mode: str, episode_length: int, seed: int) -> SupplyChainEnv:
    demand_gen = DemandGenerator(mode=demand_mode, base_mean=20.0, base_std=4.0, seed=seed)
    return SupplyChainEnv(episode_length=episode_length, demand_generator=demand_gen, seed=seed)


def run_maddpg_episode(trainer: MADDPGTrainer, env: SupplyChainEnv, seed: int) -> SupplyChainEnv:
    obs = env.reset(seed=seed)
    for t in range(env.episode_length):
        actions = trainer.act(obs, explore=False, update_stats=False)
        obs, rewards, dones, infos = env.step(actions)
    return env


def evaluate_policy_multi_seed(policy_fn, demand_mode: str, episode_length: int, n_episodes: int = N_EVAL_EPISODES):
    """
    policy_fn(seed) -> a populated SupplyChainEnv after a full episode.
    Runs across multiple seeds and aggregates metrics with mean +/- std.
    """
    all_costs, all_bullwhip, all_service, all_gini = [], [], [], []
    for ep in range(n_episodes):
        env = policy_fn(seed=1000 + ep)
        all_costs.append(env.total_cost())
        bw = env.bullwhip_ratios()
        all_bullwhip.append(np.nanmean(list(bw.values())))
        sv = env.service_levels()
        all_service.append(np.mean(list(sv.values())))
        all_gini.append(env.cost_fairness_gini())

    return {
        "cost_mean": float(np.mean(all_costs)),
        "cost_std": float(np.std(all_costs)),
        "bullwhip_mean": float(np.nanmean(all_bullwhip)),
        "bullwhip_std": float(np.nanstd(all_bullwhip)),
        "service_mean": float(np.mean(all_service)),
        "service_std": float(np.std(all_service)),
        "gini_mean": float(np.mean(all_gini)),
        "gini_std": float(np.std(all_gini)),
    }


def build_baselines(env_template: SupplyChainEnv):
    lead_times = [c.lead_time for c in env_template.configs]
    return {
        "BaseStock": BaseStockPolicy(env_template.n_agents, lead_times, demand_mean=20.0, demand_std=4.0),
        "(s,S)": SSPolicy(env_template.n_agents, lead_times, demand_mean=20.0, demand_std=4.0),
        "NaivePassThrough": NaivePassThrough(env_template.n_agents),
    }


def evaluate_all(trainer: MADDPGTrainer, episode_length: int, demand_modes: list, n_episodes: int = N_EVAL_EPISODES):
    results = {}  # results[demand_mode][policy_name] = metrics dict

    for demand_mode in demand_modes:
        results[demand_mode] = {}

        # MADDPG
        def maddpg_fn(seed, dm=demand_mode):
            env = make_env(dm, episode_length, seed)
            return run_maddpg_episode(trainer, env, seed)
        results[demand_mode]["MADDPG"] = evaluate_policy_multi_seed(maddpg_fn, demand_mode, episode_length, n_episodes)

        # Classical baselines (fresh policy objects re-instantiated per seed
        # is unnecessary since they're stateless in structure except
        # NaivePassThrough's rolling last-demand memory, which resets
        # correctly because we build a new policy object per episode call)
        template_env = make_env(demand_mode, episode_length, seed=0)
        for name, policy_builder in [
            ("BaseStock", lambda: build_baselines(template_env)["BaseStock"]),
            ("(s,S)", lambda: build_baselines(template_env)["(s,S)"]),
            ("NaivePassThrough", lambda: build_baselines(template_env)["NaivePassThrough"]),
        ]:
            def policy_fn(seed, dm=demand_mode, pb=policy_builder):
                env = make_env(dm, episode_length, seed)
                policy = pb()
                return run_policy_episode(env, policy, seed=seed)
            results[demand_mode][name] = evaluate_policy_multi_seed(policy_fn, demand_mode, episode_length, n_episodes)

    return results


def run_disruption_scenario(trainer: MADDPGTrainer, episode_length: int, n_episodes: int = N_EVAL_EPISODES):
    """
    Stress test: mid-episode, the Manufacturer's capacity is suddenly cut
    by 60% for an extended window (simulating a plant outage / logistics
    disruption), then restored. Classical fixed policies (calibrated to
    baseline capacity/demand) cannot adapt online; a learned policy that
    conditions on its own backlog/inventory state has a chance to react.

    Reports BOTH whole-episode cost (for consistency with the other
    tables) AND cost restricted to a window around the disruption itself
    (disruption start to +30 steps after it ends) -- the whole-episode
    number dilutes a 20-step disruption into a 104-step average and can
    look flat even when the disruption response differs meaningfully;
    the windowed number isolates the disruption-response signal.
    """
    disruption_start, disruption_duration = 30, 20
    window_end = disruption_start + disruption_duration + 30

    def apply_disruption(env: SupplyChainEnv, start=disruption_start, duration=disruption_duration, severity=0.4):
        original_capacity = env.configs[2].capacity  # Manufacturer
        def maybe_disrupt(t):
            if start <= t < start + duration:
                env.configs[2].capacity = original_capacity * severity
            else:
                env.configs[2].capacity = original_capacity
        return maybe_disrupt

    def windowed_cost(env: SupplyChainEnv) -> float:
        total = 0.0
        for i in range(env.n_agents):
            total += sum(env.history["cost"][i][disruption_start:window_end])
        return total

    def maddpg_disrupted(seed):
        env = make_env("stationary", episode_length, seed)
        disrupt = apply_disruption(env)
        obs = env.reset(seed=seed)
        for t in range(episode_length):
            disrupt(t)
            actions = trainer.act(obs, explore=False, update_stats=False)
            obs, rewards, dones, infos = env.step(actions)
        return env

    def baseline_disrupted(policy_builder, seed):
        env = make_env("stationary", episode_length, seed)
        disrupt = apply_disruption(env)
        policy = policy_builder()
        env.reset(seed=seed)
        for t in range(episode_length):
            disrupt(t)
            actions = policy.act(env)
            obs, rewards, dones, infos = env.step(actions)
            if hasattr(policy, "update_after_step"):
                policy.update_after_step(infos)
        return env

    template_env = make_env("stationary", episode_length, seed=0)
    policy_fns = {
        "MADDPG": maddpg_disrupted,
        "BaseStock": lambda seed: baseline_disrupted(lambda: build_baselines(template_env)["BaseStock"], seed),
        "(s,S)": lambda seed: baseline_disrupted(lambda: build_baselines(template_env)["(s,S)"], seed),
    }

    results = {}
    windowed_results = {}
    for name, fn in policy_fns.items():
        results[name] = evaluate_policy_multi_seed(fn, "stationary", episode_length, n_episodes)
        # Separately collect windowed costs across the same seeds
        windowed_costs = []
        for ep in range(n_episodes):
            env = fn(seed=1000 + ep)
            windowed_costs.append(windowed_cost(env))
        windowed_results[name] = {
            "windowed_cost_mean": float(np.mean(windowed_costs)),
            "windowed_cost_std": float(np.std(windowed_costs)),
        }
    return results, windowed_results


def print_comparison_table(results: dict, demand_mode: str):
    print(f"\n{'='*90}")
    print(f"  Demand mode: {demand_mode}   (mean +/- std over {N_EVAL_EPISODES} eval episodes)")
    print(f"{'='*90}")
    header = f"{'Policy':<18}{'Total Cost':>18}{'Bullwhip Ratio':>18}{'Service Level':>18}{'Gini (fairness)':>18}"
    print(header)
    print("-" * 90)
    for policy_name, m in results[demand_mode].items():
        print(
            f"{policy_name:<18}"
            f"{m['cost_mean']:>10.1f} +/-{m['cost_std']:>5.0f}"
            f"{m['bullwhip_mean']:>12.2f} +/-{m['bullwhip_std']:>4.2f}"
            f"{m['service_mean']:>12.3f} +/-{m['service_std']:>4.3f}"
            f"{m['gini_mean']:>12.3f} +/-{m['gini_std']:>4.3f}"
        )


def plot_comparison(results: dict, demand_modes: list, out_path: str):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    policies = list(next(iter(results.values())).keys())
    colors = plt.cm.tab10(np.linspace(0, 1, len(policies)))

    metrics = [
        ("cost_mean", "cost_std", "Total Chain Cost", axes[0, 0]),
        ("bullwhip_mean", "bullwhip_std", "Avg. Bullwhip Ratio (lower=better, 1.0=no amplification)", axes[0, 1]),
        ("service_mean", "service_std", "Avg. Service Level (fill rate)", axes[1, 0]),
        ("gini_mean", "gini_std", "Cost Fairness (Gini, lower=more even)", axes[1, 1]),
    ]

    x = np.arange(len(demand_modes))
    width = 0.8 / len(policies)

    for mean_key, std_key, title, ax in metrics:
        for pi, policy in enumerate(policies):
            means = [results[dm][policy][mean_key] for dm in demand_modes]
            stds = [results[dm][policy][std_key] for dm in demand_modes]
            ax.bar(x + pi * width, means, width, yerr=stds, label=policy, color=colors[pi], capsize=3)
        ax.set_xticks(x + width * (len(policies) - 1) / 2)
        ax.set_xticklabels(demand_modes)
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=130)
    plt.close()
    print(f"\nSaved comparison plot -> {out_path}")


def plot_training_curve(training_curve_path: str, out_path: str):
    data = np.load(training_curve_path)
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))

    axes[0, 0].plot(data["episode"], data["total_cost"], color="tab:blue")
    axes[0, 0].set_title("Total Chain Cost per Episode")
    axes[0, 0].set_xlabel("Episode")
    axes[0, 0].grid(alpha=0.3)

    axes[0, 1].plot(data["episode"], data["avg_bullwhip"], color="tab:red")
    axes[0, 1].axhline(1.0, color="gray", linestyle="--", linewidth=1, label="No amplification")
    axes[0, 1].set_title("Avg. Bullwhip Ratio per Episode")
    axes[0, 1].set_xlabel("Episode")
    axes[0, 1].legend(fontsize=8)
    axes[0, 1].grid(alpha=0.3)

    axes[1, 0].plot(data["episode"], data["avg_service_level"], color="tab:green")
    axes[1, 0].set_title("Avg. Service Level per Episode")
    axes[1, 0].set_xlabel("Episode")
    axes[1, 0].set_ylim(0, 1.05)
    axes[1, 0].grid(alpha=0.3)

    axes[1, 1].plot(data["episode"], data["critic_loss"], color="tab:purple", label="Critic loss")
    ax2 = axes[1, 1].twinx()
    ax2.plot(data["episode"], data["actor_loss"], color="tab:orange", alpha=0.6, label="Actor loss (proxy)")
    axes[1, 1].set_title("Critic / Actor Loss per Episode")
    axes[1, 1].set_xlabel("Episode")
    axes[1, 1].legend(loc="upper left", fontsize=8)
    ax2.legend(loc="upper right", fontsize=8)
    axes[1, 1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=130)
    plt.close()
    print(f"Saved training curve plot -> {out_path}")


def plot_sample_trajectory(trainer: MADDPGTrainer, episode_length: int, out_path: str, seed: int = 2024):
    """
    Plots orders placed at each echelon over one episode -- the classic
    "bullwhip visualization" showing whether order variance amplifies
    as you move upstream (Retailer -> Distributor -> Manufacturer -> Supplier).
    """
    env = make_env("seasonal", episode_length, seed)
    env = run_maddpg_episode(trainer, env, seed)

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    for i in range(env.n_agents):
        axes[0].plot(env.history["orders_placed"][i], label=env.agent_names[i])
    axes[0].plot(env.history["external_demand"], label="True consumer demand", color="black", linestyle="--", linewidth=1.5)
    axes[0].set_title("Orders Placed per Echelon (MADDPG policy) -- Bullwhip Visualization")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    for i in range(env.n_agents):
        axes[1].plot(env.history["inventory"][i], label=f"{env.agent_names[i]} inventory")
    axes[1].set_title("Inventory Level per Echelon")
    axes[1].set_xlabel("Timestep")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=130)
    plt.close()
    print(f"Saved sample trajectory plot -> {out_path}")


def main():
    global N_EVAL_EPISODES

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="results/maddpg_checkpoint.npz")
    parser.add_argument("--episode_length", type=int, default=104)
    parser.add_argument("--n_eval_episodes", type=int, default=N_EVAL_EPISODES)
    parser.add_argument("--results_dir", type=str, default="results")
    args = parser.parse_args()

    N_EVAL_EPISODES = args.n_eval_episodes

    template_env = make_env("seasonal", args.episode_length, seed=0)
    obs_dims = [template_env.obs_dim(i) for i in range(template_env.n_agents)]

    trainer = MADDPGTrainer(
        obs_dims=obs_dims,
        global_state_dim=template_env.global_state_dim(),
        max_actions=template_env.max_order,
        agent_names=template_env.agent_names,
        seed=0,
    )
    load_checkpoint(trainer, args.checkpoint)

    demand_modes = ["stationary", "seasonal", "shock"]
    print("\nEvaluating MADDPG vs. classical baselines across demand regimes...")
    results = evaluate_all(trainer, args.episode_length, demand_modes, n_episodes=args.n_eval_episodes)
    for dm in demand_modes:
        print_comparison_table(results, dm)

    print("\nRunning capacity-disruption stress test (Manufacturer capacity cut 60% for 20 steps)...")
    disruption_results, windowed_results = run_disruption_scenario(trainer, args.episode_length, n_episodes=args.n_eval_episodes)
    print(f"\n{'='*90}")
    print("  Capacity Disruption Stress Test -- Whole Episode")
    print(f"{'='*90}")
    header = f"{'Policy':<18}{'Total Cost':>18}{'Bullwhip Ratio':>18}{'Service Level':>18}"
    print(header)
    print("-" * 90)
    for policy_name, m in disruption_results.items():
        print(
            f"{policy_name:<18}"
            f"{m['cost_mean']:>10.1f} +/-{m['cost_std']:>5.0f}"
            f"{m['bullwhip_mean']:>12.2f} +/-{m['bullwhip_std']:>4.2f}"
            f"{m['service_mean']:>12.3f} +/-{m['service_std']:>4.3f}"
        )

    print(f"\n{'='*90}")
    print("  Capacity Disruption Stress Test -- Windowed Cost (steps 30-80, around the disruption)")
    print(f"{'='*90}")
    print(f"{'Policy':<18}{'Windowed Cost':>20}")
    print("-" * 90)
    for policy_name, m in windowed_results.items():
        print(f"{policy_name:<18}{m['windowed_cost_mean']:>14.1f} +/-{m['windowed_cost_std']:>5.0f}")

    os.makedirs(args.results_dir, exist_ok=True)
    plot_comparison(results, demand_modes, os.path.join(args.results_dir, "comparison_plot.png"))
    training_curve_path = os.path.join(args.results_dir, "training_curve.npz")
    if os.path.exists(training_curve_path):
        plot_training_curve(training_curve_path, os.path.join(args.results_dir, "training_curve_plot.png"))
    plot_sample_trajectory(trainer, args.episode_length, os.path.join(args.results_dir, "bullwhip_trajectory_plot.png"))

    np.savez(
        os.path.join(args.results_dir, "evaluation_results.npz"),
        results=results, disruption_results=disruption_results, windowed_results=windowed_results,
        allow_pickle=True,
    )
    print(f"\nSaved raw evaluation results -> {args.results_dir}/evaluation_results.npz")


if __name__ == "__main__":
    main()
