"""
Sanity / regression test suite for the supply-chain MARL project.

These are not exhaustive unit tests -- they are the specific checks that
were actually used during development to catch the bugs documented in
the README's "Debugging Journal" (gradient correctness, environment
stability, normalizer persistence). Running this file end-to-end before
trusting any new results is good practice for exactly the failure modes
this project already hit once.

Usage:
    python -m tests.test_project
    (or: pytest tests/test_project.py -v, if pytest is available)
"""

from __future__ import annotations
import sys
import os
import tempfile
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.supply_chain_env import SupplyChainEnv, DemandGenerator
from agents.networks import MLP, Adam, RunningNormalizer
from agents.maddpg import MADDPGTrainer, OUNoise, ReplayBuffer
from baselines.classical_policies import BaseStockPolicy, SSPolicy, NaivePassThrough, run_policy_episode
from training.train import save_checkpoint, load_checkpoint


def _assert(cond, msg):
    if not cond:
        raise AssertionError(msg)
    print(f"  OK: {msg}")


# --------------------------------------------------------------------------- #
def test_environment_basic():
    print("\n[test_environment_basic]")
    env = SupplyChainEnv(episode_length=30, seed=0)
    obs = env.reset(seed=0)
    _assert(env.n_agents == 4, "environment has 4 agents")
    _assert(all(i in obs for i in range(env.n_agents)), "reset returns obs for every agent")

    for i in range(env.n_agents):
        _assert(obs[i].shape[0] == env.obs_dim(i), f"agent {i} obs shape matches obs_dim()")

    rng = np.random.default_rng(1)
    for _ in range(30):
        actions = {i: float(rng.uniform(5, 30)) for i in range(env.n_agents)}
        obs, rewards, dones, infos = env.step(actions)
        _assert(all(np.isfinite(v) for v in rewards.values()), "no NaN/inf rewards")
        _assert(all(np.isfinite(env.inventory)), "no NaN/inf inventory")
        _assert(all(env.inventory >= -1e-6), "inventory never goes meaningfully negative")
        _assert(all(env.backlog >= -1e-6), "backlog never goes negative")

    _assert(dones[0] is True, "episode terminates at episode_length")


def test_environment_metrics():
    print("\n[test_environment_metrics]")
    env = SupplyChainEnv(episode_length=104, seed=1)
    env.reset(seed=1)
    # Pass-through policy: order = last period's observed demand.
    # This should be close to bullwhip_ratio ~= 1.0 for every echelon,
    # since it does not amplify variance at all.
    last_orders = {i: 20.0 for i in range(env.n_agents)}
    for _ in range(104):
        obs, rewards, dones, infos = env.step(last_orders)
        last_orders = {i: infos[i]["demand_received"] for i in range(env.n_agents)}

    bw = env.bullwhip_ratios()
    for name, ratio in bw.items():
        _assert(abs(ratio - 1.0) < 0.1, f"pass-through policy bullwhip ratio for {name} is ~1.0 (got {ratio:.3f})")

    service = env.service_levels()
    for name, level in service.items():
        # Allow a tiny floating-point tolerance: on some platforms/CPUs,
        # accumulated division can land a hair above 1.0 (e.g.
        # 1.0000000000000002) due to floating-point rounding order, even
        # though the true fraction can never exceed 1.0.
        _assert(-1e-9 <= level <= 1.0 + 1e-9, f"service level for {name} is a valid fraction (got {level!r})")


def test_mlp_gradient_correctness():
    print("\n[test_mlp_gradient_correctness]")
    rng = np.random.default_rng(0)
    net = MLP([4, 8, 3], out_activation="tanh", seed=1)
    x = rng.normal(size=(5, 4))

    cache = {}
    out = net.forward(x, cache=cache)
    d_out = rng.normal(size=out.shape)
    grads = net.backward(cache, d_out)

    def loss_fn(params):
        net.set_flat_params(params)
        o = net.forward(x)
        return np.sum(o * d_out)

    eps = 1e-5
    base_params = net.get_flat_params()
    max_rel_error = 0.0
    for key in ["W0", "W1", "b0", "b1"]:
        idx = tuple(rng.integers(0, s) for s in base_params[key].shape)
        p_plus = {k: v.copy() for k, v in base_params.items()}
        p_minus = {k: v.copy() for k, v in base_params.items()}
        p_plus[key][idx] += eps
        p_minus[key][idx] -= eps
        numeric_grad = (loss_fn(p_plus) - loss_fn(p_minus)) / (2 * eps)
        net.set_flat_params(base_params)
        analytic_grad = grads[key][idx] * x.shape[0]  # backward() averages; loss_fn sums
        rel_error = abs(numeric_grad - analytic_grad) / (abs(numeric_grad) + 1e-8)
        max_rel_error = max(max_rel_error, rel_error)

    _assert(max_rel_error < 1e-4, f"backprop gradients match numerical gradients (max rel error {max_rel_error:.2e})")


def test_adam_and_normalizer():
    print("\n[test_adam_and_normalizer]")
    net = MLP([3, 4, 1], seed=0)
    opt = Adam(net.params, lr=1e-2)
    x = np.random.default_rng(0).normal(size=(10, 3))
    cache = {}
    out_before = net.forward(x, cache=cache).copy()
    grads = net.backward(cache, np.ones_like(out_before))
    opt.step(net.params, grads)
    out_after = net.forward(x)
    _assert(not np.allclose(out_before, out_after), "Adam step actually changes network output")

    norm = RunningNormalizer(dim=3)
    data = np.random.default_rng(0).normal(loc=50, scale=10, size=(1000, 3))
    for row in data:
        norm.update(row)
    normed = norm.normalize(data)
    _assert(abs(normed.mean()) < 0.5, "RunningNormalizer output is approximately zero-mean")
    _assert(abs(normed.std() - 1.0) < 0.5, "RunningNormalizer output is approximately unit-variance")


def test_maddpg_trainer_smoke():
    print("\n[test_maddpg_trainer_smoke]")
    env = SupplyChainEnv(episode_length=15, seed=0)
    obs_dims = [env.obs_dim(i) for i in range(env.n_agents)]
    trainer = MADDPGTrainer(
        obs_dims=obs_dims, global_state_dim=env.global_state_dim(),
        max_actions=env.max_order, agent_names=env.agent_names,
        batch_size=8, buffer_capacity=200, seed=0,
    )

    obs = env.reset(seed=0)
    for t in range(15):
        gs = env.global_state()
        actions = trainer.act(obs, explore=True)
        _assert(all(0.0 <= a <= env.max_order[i] for i, a in actions.items()), "actions respect [0, max_order] bounds")
        obs_next, rewards, dones, infos = env.step(actions)
        ngs = env.global_state()
        shaped = trainer.shape_rewards(rewards, actions)
        trainer.store(obs, actions, shaped, obs_next, dones, gs, ngs)
        obs = obs_next

    for _ in range(5):
        metrics = trainer.update()
    _assert(metrics is not None, "update() runs once buffer has enough samples")
    _assert(np.isfinite(metrics["critic_loss"]), "critic loss is finite after a few updates")


def test_checkpoint_roundtrip():
    print("\n[test_checkpoint_roundtrip]")
    env = SupplyChainEnv(episode_length=10, seed=0)
    obs_dims = [env.obs_dim(i) for i in range(env.n_agents)]
    trainer = MADDPGTrainer(
        obs_dims=obs_dims, global_state_dim=env.global_state_dim(),
        max_actions=env.max_order, agent_names=env.agent_names, seed=0,
    )
    # Simulate some observation-normalizer statistics accumulating.
    obs = env.reset(seed=0)
    for t in range(10):
        actions = trainer.act(obs, explore=True)
        obs, rewards, dones, infos = env.step(actions)

    mean_before = trainer.obs_normalizers[0].mean.copy()
    var_before = trainer.obs_normalizers[0].var.copy()
    action_before = trainer.agents[0].act(obs[0], explore=False)

    # Use the OS's proper temp directory (works on Windows, macOS, Linux)
    # instead of a hardcoded Unix-style "/tmp" path, which does not exist
    # on Windows and was the cause of a FileNotFoundError there.
    tmp_dir = tempfile.gettempdir()
    tmp_path = os.path.join(tmp_dir, "_test_checkpoint.npz")
    save_checkpoint(trainer, tmp_path)

    # Fresh trainer with different seed -> different weights/normalizer by default
    trainer2 = MADDPGTrainer(
        obs_dims=obs_dims, global_state_dim=env.global_state_dim(),
        max_actions=env.max_order, agent_names=env.agent_names, seed=999,
    )
    _assert(
        not np.allclose(trainer2.obs_normalizers[0].mean, mean_before),
        "fresh trainer has different (default) normalizer stats before loading",
    )

    load_checkpoint(trainer2, tmp_path)
    _assert(np.allclose(trainer2.obs_normalizers[0].mean, mean_before), "normalizer mean restored exactly after load_checkpoint")
    _assert(np.allclose(trainer2.obs_normalizers[0].var, var_before), "normalizer var restored exactly after load_checkpoint")

    action_after = trainer2.agents[0].act(obs[0], explore=False)
    _assert(abs(action_before - action_after) < 1e-6, "actor produces identical action after checkpoint roundtrip")

    os.remove(tmp_path)


def test_baselines_sane():
    print("\n[test_baselines_sane]")
    env = SupplyChainEnv(episode_length=104, demand_generator=DemandGenerator(mode="stationary", seed=1), seed=1)
    lead_times = [c.lead_time for c in env.configs]

    bs = BaseStockPolicy(env.n_agents, lead_times, demand_mean=20.0, demand_std=4.0)
    env = run_policy_episode(env, bs, seed=1)
    _assert(env.total_cost() > 0, "base-stock policy produces positive total cost")
    service = env.service_levels()
    _assert(all(v > 0.9 for v in service.values()), "base-stock policy achieves high service levels")

    env2 = SupplyChainEnv(episode_length=104, demand_generator=DemandGenerator(mode="stationary", seed=1), seed=1)
    ss = SSPolicy(env2.n_agents, lead_times, demand_mean=20.0, demand_std=4.0)
    env2 = run_policy_episode(env2, ss, seed=1)
    _assert(env2.total_cost() > 0, "(s,S) policy produces positive total cost")

    env3 = SupplyChainEnv(episode_length=104, demand_generator=DemandGenerator(mode="stationary", seed=1), seed=1)
    naive = NaivePassThrough(env3.n_agents)
    env3 = run_policy_episode(env3, naive, seed=1)
    bw3 = env3.bullwhip_ratios()
    _assert(all(abs(v - 1.0) < 0.5 for v in bw3.values()), "naive pass-through has bullwhip ratio near 1.0")


def run_all():
    tests = [
        test_environment_basic,
        test_environment_metrics,
        test_mlp_gradient_correctness,
        test_adam_and_normalizer,
        test_maddpg_trainer_smoke,
        test_checkpoint_roundtrip,
        test_baselines_sane,
    ]
    failures = []
    for t in tests:
        try:
            t()
        except AssertionError as e:
            failures.append((t.__name__, str(e)))
            print(f"  FAIL: {e}")

    print("\n" + "=" * 60)
    if failures:
        print(f"{len(failures)}/{len(tests)} test functions had failures:")
        for name, msg in failures:
            print(f"  - {name}: {msg}")
        sys.exit(1)
    else:
        print(f"All {len(tests)} test functions passed.")


if __name__ == "__main__":
    run_all()
