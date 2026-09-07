"""
Classical Operations-Research Baseline Policies
=================================================

Implements the two textbook inventory policies referenced in the design
doc's evaluation plan (Lee, Padmanabhan & Whang; base-stock and (s,S)
theory), used as benchmarks against the learned MADDPG policy:

  - Base-stock policy: order-up-to a target inventory position
    (inventory on hand + pipeline - backlog) each period. This is the
    textbook-optimal policy for a single echelon under stationary
    demand and linear holding/backlog costs -- a strong baseline.

  - (s, S) policy: order nothing until inventory position drops to or
    below reorder point `s`, then order up to `S`. More realistic for
    settings with fixed ordering costs (captures lumpier, less frequent
    ordering behavior than pure base-stock).

Also includes a `naive_pass_through` policy (order = last period's
observed demand) as a trivial lower-bound-of-effort baseline, and a
helper to auto-calibrate reasonable base-stock levels from the demand
generator's parameters.
"""

from __future__ import annotations
import numpy as np
from typing import Dict, List, Optional

try:
    from scipy.stats import norm
    _HAVE_SCIPY = True
except ImportError:
    _HAVE_SCIPY = False


def _z_score(service_level: float) -> float:
    """Approximate inverse-CDF of the standard normal without requiring scipy."""
    if _HAVE_SCIPY:
        return float(norm.ppf(service_level))
    # Rational approximation (Beasley-Springer-Moro / Acklam) - good to ~1e-4
    p = service_level
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    p_low, p_high = 0.02425, 1 - 0.02425
    if p < p_low:
        q = np.sqrt(-2 * np.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    elif p <= p_high:
        q = p - 0.5
        r = q * q
        return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
               (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)
    else:
        q = np.sqrt(-2 * np.log(1 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)


class BaseStockPolicy:
    """
    Order-up-to-S policy per echelon:
        inventory_position = on_hand + pipeline - backlog
        order = max(0, S - inventory_position)

    S is set via the standard newsvendor-style safety-stock formula:
        S = mean_demand * (lead_time + 1) + z * std_demand * sqrt(lead_time + 1)

    where z is chosen from a target service level (critical ratio).
    """

    def __init__(self, n_agents: int, lead_times: List[int], demand_mean: float,
                 demand_std: float, target_service_level: float = 0.95):
        self.n_agents = n_agents
        z = _z_score(target_service_level)
        self.S = [
            demand_mean * (lt + 1) + z * demand_std * np.sqrt(lt + 1)
            for lt in lead_times
        ]

    def act(self, env) -> Dict[int, float]:
        actions = {}
        for i in range(self.n_agents):
            inv_position = env.inventory[i] + sum(e[1] for e in env.pipeline[i]) - env.backlog[i]
            order = max(0.0, self.S[i] - inv_position)
            actions[i] = min(order, env.max_order[i])
        return actions

    def name(self) -> str:
        return "BaseStock"


class SSPolicy:
    """
    (s, S) policy: if inventory position <= s, order up to S; else order 0.
    s is set as a fraction of S (reorder point), giving lumpier, less
    frequent ordering than pure base-stock -- more realistic when there's
    a meaningful fixed cost per order placed.
    """

    def __init__(self, n_agents: int, lead_times: List[int], demand_mean: float,
                 demand_std: float, target_service_level: float = 0.95,
                 reorder_fraction: float = 0.6):
        z = _z_score(target_service_level)
        self.S = [
            demand_mean * (lt + 1) + z * demand_std * np.sqrt(lt + 1)
            for lt in lead_times
        ]
        self.s = [reorder_fraction * S_i for S_i in self.S]
        self.n_agents = n_agents

    def act(self, env) -> Dict[int, float]:
        actions = {}
        for i in range(self.n_agents):
            inv_position = env.inventory[i] + sum(e[1] for e in env.pipeline[i]) - env.backlog[i]
            if inv_position <= self.s[i]:
                order = max(0.0, self.S[i] - inv_position)
            else:
                order = 0.0
            actions[i] = min(order, env.max_order[i])
        return actions

    def name(self) -> str:
        return "(s,S)"


class NaivePassThrough:
    """Order = demand observed last period. Trivial baseline (zero foresight)."""

    def __init__(self, n_agents: int):
        self.n_agents = n_agents
        self.last_demand = {i: 20.0 for i in range(n_agents)}

    def act(self, env) -> Dict[int, float]:
        actions = dict(self.last_demand)
        return actions

    def update_after_step(self, infos: dict):
        for i in range(self.n_agents):
            self.last_demand[i] = infos[i]["demand_received"]

    def name(self) -> str:
        return "NaivePassThrough"


def run_policy_episode(env, policy, episode_length: Optional[int] = None, seed: Optional[int] = None):
    """
    Runs one full episode of a classical (non-learning) policy against
    the environment and returns the env (with populated `.history`) for
    metric extraction, mirroring how a trained MADDPG rollout is scored.
    """
    env.reset(seed=seed)
    steps = episode_length or env.episode_length
    for _ in range(steps):
        actions = policy.act(env)
        obs, rewards, dones, infos = env.step(actions)
        if hasattr(policy, "update_after_step"):
            policy.update_after_step(infos)
    return env
