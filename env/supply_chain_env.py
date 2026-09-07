"""
Multi-Echelon Supply Chain Inventory Coordination Environment
================================================================

A multi-agent reinforcement learning environment modeling the classic
"Beer Game" style supply chain: a linear chain of echelons (e.g.
Retailer -> Distributor -> Manufacturer -> Supplier) where each echelon
is an independent learning agent.

Key realism features (this is what separates this from a toy env):
  - Partial observability: each agent only sees its own inventory,
    backlog, pipeline (in-transit) orders, and recent orders received
    from its downstream neighbor. It does NOT see other echelons'
    state or the true end-consumer demand (only the retailer sees that).
  - Stochastic lead times: shipments take `lead_time` steps to arrive,
    modeled as a per-agent pipeline queue.
  - Capacity constraints: each agent has a maximum production/shipping
    capacity per step, so it cannot always fulfill what's ordered.
  - Backlog accounting: unmet demand is *not* lost, it's carried forward
    as a backlog that must be fulfilled later (this is what drives the
    bullwhip effect -- agents overorder to buffer against backlog risk).
  - Heterogeneous economics: each agent has its own holding cost,
    backlog penalty, and ordering cost.

This follows the PettingZoo "Parallel API" convention (all agents act
simultaneously each step) so it can be dropped into PettingZoo-based
training code with a thin wrapper, but has zero external MARL
dependencies -- it only needs numpy.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# --------------------------------------------------------------------------- #
# Demand generation
# --------------------------------------------------------------------------- #

class DemandGenerator:
    """
    Generates end-consumer demand seen by the retailer (first echelon).

    Modes:
      - "stationary": Gaussian noise around a fixed mean.
      - "seasonal": sinusoidal seasonality + noise (mimics weekly/monthly
        retail cycles, similar in spirit to M5 data patterns).
      - "shock": stationary demand with occasional large demand spikes
        (promotions) and demand drops -- used to stress-test whether
        learned policies adapt better than fixed classical policies
        under non-stationarity.
    """

    def __init__(
        self,
        mode: str = "seasonal",
        base_mean: float = 20.0,
        base_std: float = 4.0,
        seasonal_amplitude: float = 6.0,
        seasonal_period: int = 52,
        shock_prob: float = 0.03,
        shock_multiplier_range: Tuple[float, float] = (2.0, 3.5),
        seed: Optional[int] = None,
    ):
        self.mode = mode
        self.base_mean = base_mean
        self.base_std = base_std
        self.seasonal_amplitude = seasonal_amplitude
        self.seasonal_period = seasonal_period
        self.shock_prob = shock_prob
        self.shock_multiplier_range = shock_multiplier_range
        self.rng = np.random.default_rng(seed)

    def reset(self, seed: Optional[int] = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)

    def sample(self, t: int) -> float:
        mean = self.base_mean
        if self.mode in ("seasonal", "shock"):
            mean = self.base_mean + self.seasonal_amplitude * np.sin(
                2 * np.pi * t / self.seasonal_period
            )
        demand = self.rng.normal(mean, self.base_std)

        if self.mode == "shock" and self.rng.random() < self.shock_prob:
            mult = self.rng.uniform(*self.shock_multiplier_range)
            demand *= mult

        return float(max(0.0, demand))


# --------------------------------------------------------------------------- #
# Per-agent configuration
# --------------------------------------------------------------------------- #

@dataclass
class EchelonConfig:
    name: str
    holding_cost: float = 1.0          # cost per unit of inventory held per step
    backlog_penalty: float = 2.0       # cost per unit of unmet demand per step
    ordering_cost: float = 0.1         # cost per unit ordered (variable cost)
    capacity: float = 40.0             # max units this echelon can ship/produce per step
    lead_time: int = 2                 # steps between order placed and order arriving
    init_inventory: float = 20.0
    order_history_len: int = 4         # rolling window of recent incoming orders in obs


# --------------------------------------------------------------------------- #
# The environment
# --------------------------------------------------------------------------- #

class SupplyChainEnv:
    """
    Linear multi-echelon supply chain environment.

    Chain topology (index 0 is closest to the end consumer):
        agents[0] = Retailer      <- sees true end-consumer demand
        agents[1] = Distributor
        agents[2] = Manufacturer
        agents[3] = Supplier      <- assumed infinite upstream supply

    Each timestep, simultaneously:
      1. Each agent (except the last) places an order with its upstream
         neighbor (the action). The supplier's upstream is an infinite
         raw-material source (always fully available).
      2. Each agent ships out product to fulfill its downstream neighbor's
         *pending* order (from `lead_time` steps ago, subject to its own
         capacity and available inventory) -- i.e. shipments already in
         the pipeline arrive.
      3. Inventory, backlog and pipeline queues update.
      4. Local reward (negative cost) is computed for every agent.

    Observation (per agent) is partial:
        [ inventory_on_hand, backlog, pipeline_total,
          pipeline_bucket_1 ... pipeline_bucket_L,
          recent_incoming_orders (last `order_history_len` steps),
          normalized_timestep ]

    Action (per agent): a single continuous, non-negative order quantity
    placed with the upstream neighbor (clipped to [0, max_order]).
    """

    def __init__(
        self,
        configs: Optional[List[EchelonConfig]] = None,
        demand_generator: Optional[DemandGenerator] = None,
        episode_length: int = 104,
        max_order_multiplier: float = 3.0,
        seed: Optional[int] = None,
    ):
        if configs is None:
            configs = [
                EchelonConfig(name="Retailer",     holding_cost=1.0, backlog_penalty=2.0, ordering_cost=0.05, capacity=45, lead_time=2),
                EchelonConfig(name="Distributor",  holding_cost=0.8, backlog_penalty=1.6, ordering_cost=0.08, capacity=45, lead_time=2),
                EchelonConfig(name="Manufacturer", holding_cost=0.6, backlog_penalty=1.2, ordering_cost=0.10, capacity=50, lead_time=3),
                EchelonConfig(name="Supplier",     holding_cost=0.4, backlog_penalty=0.8, ordering_cost=0.12, capacity=55, lead_time=3),
            ]
        self.configs = configs
        self.n_agents = len(configs)
        self.agent_names = [c.name for c in configs]
        self.agent_ids = list(range(self.n_agents))

        self.demand_gen = demand_generator or DemandGenerator(seed=seed)
        self.episode_length = episode_length
        self.max_order_multiplier = max_order_multiplier
        self.rng = np.random.default_rng(seed)

        # Per-agent max feasible order (used to scale/clip actions)
        self.max_order = [c.capacity * max_order_multiplier for c in configs]

        self._t = 0
        self.reset(seed=seed)

    # ------------------------------------------------------------------ #
    # Core gym-like API
    # ------------------------------------------------------------------ #

    def reset(self, seed: Optional[int] = None) -> Dict[int, np.ndarray]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.demand_gen.reset(seed=seed)

        self._t = 0
        n = self.n_agents

        self.inventory = np.array([c.init_inventory for c in self.configs], dtype=np.float64)
        self.backlog = np.zeros(n, dtype=np.float64)

        # Pipeline[i] is a queue (list) of shipments from echelon i+1 -> i,
        # each entry is (arrival_time_remaining, quantity). Staggered so
        # that on reset there's already a smooth, in-flight steady-state
        # pipeline (arrivals spread across the next `lead_time` steps)
        # rather than everything landing simultaneously.
        self.pipeline: List[List[List[float]]] = [
            [[k + 1, c.init_inventory / max(c.lead_time, 1)] for k in range(c.lead_time)]
            for c in self.configs
        ]

        # Order history (what each agent received as incoming demand each step)
        self.order_history = [
            [self.configs[i].init_inventory / max(self.configs[i].lead_time, 1)]
            * self.configs[i].order_history_len
            for i in range(n)
        ]

        # Pending order requests each agent has placed upstream, awaiting shipment.
        # pending_orders[i] = queue of orders agent i placed with agent i+1
        self.pending_orders_upstream = [[] for _ in range(n)]

        # Track last true demand for retailer-facing metrics
        self.last_external_demand = 0.0

        # Logging buffers for the episode (useful for bullwhip metric etc.)
        self.history = {
            "orders_placed": [[] for _ in range(n)],
            "demand_received": [[] for _ in range(n)],
            "inventory": [[] for _ in range(n)],
            "backlog": [[] for _ in range(n)],
            "cost": [[] for _ in range(n)],
            "external_demand": [],
            "fulfilled": [[] for _ in range(n)],
        }

        return self._get_all_observations()

    def step(
        self, actions: Dict[int, float]
    ) -> Tuple[Dict[int, np.ndarray], Dict[int, float], Dict[int, bool], Dict[int, dict]]:
        n = self.n_agents
        orders = np.array([float(np.clip(actions[i], 0, self.max_order[i])) for i in range(n)])

        # -------------------------------------------------------------- #
        # 1. Determine incoming demand for each echelon this step.
        #    - Echelon 0 (retailer) faces true end-consumer demand.
        #    - Echelon i>0 faces the order just placed by echelon i-1.
        # -------------------------------------------------------------- #
        external_demand = self.demand_gen.sample(self._t)
        self.last_external_demand = external_demand

        incoming_demand = np.zeros(n)
        incoming_demand[0] = external_demand
        for i in range(1, n):
            incoming_demand[i] = orders[i - 1]

        # -------------------------------------------------------------- #
        # 2. Each echelon ships out to fulfill (backlog + incoming demand),
        #    limited by capacity and on-hand inventory. The supplier
        #    (last echelon) has an infinite upstream, so it can always
        #    fully replenish -- but is still capacity constrained on how
        #    much it can *produce and ship* in one step.
        # -------------------------------------------------------------- #
        fulfilled = np.zeros(n)
        for i in range(n):
            total_demand_i = self.backlog[i] + incoming_demand[i]
            available = self.inventory[i]
            ship_qty = min(total_demand_i, available, self.configs[i].capacity)
            ship_qty = max(ship_qty, 0.0)
            fulfilled[i] = ship_qty

            self.inventory[i] -= ship_qty
            self.backlog[i] = max(total_demand_i - ship_qty, 0.0)

        # -------------------------------------------------------------- #
        # 3. Orders placed this step enter the upstream pipeline (except
        #    for the last echelon, whose "order" is a production order
        #    against unlimited raw materials -- it just enters its own
        #    production pipeline).
        # -------------------------------------------------------------- #
        for i in range(n):
            self.pipeline[i].append([self.configs[i].lead_time, orders[i]])

        # -------------------------------------------------------------- #
        # 4. Advance pipelines: anything with 0 remaining lead time arrives
        #    into inventory now. Everything else has its countdown reduced.
        # -------------------------------------------------------------- #
        for i in range(n):
            arrived = 0.0
            still_pending = []
            for entry in self.pipeline[i]:
                entry[0] -= 1
                if entry[0] <= 0:
                    arrived += entry[1]
                else:
                    still_pending.append(entry)
            self.inventory[i] += arrived
            self.pipeline[i] = still_pending

        # -------------------------------------------------------------- #
        # 5. Update order-history rolling windows (used in observations
        #    for local demand estimation).
        # -------------------------------------------------------------- #
        for i in range(n):
            self.order_history[i].append(incoming_demand[i])
            if len(self.order_history[i]) > self.configs[i].order_history_len:
                self.order_history[i].pop(0)

        # -------------------------------------------------------------- #
        # 6. Compute local rewards (negative costs).
        # -------------------------------------------------------------- #
        rewards = {}
        infos = {}
        for i in range(n):
            cfg = self.configs[i]
            holding_cost = cfg.holding_cost * self.inventory[i]
            backlog_cost = cfg.backlog_penalty * self.backlog[i]
            order_cost = cfg.ordering_cost * orders[i]
            cost = holding_cost + backlog_cost + order_cost
            rewards[i] = -float(cost)
            infos[i] = {
                "holding_cost": holding_cost,
                "backlog_cost": backlog_cost,
                "order_cost": order_cost,
                "inventory": self.inventory[i],
                "backlog": self.backlog[i],
                "order_placed": orders[i],
                "demand_received": incoming_demand[i],
                "fulfilled": fulfilled[i],
            }

            self.history["orders_placed"][i].append(orders[i])
            self.history["demand_received"][i].append(incoming_demand[i])
            self.history["inventory"][i].append(self.inventory[i])
            self.history["backlog"][i].append(self.backlog[i])
            self.history["cost"][i].append(cost)
            self.history["fulfilled"][i].append(fulfilled[i])
        self.history["external_demand"].append(external_demand)

        self._t += 1
        done = self._t >= self.episode_length
        dones = {i: done for i in range(n)}

        obs = self._get_all_observations()
        return obs, rewards, dones, infos

    # ------------------------------------------------------------------ #
    # Observation construction (this is where partial observability is
    # enforced -- each agent's obs vector ONLY contains its own local state)
    # ------------------------------------------------------------------ #

    def _get_observation(self, agent_id: int) -> np.ndarray:
        cfg = self.configs[agent_id]
        pipeline_buckets = np.zeros(cfg.lead_time)
        for entry in self.pipeline[agent_id]:
            remaining, qty = entry
            bucket = int(np.clip(remaining - 1, 0, cfg.lead_time - 1))
            pipeline_buckets[bucket] += qty
        pipeline_total = pipeline_buckets.sum()

        recent_orders = np.array(self.order_history[agent_id], dtype=np.float64)

        obs = np.concatenate([
            [self.inventory[agent_id]],
            [self.backlog[agent_id]],
            [pipeline_total],
            pipeline_buckets,
            recent_orders,
            [self._t / self.episode_length],
        ]).astype(np.float64)
        return obs

    def _get_all_observations(self) -> Dict[int, np.ndarray]:
        return {i: self._get_observation(i) for i in range(self.n_agents)}

    def obs_dim(self, agent_id: int) -> int:
        return self._get_observation(agent_id).shape[0]

    def global_state(self) -> np.ndarray:
        """
        Full chain state, used ONLY by the centralized critic during
        training (CTDE) -- never exposed to actors at execution time.
        """
        parts = [self._get_observation(i) for i in range(self.n_agents)]
        parts.append(np.array([self.last_external_demand]))
        return np.concatenate(parts)

    def global_state_dim(self) -> int:
        return self.global_state().shape[0]

    # ------------------------------------------------------------------ #
    # Metrics helpers (used heavily by evaluation/evaluate.py)
    # ------------------------------------------------------------------ #

    def bullwhip_ratios(self) -> Dict[str, float]:
        """
        bullwhip_ratio[i] = Var(orders placed by i) / Var(demand received by i)

        A ratio near 1.0 means the agent is passing demand through without
        amplifying it. A ratio >> 1.0 is the classic bullwhip signature.
        """
        ratios = {}
        for i in range(self.n_agents):
            orders = np.array(self.history["orders_placed"][i])
            demand = np.array(self.history["demand_received"][i])
            var_demand = np.var(demand)
            var_orders = np.var(orders)
            ratios[self.agent_names[i]] = float(var_orders / var_demand) if var_demand > 1e-6 else float("nan")
        return ratios

    def service_levels(self) -> Dict[str, float]:
        """Fraction of total demand fulfilled without backlog, per echelon."""
        levels = {}
        for i in range(self.n_agents):
            demand = np.array(self.history["demand_received"][i])
            fulfilled = np.array(self.history["fulfilled"][i])
            total_demand = demand.sum()
            if total_demand > 0:
                # fulfilled can never exceed total_demand by construction
                # (see step()), but floating-point summation order can
                # differ slightly across platforms/CPUs and land a hair
                # above 1.0 (e.g. 1.0000000000000002). Clip to keep this
                # a clean fraction everywhere it's consumed downstream.
                level = float(np.clip(fulfilled.sum() / total_demand, 0.0, 1.0))
            else:
                level = 1.0
            levels[self.agent_names[i]] = level
        return levels

    def total_cost(self) -> float:
        return float(sum(sum(self.history["cost"][i]) for i in range(self.n_agents)))

    def per_agent_cost(self) -> Dict[str, float]:
        return {self.agent_names[i]: float(sum(self.history["cost"][i])) for i in range(self.n_agents)}

    def cost_fairness_gini(self) -> float:
        """
        Gini coefficient over per-agent total cost. 0 = perfectly even
        cost distribution across echelons, higher = more concentrated
        onto a single node (e.g. dumping backlog risk upstream).
        """
        costs = np.array(list(self.per_agent_cost().values()))
        if costs.sum() == 0:
            return 0.0
        costs_sorted = np.sort(costs)
        nn = len(costs)
        cum = np.cumsum(costs_sorted)
        gini = (nn + 1 - 2 * np.sum(cum) / cum[-1]) / nn
        return float(gini)
