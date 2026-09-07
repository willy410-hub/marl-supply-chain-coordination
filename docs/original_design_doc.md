# MARL Environment Design Challenge — Solution
## Domain: Multi-Echelon Supply Chain Inventory Coordination

---

## 1. Domain Selection

**Chosen Domain:**
`Multi-Echelon Supply Chain Inventory Coordination`

**Domain Motivation & Value:**

Supply chains are a naturally multi-agent system that classical operations research has studied for decades, but almost always through centralized optimization or simple heuristics — not through agents that independently learn to coordinate under realistic information constraints. Each node in the chain (retailer, distributor, manufacturer, supplier) observes only its own inventory, backlog, and the orders placed by its immediate downstream neighbor — never the true end-consumer demand or the state of other echelons. This partial observability, combined with local cost-minimizing incentives that don't always align with total chain efficiency, produces the well-documented bullwhip effect: small demand fluctuations amplify into large order swings as they propagate upstream. The domain is underexplored in MARL specifically because most bullwhip research uses fixed analytical policies (base-stock, (s,S) rules) rather than learning agents, and because building a realistic simulator requires blending inventory theory, stochastic demand modeling, and MARL infrastructure — a combination few groups pursue together. The real-world payoff is significant: global supply chain disruptions (most visibly post-2020) have made resilient, adaptive replenishment policies a major economic priority, and learned coordination policies that dampen the bullwhip effect could materially reduce both stockouts and excess inventory costs across entire industries.

---

## 2. Technical Specification

**Agent Types & State Space:**

Agents are heterogeneous echelon controllers arranged along a supply chain network (linear chain or branching, e.g., one manufacturer feeding multiple regional distributors). Each agent differs in holding cost, backlog penalty, production/shipping capacity, and lead time. Observation is partial: each agent sees its own current inventory position, pipeline inventory (orders already placed but not yet arrived), backlog of unfulfilled orders, and a short window of recent orders received from its downstream neighbor — but not downstream agents' inventory levels, not the true end-consumer demand (only the retailer-facing node sees that), and not other echelons' state. State space dimensions include inventory level, backlog quantity, in-transit orders bucketed by remaining lead time, and a rolling order-history window used for local demand estimation.

**Action Space & Interdependence:**

*Interdependence Type:* `Mixed-Motive (Cooperation + Competition)`

Each agent's action is the order/production quantity to place with its upstream neighbor this timestep, bounded by that neighbor's production or shipping capacity. Interdependence is both spatial and temporal: an upstream agent's fulfilled shipment quantity directly constrains what inventory becomes available downstream, while lead-time delays mean today's ordering decision only affects downstream state several timesteps later — this delayed, cumulative propagation is exactly the mechanism that generates the bullwhip effect. The mixed-motive structure comes from a genuine tension: every agent ultimately wants the chain to serve the end customer efficiently, but each also minimizes its own local holding-and-backlog cost, which incentivizes an agent to over-order as a buffer against uncertainty — a locally rational move that shifts variance and cost upstream onto other agents.

---

## 3. Reward Architecture

*Reward Architecture Approach:* `Centralized Critic with Decentralized Actors`

**Reward Model Design:**

Each agent receives a local, per-timestep reward: r_local = −(holding_cost * inventory_on_hand) − (backlog_penalty * unmet_demand) − (ordering_cost * order_quantity). This is the reward each actor optimizes at execution time, using only its own local observation. During training, however, a centralized critic has access to the full chain state — every echelon's inventory, backlog, and the realized bullwhip ratio (variance of orders placed / variance of true demand) — and uses this to compute a shared value estimate that helps each decentralized actor understand how its local ordering decisions contribute to whole-chain cost, not just its own myopic cost. I'd also add an explicit shaping penalty tied to order-variance amplification relative to the immediately upstream neighbor, directly discouraging bullwhip-inducing behavior rather than hoping it falls out of cost minimization alone. Incentivized: smoothing orders to track true demand signal, holding inventory buffers proportional to actual lead-time risk rather than panic-ordering. Penalized: erratic over-ordering, stockouts, and locally "dumping" demand variance onto upstream neighbors to protect one's own cost at their expense.

---

## 4. Implementation Planning

**Evaluation Criteria & Metrics:**

Total chain cost is benchmarked against a centralized-information optimal ordering policy and against classical heuristics (base-stock, (s,S) policies) — the domain has strong analytical baselines, which is actually a gift for evaluation. The core domain-specific metric is the bullwhip ratio per echelon (variance of orders placed ÷ variance of actual demand received); a ratio near 1 indicates the learned policy has suppressed amplification, while a ratio much greater than 1 signals the classic failure mode is still present. Service level (fill rate — percentage of demand met without backlog) captures customer-facing quality. Finally, cost distribution fairness across echelons matters: a policy that achieves low total cost by concentrating stockouts or overproduction onto one node (often the manufacturer, who has the least visibility into true demand) isn't actually solving the coordination problem, just relocating it.

**Required Domain Expertise & Data:**

Required expertise spans inventory theory and operations research (the bullwhip effect literature — Lee, Padmanabhan, and Whang's foundational work — plus base-stock and (s,S) policy theory) and MARL engineering for the CTDE training setup. Data sources: real retail point-of-sale demand data is publicly available (the M5 Forecasting competition dataset is a strong candidate, offering realistic seasonality and promotional spikes), and supply chain network topologies can be synthetic multi-echelon structures calibrated to realistic branching factors when real network data isn't accessible. Existing open-source environments like OR-Gym provide a starting point rather than building the inventory simulator entirely from scratch, which meaningfully de-risks the timeline.

**Estimated Compute Budget Range:**
`$5K-$20K (moderate complexity)` — the underlying dynamics (queueing/inventory equations) are computationally lighter than a hydrology simulator, and strong existing baselines and open-source simulators reduce the engineering lift.

**Confidence in 6-Month Feasibility:**
`4` (High) — well-understood theory, public real-world data, and existing simulators to build on all reduce risk relative to a domain built from scratch.

**Anticipated Key Challenges:**

The first challenge is credit assignment across long, variable lead-time delays — an order placed today affects downstream state many steps later, across multiple agents, making it hard for any single agent to learn cause and effect. The centralized critic during training directly targets this, and adding a dense shaping term for local order-variance reduction gives agents a more immediate learning signal than waiting for delayed cost feedback. The second challenge is proving learned policies actually beat classical OR baselines, which are already near-optimal under stationary demand — this is addressed by deliberately testing under non-stationary demand shocks and network disruptions (a sudden capacity loss at one node) where fixed analytical policies can't adapt but a learned policy plausibly can. The third challenge is balancing demand-model realism against simulation speed; resampling from real historical demand data (M5) rather than building a full generative demand model keeps realism high without the compute cost of learning a separate demand simulator.
