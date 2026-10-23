# D-HELM OPF Reproduction Status

## Project

Independent reproduction of the paper's D-HELM-based physics-informed deep reinforcement learning approach for optimal power flow.

## Reproduction Approach

The implementation is being developed from the mathematical and methodological information available in the paper.

The project is intentionally separating:

1. Information explicitly specified by the paper.
2. Information derived from the paper's equations and algorithm.
3. Implementation choices required to reproduce the method.
4. Parameters or system details that depend on the electronic companion or author-provided code.

This distinction is maintained so that assumptions are not presented as exact author implementation details.

---

## Current Progress

### 1. Repository and project structure

**Status: Complete**

The project repository and Python package structure have been created.

### 2. IEEE 5-bus power-system model

**Status: Complete**

The IEEE 5-bus network has been implemented using pandapower.

Validated network characteristics:

* Buses: 5
* Loads: 3
* Generators: 3
* Lines: 6
* Ybus dimension: 5 × 5

A conventional AC power-flow calculation has also been successfully executed.

### 3. HELM physics infrastructure

**Status: NumPy reference solver and differentiable PyTorch D-HELM both complete and validated.**

Implemented following the paper's equations:

* No-load germ V₀ = 1, W₀ = 1, Q₀ = 0 (paper, after Eq. 11).
* Inverse-voltage recursion, Eq. (11).
* Coefficient matrix A with the Y^tr / Y^sh split, Eq. (12).
* R_n and U_n vectors for PQ, PV and slack buses, Eq. (13).
* Termination on the Eq. (14a)–(14c) mismatches, with Algorithm 1's settings ε = 10⁻⁵ and n_max = 30.

Validation (`scripts/test_helm_random_vs_pandapower.py`): 200 random scenarios varying the slack and PV setpoints (0.95–1.05 p.u.), demand and wind.

* With a tight tolerance, the solver matches pandapower to 7×10⁻¹¹.
* With the paper's tolerance, 200/200 scenarios converge using 4–6 terms, with a maximum voltage error of 9×10⁻⁷.

This NumPy solver is the reference implementation. The differentiable D-HELM is validated against it.

#### Differentiable D-HELM (`helm/differentiable.py`)

The differentiable version follows Sec. 3.2.2 and Eq. (17) of the paper:

* `DHelmNetwork` precomputes the constant network data once: Ybus, the Y^tr / Y^sh split, A with its LU factors, and the incidence matrices.
* The linear assignment (state, normalized action) → (P, Q, |v|_sp) is differentiable, so gradients start from the policy output.
* The batched recursion uses real arithmetic and needs one `lu_solve` per series order for the whole minibatch.
* The Eq. (14) stopping test is applied outside the autograd graph. The loop stops once every sample in the batch has converged. The gradient is exactly that of the resulting N-term series.

Reproduction choices:

* float64 inside the solver.
* The ∇U_n expression in the paper's appendix (A.13) shows only R(V)·R(V) terms, while Eq. (13) uses Re(V*V). Autograd differentiates Eq. (13) as written, and gradcheck verifies it.

Validation (`scripts/test_dhelm_differentiable.py`):

* Torch matches NumPy at the same number of terms to 2×10⁻¹⁶.
* With adaptive stopping, the solution matches pandapower to 1.5×10⁻⁷.
* 500/500 random samples over the full action range converge within n_max.
* `torch.autograd.gradcheck` passes with respect to both action and state.
* The autograd Jacobian of |v|, slack P and PV/slack Q with respect to the action matches pandapower finite differences to 7×10⁻¹¹ (relative).
* Forward plus backward for a minibatch of 32 takes about 10 ms on CPU.

### 4. Operating-condition uncertainty

**Status: Partially complete**

The uncertainty model currently represents the paper's stated structure for the IEEE 5-bus case:

* 3 demand active-power variables.
* 3 demand reactive-power variables.
* 2 wind-power variables.
* Total uncertainty dimension: 8.

The documented uncertainty ranges currently implemented are:

* Demand scaling: [0.60, 1.10]
* Wind scaling: [0.40, 1.00]

Demand uncertainty has been tested and verified to correctly modify the pandapower load values.

The resulting uncertainty-state vector has also been tested and verified to have dimension 8.

### 5. Wind-bus configuration

**Status: Reproduction assumption**

The paper specifies two wind buses for the 5-bus case but does not give their placement or ratings. The placement and ratings would be in the electronic companion, which is not publicly available.

The reproduction assumes:

* Wind buses: 1 and 4 (zero-based pandapower indices).
* Rated capacity: 100 MW per wind unit.

Both values are configurable in `UncertaintyConfig`. Results that depend on them should not be read as an exact reproduction of the paper's renewable setup.

Each wind unit is a pandapower `sgen` with `type="WP"` and `controllable=False`. That tag separates wind units from the conventional controllable `sgen` units, such as the 170 MW unit at bus 0.

### 6. Action space (5-bus)

**Status: Complete. The mapping is derived from Table 1 and the case data, not assumed.**

Table 1 defines the policy output as |B_v ∪ B_s| + |G / G_s| = 4 + 4 = 8. `action_config_from_network()` reads the mapping directly from the network:

* Voltage actions are bus 0, 2, 4 (PV) and bus 3 (slack), each limited to [0.9, 1.1] p.u. by the bus limits.
* Generator P actions are gen0@bus0 (0–40 MW), sgen0@bus0 (0–170 MW), gen1@bus2 (0–520 MW) and gen2@bus4 (0–600 MW).
* The slack unit, ext_grid@bus3 (0–200 MW), is not an action. Its P and Q are critic outputs.
* Costs come from `net.poly_cost`: 14, 15, 30, 10 $/MWh for the four units and 40 $/MWh for the slack.

Two further checks support this mapping:

* The paper's HELM output dimension, 2(|B| + |G| + |L|) = 2(5 + 5 + 6) = 32, matches five generating units.
* The paper's L6 current limit of 0.6 kA matches `line.max_i_ka` for line 5 (bus 3–4).

The ordering of the four generator actions (by bus, then `gen` before `sgen`) is a reproduction convention, because the paper does not list one.

Validation: `scripts/test_action_config.py` and `scripts/test_action_decoder.py`. The decoder test also confirms that HELM matches pandapower (max error 5×10⁻¹¹) over 50 random actions with random uncertainty.

### 7. Physics-driven critic (`models/dhelm_critic.py`)

**Status: Complete and validated.**

`DHELMCritic` takes the batched D-HELM solution and computes the critic quantities, all differentiable with respect to the action and the state:

* Slack P, Eq. (15).
* PQ-bus voltages, Eq. (16a).
* Slack P limits with optional ramping, Eq. (16b).
* Unit Q at the voltage buses, Eq. (16c).
* Sending- and receiving-end line currents, Eqs. (16d)–(16e).
* The violation vector z, Eq. (6).
* The reward r, Eq. (5).

PV and slack voltages are not constraints, because they are actions bounded by Eq. (18). The Lagrangian of Eqs. (19)–(20) is left to the training algorithm.

For the 5-bus case there are 10 constraints: v[bus1], p_slack, q at buses 0, 2, 3 and 4, and i_from / i_to for line 0 (1.004 kA) and line 5 (0.602 kA).

Reproduction choices and interpretations:

* **Q limits are per bus (Eq. 16c literally).** A bus limit is the sum of the limits of the controllable units on it. Bus 0 (gen0 ±30 and sgen0 ±127.5) therefore has ±157.5 MVAr. This also matches how pandapower's OPF treats a controllable sgen.
* **η_n = 1 / (x̄_n − x̲_n)** is our reading of Table 1's "η = (Z_max − Z_min)⁻¹". For line currents the bounds of Eqs. (16d)–(16e) are −i … i, so η = 1 / (2 i_max).
* **w_r = 1 / cost_max** is our reading of Table 1's "w_r = cost1max". Here cost_max = Σ f_g(P̄_g) over all units, slack included, which is 32,710 $/h. With w₀ = 2 the reward lies roughly in [1, 2].
* **Ramping (16b) is implemented but off by default.** The case data has no ramp rates, and the paper's states are sampled independently, with no time series. `CriticConfig.ramp_up_mw` / `ramp_down_mw` and the critic input `p_slack_prev_mw` enable it.
* **Lines with no rating have no current constraint.** In case5, lines 1–4 have pandapower's placeholder max_i_ka = 99,999 kA, which comes from MATPOWER rateA = 0.

Validation (`scripts/test_dhelm_critic.py`), on 100 random samples over the full action range:

* Slack P, unit Q per bus, PQ voltage, line currents and cost all match pandapower's `res_*` tables, to 1×10⁻⁸ MW/MVAr or better.
* z and r match an independent NumPy computation that reads the limits directly from the network tables (3×10⁻¹¹).
* Every one of the 10 constraints is violated in at least one sample, so each branch of Eq. (6) is exercised.
* The ramping constraint gives the expected violation.
* gradcheck passes for r and z with respect to both action and state.

### 8. Policy and constrained policy optimization

**Status: Implemented and validated. It learns a feasible policy, but is not yet at the paper's claimed optimality within 500 steps.**

Components:

* `models/policy.py`: a deterministic policy, LSTM(64) → Dense 64 → Dense 64 (SELU), with a tanh output (Eq. 18).
* `environment/state_sampler.py`: draws states in the paper's ranges and keeps the states-only replay buffer (Fig. 2b), stored in episode order so it can return windows of consecutive states.
* `algorithms/dhelm_cpo.py`: the loss −mean(r) + Σ λₙ mean(zₙ) (Eq. 19, A.1) and the λ update (Eq. 20), with λ⁽⁰⁾ = 10, Adam at lr 1e-3 and minibatch 32 (Table 1). There is no critic training and no target network, because gradients flow through the exact D-HELM critic.
* `algorithms/case5_setup.py`: builds the full 5-bus agent in one call.
* `scripts/train_dhelm_case5.py`: writes `metrics.csv`, `policy.pt` and `training.png` to `results/runs/<name>/` (git-ignored).
* `scripts/evaluate_vs_opf.py`: compares the policy with pandapower's AC OPF on 100 held-out states.

Reproduction choices:

* **Policy input = 8 states + the previous normalized P actions of the 4 non-slack units**, giving 12 inputs. The Table 1 symbol is unreadable. |G/G_s| fits the 5- and 30-bus columns, while the 118- and 300-bus columns are one larger. Our derived *output* sizes match Table 1 for all four systems.
* **The LSTM runs autoregressively over a window of L consecutive states**, feeding back its own previous generator action. That action is 0 at the start of a window. Episodes are 25 steps long, inferred from "20 episodes (≤ 500 steps)".
* **States are scaled to [-1, 1]** with the sampling ranges before entering the network.
* **η_λ = 1.0 and λ_max = 1000.** The paper gives neither. As written, Eq. (20) never decreases λ.

Matching the OPF reference to the paper's problem. Two adjustments are needed, both found during this step:

1. **Unrated lines must be unconstrained in the OPF.** pandapower turns max_i_ka = 99,999 into ~4×10⁷ MVA limits. The interior-point multipliers then blow up and `runopp` converges on only 28/100 states with high wind at bus 4. With those lines unconstrained it converges on 100/100, and the costs on the previously converging states are unchanged (0.02 $/h).
2. **The slack voltage must be free.** pandapower's OPF holds the ext_grid at its setpoint (1.0 p.u.). In the paper the slack voltage is an action (B_s in Eq. 4). For evaluation the ext_grid is therefore replaced by an equivalent slack `gen`, which the OPF sets to about 1.095 p.u. That lowers the mean OPF cost from 9,518 to 8,541 $/h. Against the fixed-slack OPF, a trained policy appeared to beat the "optimum" on more than half the states.

Line limits already match: pandapower's OPF uses current magnitude (`OPF_FLOW_LIM = 2`), as in Eqs. (16d)–(16e). Each OPF solution passed through the critic reproduces its own cost to 0.02 $/h, with Σz ≤ 5×10⁻⁷.

Results on 100 held-out states, against the matched OPF (mean 8,541 $/h):

| Run | Steps | Feasible | Mean gap | Median gap |
|---|---|---|---|---|
| L = 1 | 500 | 97% | +31.5% | +33.9% |
| L = 5 | 500 | 94% | +22.8% | +23.1% |
| L = 1, batch 160 | 500 | 94% | +24.6% | +27.1% |
| L = 1 | 2,500 | 100% | +12.4% | +10.3% |

Timing: the policy plus critic takes about 9 ms for all 100 states, against about 800 ms per state for the OPF.

Observations:

* **Violations collapse within about 100 steps, then the cost falls slowly.** At 500 steps the policy is feasible but not near-optimal, whereas the paper reports convergence within 500 steps. The cost is still falling at 2,500 steps.
* **L = 5 does about as well as L = 1 with a batch of 160.** Its advantage comes from seeing 5× more states per update, not from LSTM memory. That is expected, since states are i.i.d. and ramping is off.
* **λ barely moves (10 → 11.4)**, because z ≈ 0 after the first ~100 steps.
* **The number of D-HELM terms N rises from 5 to 9 during training**, as the policy moves toward more heavily loaded, cheaper operating points.

Validation (`scripts/test_policy_training.py`):

* Shapes and the 12-input layout are correct.
* The autoregressive feedback works as intended.
* Buffer windows are consecutive states.
* Gradients reach all 28,808 parameters through the critic.
* The Eq. (20) update and clipping are correct.
* A 60-step smoke run lowers the loss.

### 9. Remaining work

* Closing the optimality gap: longer training, the learning rate, the λ schedule, and the post-training "few gradient steps" refinement of Remark 5.
* Baselines: SAC, PPO and DDPG (Tianshou, gymnasium environment) and MBPO.
* The 30-, 118- and 300-bus systems.

---

## Validation Policy

Each major component will be tested independently before being incorporated into the final training pipeline.

The intended validation sequence is:

1. Validate IEEE 5-bus network.
2. Validate uncertainty representation.
3. Validate demand uncertainty propagation.
4. Validate HELM power-flow solution against conventional AC power flow.
5. Validate D-HELM-derived physics quantities.
6. Integrate the physics model with the DRL architecture.
7. Train and evaluate the reproduction.
8. Compare reproduced results with the paper's reported results.

---

## Reproduction Limitations

The reproduction is being developed without relying on an available author implementation.

Where information required by the paper is unavailable from the publicly accessible material, the corresponding parameter is kept explicit and configurable rather than silently guessed.

Such cases will be identified in this document and in the relevant source-code documentation.

---

## Current Reproduction State

The following are complete and validated:

* The IEEE 5-bus network and the uncertainty-state representation.
* The NumPy HELM reference solver, which matches pandapower.
* The action space derived from the network.
* The differentiable, batched PyTorch D-HELM, with its forward pass and gradients both verified.
* The physics-driven critic: Eqs. (15), (16), (6) and (5), checked against pandapower and verified with gradcheck.

* The policy and the Lagrangian training (Eqs. 18–20), plus an evaluation against an OPF reference matched to the paper's problem.

Training learns feasible policies (100% feasible after 2,500 steps), but the gap to the OPF is still +12% and the cost is still falling. The next target is closing that gap (Section 9).
