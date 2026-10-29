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

## Notation

Variable names in the code follow the paper's equation symbols. `NOTATION.md` lists every paper symbol with its Python spelling (for example α → `alpha`, λ → `lam`, \|v_i\|^sp → `v_sp`, Y^sh → `Y_sh`).

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

The action vector follows Eq. (4): a = [p_g of G_ns, |v| of B_vs], generator powers first. Runs trained before this change used voltages first; their checkpoints have no `p_first` entry and load with the old layout (`p_first=False`).

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
* **w_r = 1 / f_max** is our reading of Table 1's "w_r = cost1max". Here f_max = Σ f_g(P̄_g) over all units, slack included, which is 32,710 $/h. With w₀ = 2 the reward lies roughly in [1, 2].
* **Ramping (Eq. 3, 16b) is on in the paper protocol** (Section 10). The paper states ramp limits but gives no rates, and the case data has none, so R_u = R_d = 30% of P_max per step is our choice (`RampConfig`). Non-slack units are ramp-limited through the action mapping (the Eq. 18 window is the Eq. 3 window), the slack through the 16b penalty. With `ramp=None` the code reproduces the earlier no-ramp runs bit for bit.
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

### 9. Multi-seed evaluation and refinement (R1 / R2)

**Status: Complete. Results are reported over 5 seeds against a cached, matched OPF reference.**

> **These runs are extensions, not the paper protocol:** 2,500–5,000 steps, window 1, no ramping, voltage-first action layout. The paper-protocol results are in Section 10.

Components:

* **Best checkpoint** (`scripts/train_dhelm_case5.py`). Among the validation evaluations (256 states, seed 10000 + seed, never trained on), it is the one with the lowest validation cost that has ≥ 99% feasible states. If none qualifies, the lowest validation loss is used instead. The output files are `policy_best.pt` and `summary.json`.
* **Optional cosine learning-rate decay** (`--lr-schedule cosine`, decaying to 10% of the initial rate). The default is constant, as in the paper.
* **Cached OPF reference** (`utils/opf_reference.py`). The matched OPF (unrated lines unconstrained, slack voltage free; see Section 8) is solved once for the 100 test states (seed 20000) and stored in `results/opf_reference/`. It is reused only if the states match exactly.
* **Shared evaluation** (`utils/evaluation.py`). This is used by `scripts/evaluate_vs_opf.py` (`--checkpoint best|last`, `--refine none|r1|r2`, `--refine-K K`) and by `scripts/run_seeds.py`, so both report identical numbers.
* **Refinement** (`algorithms/refine.py`). Gradient steps on each state's action through the critic, on J = −r + Σ λₙ zₙ with the trained λ. The policy weights are untouched. The variable is the pre-tanh action, so actions stay in bounds. Each state keeps its best iterate: feasible beats infeasible, then lower cost, then lower violation. The starting action counts as a candidate, and iterates where D-HELM did not converge are rejected, so refinement never makes a state worse. Defaults: Adam, lr 0.03.
  * **R1, paper-faithful (Remark 5).** Only states that fail the feasibility test are refined, and each stops as soon as it is feasible. Feasible states are returned unchanged.
  * **R2, our extension.** Every state is refined for K steps, which lowers the cost as well. The policy becomes a warm start for a first-order OPF solve through the critic.
* **Runners.** `scripts/run_seeds.py` trains N seeds and evaluates the last checkpoint, the best checkpoint, best + R1, and best + R2 at K = 5, 20, 50 and 200. `scripts/compare_experiments.py` prints experiments side by side.

Results: 5 seeds (0–4), 100 test states, mean ± std over seeds. The OPF reference averages 8,541 $/h. Every OPF solution passed through the critic reproduces its cost to 5×10⁻³ $/h, with Σz ≤ 5×10⁻⁷.

Optimality gap against the OPF:

| Variant | 2,500 steps | 2,500 steps, cosine LR | 5,000 steps |
|---|---|---|---|
| last checkpoint | +6.77% ± 3.16 | +9.32% ± 1.87 | +3.52% ± 0.48 |
| best checkpoint | +5.44% ± 0.74 | +9.34% ± 2.01 | +3.44% ± 0.27 |
| best + R1 (K = 20) | +5.50% ± 0.74 | +9.39% ± 2.02 | +3.51% ± 0.25 |
| best + R2 (K = 5) | +4.06% ± 0.45 | +8.27% ± 1.76 | +2.64% ± 0.13 |
| best + R2 (K = 50) | +3.59% ± 0.34 | +7.48% ± 1.64 | +2.41% ± 0.10 |
| best + R2 (K = 200) | +2.53% ± 0.23 | +5.98% ± 1.38 | **+1.78% ± 0.09** |

Feasible test states:

| Variant | 2,500 steps | 2,500 steps, cosine LR | 5,000 steps |
|---|---|---|---|
| last checkpoint | 93.4% | 98.4% | 89.2% |
| best checkpoint | 98.8% | 98.8% | 98.4% |
| best + R1 / R2 | 100% | 100% | 100%¹ |

¹ R2 at K = 5 reaches 99.8%: one state was not yet fixed within 5 steps.

Time per state on CPU, for the 5,000-step run:

* Policy alone: about 0.1 ms.
* Policy + R1: about 1 ms.
* Policy + R2: about 5 ms at K = 20 and about 44 ms at K = 200.
* pandapower OPF: 300–800 ms, depending on the machine.

Findings:

* **Seed noise is large in the last checkpoint.** Its std over seeds is 3.2 percentage points at 2,500 steps. Selecting the best checkpoint by validation cuts this more than 4×, to 0.7, and raises feasibility. All comparisons therefore use the best checkpoint.
* **Training length is the main lever.** The gap shrinks from 5.4% to 3.4% when training doubles from 2,500 to 5,000 steps. The paper reports near-optimal policies within 500 steps, which is not reproduced here.
* **Cosine LR decay hurts** (+9.3% against +5.4%). The policy is under-trained rather than oscillating, so decaying the step size slows the descent. Constant LR (the paper's setting) is kept.
* **R1 reproduces the paper's claim.** A few gradient steps on the infeasible states give 100% feasibility on every seed, with a negligible effect on cost (+0.06 points). R1 touches only the 1–2% of states the best checkpoint leaves infeasible.
* **R2 is the improvement.** It brings the 5,000-step policy to +1.8% ± 0.1 at 100% feasibility, while still running 7–18× faster than the OPF.
* **Why R2 stalls.** In many states the optimum lies on the action bounds (gen2 at 600 MW, other units at 0, voltages at 1.1 p.u.) and on the slack's 0 MW lower limit. The remaining improving direction runs *along* that limit, shifting power between units at roughly 5 $/MWh saving, while the penalty kink is about 60× steeper. Penalty-gradient steps therefore zig-zag across the boundary and progress slowly.
* **Things that did not help R2:** larger λ, projected gradient steps in action space, and a decaying step size.

### 10. Paper protocol: 500 steps, full-episode LSTM, ramping

**Status: Run over 5 seeds. These are the reproduction results; Section 9's numbers are extended-training runs, not the paper's settings.**

Settings, all from the paper unless marked:

* 20 episodes × 25 steps = 500 updates, Adam lr 1e-3 constant, minibatch 32, λ⁽⁰⁾ = 10 (Table 1).
* The LSTM runs over the whole 25-step episode (window 25), step by step, with the previous p_g of G_ns as its extra input.
* Eq. (4) action order: p_g first, then |v|.
* Ramping (Eq. 3) on for every unit, slack included. **Our choice: R_u = R_d = 30% of P_max per step**, because the paper names ramp limits but gives no rates. Non-slack units are limited through the Eq. (18) window, the slack through 16b. There is no ramp at t = 0.
* Primary result: the last (trained) checkpoint. The best checkpoint is shown for reference only.
* Test set: 1,000 held-out states (40 episodes, seed 20000).

**OPF reference with ramping** (`utils/opf_reference.py`). Each test episode is solved as a sequence of OPFs. At t > 0 every unit's limits become the Eq. (3) window around the OPF's own previous dispatch. Results are cached per ramp fraction, episode length and slack-voltage mode. Two versions are reported:

* **Slack voltage free** (the paper's problem, since B_s is an action): 993/1,000 converged, mean 9,058 $/h.
* **Slack voltage fixed** at its setpoint (pandapower's default): 986/1,000 converged, mean 10,032 $/h.

The non-converged steps are cases where the ramp window cannot reach the net demand (checked at tighter ramps: reachable generation is below load minus wind). Passed through the ramp-aware critic, the OPF solutions reproduce their cost to 0.02 $/h, with Σz ≤ 1.1×10⁻⁶.

Results, 5 seeds, mean ± std over seeds:

| Variant | Gap vs OPF (slack V free) | Gap vs OPF (slack V fixed) | Feasible | ms/state |
|---|---|---|---|---|
| trained policy (last) | +40.7% ± 28.8 | +27.1% ± 26.2 | 91.9% ± 2.5 | 0.2 |
| + R1, K = 20 (paper, Remark 5) | +41.1% ± 28.8 | +27.4% ± 26.2 | 99.3% ± 0.4 | 5 |
| + R2, K = 5 | +32.3% ± 29.0 | +19.4% ± 26.3 | 93.8% ± 3.4 | 3 |
| + R2, K = 20 | +29.7% ± 29.4 | +17.0% ± 26.7 | 93.9% ± 3.5 | 11 |
| + R2, K = 50 | +29.5% ± 29.4 | +16.9% ± 26.8 | 97.5% ± 1.5 | 29 |
| + R2, K = 200 | +25.5% ± 27.8 | +13.3% ± 25.3 | 99.5% ± 0.3 | 123 |
| best checkpoint (reference) | +44.0% ± 26.6 | +30.0% ± 24.2 | 93.9% ± 3.0 | 0.2 |

Per seed (slack V free), trained policy → + R2 K = 200: seed 0 +28.3 → +13.3%, seed 1 +33.7 → +15.6%, seed 2 +90.3 → +74.3%, seed 3 +15.3 → +4.6%, seed 4 +36.1 → +19.7%.

Ramp binding (steps t > 0 with at least one unit within 1 MW of a ramp edge tighter than its static limit):

* OPF reference: 55% of steps (slack V free), 32% (fixed). It binds mostly on the two small cheap units at bus 0, gen0 (12 MW/step) and sgen0 (51 MW/step).
* Trained policy: 3.5% of steps.

Findings:

* **The paper's near-optimality within 500 steps is not reproduced.** Under the paper's own protocol the trained policy is +41% above the matched OPF, against +3.5% after 5,000 steps without ramping (Section 9).
* **Seed variance is very large** (std 29 points). Seed 2 stalls at +90%; seed 3 reaches +15%.
* **R1 reproduces Remark 5**: feasibility rises from 92% to 99.3% at essentially unchanged cost.
* **R2 helps but cannot rescue an under-trained policy**: −15 points at K = 200, at about 120 ms per state.
* **The policy rarely uses the ramp limits the OPF uses** (3.5% against 55% of steps). Cheap units are not pushed to their ramp edges, which accounts for part of the gap.
* The fixed-slack-voltage OPF is a weaker optimum (+11% cost), so gaps against it look smaller. It is shown for comparison only; the slack-voltage-free OPF is the paper's problem.

Reproducing the earlier runs: `--window 1 --ramp-frac 0 --voltage-first --episodes 200` (see the docstrings of `scripts/train_dhelm_case5.py` and `scripts/run_seeds.py`).

### Departures and choices (summary)

Choices where the paper names something without a number, or is unreadable:

* Ramp rate 30% of P_max per step (Eq. 3).
* Episode length 25, from "20 episodes (≤ 500 steps)".
* Policy input "+4" = the previous p_g of G_ns (Table 1 symbol unreadable).
* η_n = 1/(x̄ − x̲), w_r = 1/f_max, η_λ = 1, λ_max = 1,000 (Section 7–8).
* Wind buses 1 and 4 (Section 5).
* Action order inside the p_g and |v| blocks: by bus, `gen` before `sgen`.

Matching the OPF reference to the paper's problem (not a change to the method): unrated lines unconstrained, slack voltage free, sequential OPF with the same ramp windows.

Extensions, reported separately from the reproduction: R2 refinement, longer training (2,500 / 5,000 steps), cosine LR, best-checkpoint selection, single-step windows without ramping.

### 11. Remaining work

* A constrained refinement using the critic's exact Jacobians (SQP or augmented Lagrangian) to remove R2's boundary crawl.
* Training-side improvements toward the paper's 500-step claim (Section 10 shows the gap and the large seed variance), such as learning rate and network size. Since the optimum often lies at action bounds, the tanh parametrization (Eq. 18) is worth examining.
* The physics-informed extensions planned for this project.
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
* The multi-seed evaluation, and the R1 (paper, Remark 5) and R2 (extension) refinements (Section 9).

Results, over 5 seeds against the matched OPF:

* **Reproduction (paper protocol, Section 10):** 500 steps, full-episode LSTM, ramping at 30% of P_max. The trained policy is +40.7% ± 28.8 above the matched OPF; + R1 it is 99.3% feasible at +41.1%. The paper's near-optimality within 500 steps is not reproduced.
* **Improvement (R2, K = 200) under the paper protocol:** +25.5% ± 27.8 at 99.5% feasibility.
* **Extended training (Section 9, not the paper's settings):** 5,000 steps without ramping, +3.5% (R1) and +1.8% ± 0.1 (R2, K = 200).

Next targets are listed in Section 11.
