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

**Status: NumPy reference solver complete and validated. Differentiable (PyTorch) version not yet started.**

Implemented following the paper's equations:

* No-load germ V₀ = 1, W₀ = 1, Q₀ = 0 (paper, after Eq. 11).
* Inverse-voltage recursion, Eq. (11).
* Coefficient matrix A with the Y^tr / Y^sh split, Eq. (12).
* R_n and U_n vectors for PQ, PV and slack buses, Eq. (13).
* Termination on the Eq. (14a)–(14c) mismatches, with Algorithm 1's settings ε = 10⁻⁵ and n_max = 30.

Validation (`scripts/test_helm_random_vs_pandapower.py`): 200 random scenarios varying the slack and PV setpoints (0.95–1.05 p.u.), demand and wind.

* With a tight tolerance, the solver matches pandapower to 7×10⁻¹¹.
* With the paper's tolerance, 200/200 scenarios converge using 4–6 terms, with a maximum voltage error of 9×10⁻⁷.

This NumPy solver is the reference implementation. The differentiable D-HELM critic will be validated against it.

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

### 7. Physics-informed DRL

### 6. Physics-informed DRL

**Status: Not yet implemented**

The following components remain to be implemented and validated:

* Differentiable (PyTorch, batched) D-HELM solver.
* Physics-derived critic: Eqs. (15)–(16), violations (6), reward (5).
* Policy network.
* Reinforcement-learning environment/interface.
* Policy optimization algorithm.
* Training procedure.
* Baselines.
* Experimental evaluation.

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

The next major implementation target is the differentiable D-HELM solver. It will be validated against the NumPy solver, along with a gradient check, before it is connected to the critic and the policy.
