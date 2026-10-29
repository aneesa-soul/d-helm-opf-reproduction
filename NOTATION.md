# Notation: paper symbols and code names

The code follows the notation of Sayed et al., "Efficient optimal power flow learning: A deep reinforcement learning with physics-driven critic model" (IJEPES 167, 2025). The table lists each paper symbol with the Python name used for it.

Python names cannot carry sub/superscripts, bars or calligraphic letters, so the spelling rules are:

* A subscript becomes `_x`.
* A superscript `sp`, `sh`, `tr`, `u`, `d` or `w` becomes `_sp`, `_sh`, `_tr` and so on.
* An upper bar (maximum) becomes `_max`, and a lower bar (minimum) becomes `_min`.
* Real and imaginary parts get `_re` and `_im` suffixes.
* λ is spelled `lam`, because `lambda` is reserved in Python.

Names with no paper symbol (pandapower table rows, incidence matrices, network layers) stay descriptive and are marked "—".

## Sets and indices

| Paper | Meaning | Code |
|---|---|---|
| 𝓑 | all buses | `range(n_bus)` |
| 𝓑_s | slack bus | `B_s` (bus index) |
| 𝓑_v | PV buses | `B_v` |
| 𝓑_q | PQ buses | `B_q` |
| 𝓑_v ∪ 𝓑_s | voltage-controlled buses (voltage actions, Eq. 4) | `B_vs` |
| 𝓑_d, 𝓑_w | demand and wind buses (Eq. 2) | `B_d`, `B_w` |
| 𝓖 / 𝓖_s | non-slack generating units (P actions, Eq. 4) | `G_ns` |
| 𝓖_s | slack unit | `G_s` |
| n | series order (Eqs. 10–13) | `n` |
| N, n_max | terms used / maximum terms (Eq. 17, Algorithm 1) | `N`, `n_max` |
| t | time step | `t` |

## MDP (Eqs. 2–7)

| Paper | Meaning | Code |
|---|---|---|
| s_t (Eq. 2) | state: p^d, q^d, p^w | `s` |
| a_t (Eq. 4) | physical action: p_g (𝓖/𝓖_s), \|v_i\| (𝓑_v ∪ 𝓑_s) | `a` |
| tanh(logit) (Eq. 18) | normalized action in [−1, 1] produced by the policy | `a_hat` |
| logit (Eq. 18) | pre-tanh policy output | `logit` |
| a^min, a^max (Eq. 18) | action bounds | `a_min`, `a_max` |
| p^d_i, q^d_i | active / reactive demand | `p_d`, `q_d` |
| p^w_i | wind output | `p_w` |
| p_g, q_g | unit active / reactive power | `p_g`, `q_g` |
| P̲_g, P̄_g | unit P limits | `P_min`, `P_max` |
| R^u_g, R^d_g (Eq. 3) | ramp-up / ramp-down limits | `R_u`, `R_d` |
| p_{g,t−1}, p*_g (Eqs. 3, 16b) | previous-step output | `p_g_prev` |
| r_t (Eq. 5) | reward | `r` |
| w_0, w_r (Eq. 5) | reward weights | `w_0`, `w_r` |
| Σ_g f_g(p_g) (Eq. 5) | total generation cost | `f_sum` |
| z_{n,t} (Eq. 6) | constraint violations | `z` |
| η_n (Eq. 6) | violation normalizer | `eta` |
| x_n, x̲_n, x̄_n (Eq. 6) | bounded quantity and its limits | `x`, `x_min`, `x_max` |

## Power flow and HELM (Eqs. 8–14, Algorithm 1)

| Paper | Meaning | Code |
|---|---|---|
| v_i = \|v_i\|∠θ_i (Eq. 8) | complex bus voltage | `v` (NumPy) / `v_re`, `v_im` (torch) |
| \|v_i\|, θ_i | magnitude, angle | `v_abs`, `theta` |
| \|v_i\|^sp | voltage setpoint (PV, slack) | `v_sp` |
| e_i = p_i + j q_i (Eq. 8) | complex power injection computed from v | `e` / `e_re`, `e_im` |
| e_i^sp, p_i, q_i (Eq. 8, 14) | specified net injection | `e_sp` (NumPy) / `p`, `q` (torch) |
| P_i (Eq. 13) | specified P at a PV bus | `P` |
| Y_ij | bus admittance matrix | `Y` / `Y_re`, `Y_im` |
| Y^tr, Y^sh (Eq. 9) | transmission part, shunt part | `Y_tr`, `Y_sh` |
| α (Eq. 9) | embedding parameter (α = 0 no load, α = 1 full load) | `alpha` |
| V_{i,n}, W_{i,n}, Q_{i,n} (Eqs. 10–11) | series coefficients of v, 1/v*, q | `V`, `W`, `Q` |
| A, x_n, b_n (Eq. 12) | linear system per order | `A`, `x_n`, `b_n` |
| B (Eq. 12) | PV-bus selection matrix | `B` |
| R_n, U_n (Eq. 13) | right-hand-side vectors | `R_n`, `U_n` |
| ϵ, ε (Eq. 14) | voltage / power tolerances | `epsilon` (ϵ = ε = 10⁻⁵, Table 1) |

## Critic (Eqs. 15–17)

| Paper | Meaning | Code |
|---|---|---|
| p_g, i ∈ 𝓑_s (Eq. 15) | slack active power | `p_g_s` |
| v̲_i, v̄_i (Eq. 16a) | PQ voltage limits | `v_min`, `v_max` |
| q̲_g, q̄_g (Eq. 16c) | unit Q limits (per bus) | `q_g_min`, `q_g_max` |
| i_f, i_t (Eqs. 16d–e) | sending / receiving line currents | `i_f`, `i_t` |
| Y_f, Y_t | branch admittance matrices | `Y_f_re`, `Y_f_im`, `Y_t_re`, `Y_t_im` |
| I_base | base current | `I_base_f`, `I_base_t` |
| ī | line current limit | `i_max` |

## Constrained policy optimization (Eqs. 19–20, Table 1)

| Paper | Meaning | Code |
|---|---|---|
| λ_n | Lagrange multipliers | `lam` |
| λ^(0) | initial multiplier (Table 1: 10) | `lam_0` |
| η_λ | multiplier step size | `eta_lam` |
| λ^max | multiplier cap | `lam_max` |
| Γ_λ[·] | projection onto [0, λ^max] | `torch.clamp` |

## No paper symbol (descriptive names kept)

| Code | Meaning |
|---|---|
| `C_g`, `C_w`, `C_d` | incidence matrices mapping units / wind / loads onto buses |
| `cp0`, `cp1`, `cp2` | pandapower `poly_cost` coefficients of f_g |
| `lstm`, `fc1`, `fc2`, `out` | policy layers (Table 1) |
| `n_v`, `n_g` | number of voltage / P actions |
