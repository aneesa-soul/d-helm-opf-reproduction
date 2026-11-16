"""
Test systems of the paper (Sec. 4.1, Table 1) and their configuration.

Each CaseSpec holds what differs between systems: the pandapower
network, the wind farms, and the Table 1 learning parameters.

Paper basis
-----------
* Networks: "available in the dataset of pandapower v.2.13.1".
  The 30-bus system is pandapower's `case30` (MATPOWER case30): Table 1
  gives 20 x 2 demand uncertainties and a HELM output of
  2(|B| + |G| + |L|) = 2(30 + 6 + 41) = 154, which matches case30
  (20 loads, 6 units, 41 lines).  `case_ieee30` has 21 loads.
* Wind: |W| = 2 / 4 farms for the 5- / 30-bus systems (Table 1), output
  in [40%, 100%] of rated.  Their buses and ratings are in the paper's
  electronic companion [38], which is not available online.
* Table 1: minibatch / lr and policy layers (64L/64/64 for 5- and
  30-bus; 128L/128/128 and 128 / 5e-4 for 118-bus; 256L/256/256 and
  128 / 5e-4 for 300-bus).  Sec. 4.2: Table 2 agents trained for
  1000 / 2000 / 3000 / 8000 episodes.
* 118-bus = pandapower `case118`: Table 1's 99 x 2 + 7 uncertainties,
  54 + 53 actions and HELM output 2(118 + 54 + 173) = 690 (lines only).
* 300-bus = pandapower `case300`: 193 x 2 + 10 uncertainties, 69 + 68
  actions; HELM output 2(300 + 77 + 283) = 1320 counts the 8 fixed
  (non-controllable) sgens as units and the lines only.
* Policy input: Table 1 gives +4 / +5 for 5- / 30-bus (|G / G_s|) but
  +54 / +69 for 118- / 300-bus (|G|).  Followed row by row: for 118 and
  300 the previous slack output is an extra input (prev_slack).

Our choices (not in the paper)
------------------------------
* Wind placement rule, applied to both systems: the farms sit at
  load buses, with total rated wind = 20% of the nominal total active
  load, split equally.
    - 5-bus : buses 1 and 4 (chosen earlier), 2 x 100 MW (1,000 MW load).
    - 30-bus: the four largest-load PQ buses, 0-based 6, 7, 11, 20
      (MATPOWER buses 7, 8, 12, 21), 4 x 9.46 MW (189.2 MW load).
    - 118-bus: the seven largest-load PQ buses, 0-based
      10, 44, 59, 74, 77, 81, 87, 7 x 121.2 MW (4,242 MW load).
    - 300-bus: the ten largest-load PQ buses, 0-based
      16, 98, 99, 117, 156, 170, 201, 203, 210, 212, 10 x 477.0 MW
      (23,848 MW load).  NOTE: with the paper's HELM settings
      (n_max = 30) the 300-bus power flow does not converge for most
      sampled operating points; see REPRODUCTION_STATUS.md.
  Placing farms at load centres and choosing a moderate penetration is
  common practice when a test case has no renewable data.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandapower as pp
import pandapower.networks as pn


@dataclass(frozen=True)
class CaseSpec:
    name: str
    B_w: tuple[int, ...]              # wind buses (0-based pandapower index)
    p_w_rated_mw: tuple[float, ...]   # rated wind output per farm, MW
    hidden: int                       # Table 1: policy layers hidden/hidden/hidden
    batch_size: int                   # Table 1: minibatch
    lr: float                         # Table 1: learning rate
    table2_episodes: int              # Sec. 4.2: training episodes for Table 2
    prev_slack: bool = False          # Table 1 policy input +|G| (118/300)

    def build(self) -> pp.pandapowerNet:
        return getattr(pn, self.name)()


CASES: dict[str, CaseSpec] = {
    "case5": CaseSpec(
        name="case5",
        B_w=(1, 4),
        p_w_rated_mw=(100.0, 100.0),
        hidden=64, batch_size=32, lr=1e-3,
        table2_episodes=1000,
    ),
    "case30": CaseSpec(
        name="case30",
        B_w=(6, 7, 11, 20),
        p_w_rated_mw=(9.46, 9.46, 9.46, 9.46),
        hidden=64, batch_size=64, lr=1e-3,
        table2_episodes=2000,
    ),
    "case118": CaseSpec(
        name="case118",
        B_w=(10, 44, 59, 74, 77, 81, 87),
        p_w_rated_mw=(121.2,) * 7,
        hidden=128, batch_size=128, lr=5e-4,
        table2_episodes=3000,
        prev_slack=True,
    ),
    "case300": CaseSpec(
        name="case300",
        B_w=(16, 98, 99, 117, 156, 170, 201, 203, 210, 212),
        p_w_rated_mw=(477.0,) * 10,
        hidden=256, batch_size=128, lr=5e-4,
        table2_episodes=8000,
        prev_slack=True,
    ),
}


def get_case(name: str) -> CaseSpec:
    try:
        return CASES[name]
    except KeyError:
        raise ValueError(f"Unknown case {name!r}; known: {sorted(CASES)}") from None
