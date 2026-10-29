"""
Action configuration for the D-HELM OPF reproduction.

Paper basis
-----------
Table 1 defines the policy output dimension as

    |B_v ∪ B_s| + |G / G_s|

i.e. one voltage setpoint per voltage-controlled bus (PV buses B_v and
the slack bus B_s) plus one active-power setpoint per non-slack
generator.

How the mapping is obtained
---------------------------
The mapping is DERIVED from the paper's Table 1 definition and the case
data; it is not a free assumption.  `action_config_from_network()`
reads it from the pandapower network:

    B_vs = B_v ∪ B_s : buses of in-service `gen` units  ∪  ext_grid bus
    G_ns = G / G_s   : in-service `gen` units and controllable, non-wind
                       `sgen` units (the slack unit G_s is excluded)
    limits        : bus min/max_vm_pu, unit min/max_p_mw, min/max_q_mvar
    costs         : net.poly_cost

For the IEEE 5-bus case this gives 4 + 4 = 8, matching Table 1:

    B_vs : (0, 2, 3, 4)            B_v = {0, 2, 4},  B_s = {3}
    G_ns : gen0@0, sgen0@0, gen1@2, gen2@4

Wind units are tagged with sgen.type == WIND_SGEN_TYPE (see
environment/uncertainty.py) and are never treated as controllable
generators.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandapower as pp

from dhelm_opf.environment.uncertainty import WIND_SGEN_TYPE


@dataclass(frozen=True)
class GeneratorUnit:
    """
    One controllable generating unit, identified by its pandapower table.

    P_min, P_max : active-power limits  (P̲_g, P̄_g), MW
    q_min, q_max : reactive limits      (q̲_g, q̄_g), MVAr

    Cost f_g(p_g) (Eq. 5) follows pandapower's poly_cost:

        f_g = cp0 + cp1 * p_g + cp2 * p_g**2
    """

    table: str          # "gen", "sgen" or "ext_grid"
    index: int          # row index in that table
    bus: int

    P_min: float
    P_max: float
    q_min: float
    q_max: float

    cp0: float
    cp1: float
    cp2: float

    @property
    def label(self) -> str:
        return f"{self.table}{self.index}@bus{self.bus}"


@dataclass(frozen=True)
class ActionConfig:
    """
    Layout and physical bounds of the policy action.

    Action layout (p_first = True, the order of Eq. 4)
    -------------------------------------------------
    a[0 : n_g]          p_g,   g in G_ns = G / G_s     (bounds P_min, P_max)
    a[n_g : n_g + n_v]  |v_i|, i in B_vs = B_v ∪ B_s   (bounds v_min, v_max)

    p_first = False gives the voltage-first layout used before the
    paper-protocol update; it is kept so earlier checkpoints decode
    correctly.

    The slack unit G_s is not an action; its p_g and q_g are outputs
    of the critic (paper Eqs. 15-16).  Its limits and cost are kept
    here because the critic needs them.
    """

    B_vs: Tuple[int, ...]
    v_min: Tuple[float, ...]
    v_max: Tuple[float, ...]

    G_ns: Tuple[GeneratorUnit, ...]
    G_s: GeneratorUnit

    base_mva: float

    # Eq. (4) order: p_g first, then |v_i|.
    p_first: bool = True

    def __post_init__(self) -> None:

        n_v = len(self.B_vs)

        if not (n_v == len(self.v_min) == len(self.v_max)):
            raise ValueError(
                "B_vs, v_min and v_max must "
                "have equal length."
            )

        for bus, vmin, vmax in zip(
            self.B_vs, self.v_min, self.v_max
        ):
            if vmin >= vmax:
                raise ValueError(
                    f"Invalid voltage bounds at bus {bus}: "
                    f"[{vmin}, {vmax}]."
                )

        for unit in (*self.G_ns, self.G_s):
            if unit.P_min > unit.P_max:
                raise ValueError(f"Pmin > Pmax for {unit.label}.")
            if unit.q_min > unit.q_max:
                raise ValueError(f"Qmin > Qmax for {unit.label}.")

        if self.G_s.bus not in self.B_vs:
            raise ValueError(
                "The slack bus must be a voltage-controlled bus."
            )

        if self.base_mva <= 0:
            raise ValueError("base_mva must be positive.")

    # ------------------------------------------------------------------
    # Dimensions and index helpers
    # ------------------------------------------------------------------

    @property
    def n_v(self) -> int:
        return len(self.B_vs)

    @property
    def n_g(self) -> int:
        return len(self.G_ns)

    @property
    def action_dim(self) -> int:
        """|B_v ∪ B_s| + |G / G_s| (paper, Table 1)."""
        return self.n_v + self.n_g

    @property
    def voltage_action_indices(self) -> Tuple[int, ...]:
        """Positions of the |v_i| actions in a."""
        start = self.n_g if self.p_first else 0
        return tuple(range(start, start + self.n_v))

    @property
    def generator_action_indices(self) -> Tuple[int, ...]:
        """Positions of the p_g actions in a."""
        start = 0 if self.p_first else self.n_v
        return tuple(range(start, start + self.n_g))

    # ------------------------------------------------------------------
    # Bound vectors (convenient for the decoder / critic)
    # ------------------------------------------------------------------

    @property
    def P_min(self) -> np.ndarray:
        """P̲_g of the non-slack units G_ns (MW)."""
        return np.array([u.P_min for u in self.G_ns])

    @property
    def P_max(self) -> np.ndarray:
        """P̄_g of the non-slack units G_ns (MW)."""
        return np.array([u.P_max for u in self.G_ns])

    def describe(self) -> str:
        """Human-readable description of the action layout."""

        lines = [
            "D-HELM OPF action configuration (derived from network)",
            "",
            f"Action dimension : {self.n_g} p_g "
            f"+ {self.n_v} |v| = {self.action_dim}  "
            f"({'Eq. 4 order: p_g first' if self.p_first else 'voltage first'})",
            "",
        ]

        v_lines = ["|v_i| setpoints (B_v ∪ B_s):"]
        for k, bus, vmin, vmax in zip(
            self.voltage_action_indices, self.B_vs, self.v_min, self.v_max
        ):
            kind = "slack" if bus == self.G_s.bus else "PV"
            v_lines.append(
                f"  a[{k}]  bus {bus} ({kind:5s})  "
                f"[{vmin:.2f}, {vmax:.2f}] p.u."
            )

        p_lines = ["p_g setpoints (G / G_s):"]
        for k, unit in zip(self.generator_action_indices, self.G_ns):
            p_lines.append(
                f"  a[{k}]  {unit.label:15s}  "
                f"[{unit.P_min:.1f}, {unit.P_max:.1f}] MW  "
                f"cost {unit.cp1:.1f} $/MWh"
            )

        blocks = (p_lines, v_lines) if self.p_first else (v_lines, p_lines)
        lines += blocks[0] + [""] + blocks[1]

        lines.append("")
        lines.append(
            f"Slack (critic output, not an action): {self.G_s.label}  "
            f"[{self.G_s.P_min:.1f}, {self.G_s.P_max:.1f}] MW  "
            f"cost {self.G_s.cp1:.1f} $/MWh"
        )

        return "\n".join(lines)


@dataclass(frozen=True)
class RampConfig:
    """
    Ramp limits R_u, R_d of Eq. (3) / (16b), per step.

    The paper includes ramping (Eqs. 3 and 16b) but gives no ramp rates
    for the test systems, so the rates are a reproduction choice:
    R_u = frac_u * P_max and R_d = frac_d * P_max for every unit, the
    slack included.  Default 30% of P_max per step (see
    REPRODUCTION_STATUS.md for the reasoning).
    """

    frac_u: float = 0.30
    frac_d: float = 0.30

    def __post_init__(self) -> None:
        if not (0.0 < self.frac_u and 0.0 < self.frac_d):
            raise ValueError("Ramp fractions must be positive.")

    def R_u(self, unit: "GeneratorUnit") -> float:
        """R^u_g in MW per step."""
        return self.frac_u * unit.P_max

    def R_d(self, unit: "GeneratorUnit") -> float:
        """R^d_g in MW per step."""
        return self.frac_d * unit.P_max


# ----------------------------------------------------------------------
# Construction from a pandapower network
# ----------------------------------------------------------------------

def _required(row, column: str, label: str) -> float:
    """Read a numeric field, refusing to invent a missing value."""

    if column not in row.index or not np.isfinite(float(row[column])):
        raise ValueError(
            f"{label}: '{column}' is missing from the network data. "
            "Supply it in the case definition rather than guessing it."
        )

    return float(row[column])


def _poly_cost(net: pp.pandapowerNet, table: str, index: int):
    """Return (cp0, cp1, cp2) for one unit from net.poly_cost."""

    rows = net.poly_cost[
        (net.poly_cost["et"] == table)
        & (net.poly_cost["element"] == index)
    ]

    if len(rows) != 1:
        raise ValueError(
            f"Expected exactly one poly_cost entry for {table}{index}, "
            f"found {len(rows)}."
        )

    row = rows.iloc[0]

    return (
        float(row["cp0_eur"]),
        float(row["cp1_eur_per_mw"]),
        float(row["cp2_eur_per_mw2"]),
    )


def _make_unit(net, table: str, index: int) -> GeneratorUnit:

    row = net[table].loc[index]
    label = f"{table}{index}"

    cp0, cp1, cp2 = _poly_cost(net, table, index)

    return GeneratorUnit(
        table=table,
        index=int(index),
        bus=int(row["bus"]),
        P_min=_required(row, "min_p_mw", label),
        P_max=_required(row, "max_p_mw", label),
        q_min=_required(row, "min_q_mvar", label),
        q_max=_required(row, "max_q_mvar", label),
        cp0=cp0,
        cp1=cp1,
        cp2=cp2,
    )


def is_wind_sgen(net: pp.pandapowerNet, index: int) -> bool:
    """True if sgen `index` is a wind unit (tagged by type)."""

    return str(net.sgen.at[index, "type"]) == WIND_SGEN_TYPE


def action_config_from_network(
    net: pp.pandapowerNet,
    p_first: bool = True,
) -> ActionConfig:
    """
    Derive the action layout, limits and costs from the network.

    p_first=True uses the Eq. (4) order (p_g, then |v_i|).

    Generators are ordered by bus, with `gen` units before `sgen`
    units on the same bus.
    """

    if len(net.ext_grid) != 1:
        raise ValueError(
            "Exactly one ext_grid (slack) is expected."
        )

    slack_index = int(net.ext_grid.index[0])
    slack = _make_unit(net, "ext_grid", slack_index)

    # ------------------------------------------------------------
    # Controllable non-slack generators: G / G_s
    # ------------------------------------------------------------
    units = []

    for index, row in net.gen.iterrows():
        if bool(row["in_service"]):
            units.append(_make_unit(net, "gen", index))

    for index, row in net.sgen.iterrows():
        if (
            bool(row["in_service"])
            and bool(row.get("controllable", False))
            and not is_wind_sgen(net, index)
        ):
            units.append(_make_unit(net, "sgen", index))

    table_order = {"gen": 0, "sgen": 1}

    units.sort(key=lambda u: (u.bus, table_order[u.table], u.index))

    # ------------------------------------------------------------
    # Voltage-controlled buses: B_v ∪ B_s
    #
    # Only `gen` units regulate voltage in pandapower; an sgen is a
    # PQ injection, so it does not add a voltage-controlled bus.
    # ------------------------------------------------------------
    B_vs = sorted(
        {
            int(row["bus"])
            for _, row in net.gen.iterrows()
            if bool(row["in_service"])
        }
        | {slack.bus}
    )

    v_min = tuple(
        _required(net.bus.loc[bus], "min_vm_pu", f"bus{bus}")
        for bus in B_vs
    )
    v_max = tuple(
        _required(net.bus.loc[bus], "max_vm_pu", f"bus{bus}")
        for bus in B_vs
    )

    return ActionConfig(
        B_vs=tuple(B_vs),
        v_min=v_min,
        v_max=v_max,
        G_ns=tuple(units),
        G_s=slack,
        base_mva=float(net.sn_mva),
        p_first=p_first,
    )
