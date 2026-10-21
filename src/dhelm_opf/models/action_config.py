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

    voltage buses : buses of in-service `gen` units  ∪  ext_grid bus
    generators    : in-service `gen` units and controllable, non-wind
                    `sgen` units (the slack unit is excluded)
    limits        : bus min/max_vm_pu, unit min/max_p_mw, min/max_q_mvar
    costs         : net.poly_cost

For the IEEE 5-bus case this gives 4 + 4 = 8, matching Table 1:

    voltage buses : (0, 2, 3, 4)            PV {0, 2, 4} + slack {3}
    generators    : gen0@0, sgen0@0, gen1@2, gen2@4

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

    Power limits are in MW / MVAr.  Cost follows pandapower's poly_cost:

        cost = cp0 + cp1 * p_mw + cp2 * p_mw**2
    """

    table: str          # "gen", "sgen" or "ext_grid"
    index: int          # row index in that table
    bus: int

    p_min_mw: float
    p_max_mw: float
    q_min_mvar: float
    q_max_mvar: float

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

    Action layout
    -------------
    action[0 : n_v]          voltage setpoints of `voltage_buses`
    action[n_v : n_v + n_g]  active-power setpoints of `generators`

    The slack unit is not an action; its P and Q are outputs of the
    critic (paper Eqs. 15-16).  Its limits and cost are kept here
    because the critic needs them.
    """

    voltage_buses: Tuple[int, ...]
    voltage_min: Tuple[float, ...]
    voltage_max: Tuple[float, ...]

    generators: Tuple[GeneratorUnit, ...]
    slack: GeneratorUnit

    base_mva: float

    def __post_init__(self) -> None:

        n_v = len(self.voltage_buses)

        if not (n_v == len(self.voltage_min) == len(self.voltage_max)):
            raise ValueError(
                "voltage_buses, voltage_min and voltage_max must "
                "have equal length."
            )

        for bus, vmin, vmax in zip(
            self.voltage_buses, self.voltage_min, self.voltage_max
        ):
            if vmin >= vmax:
                raise ValueError(
                    f"Invalid voltage bounds at bus {bus}: "
                    f"[{vmin}, {vmax}]."
                )

        for unit in (*self.generators, self.slack):
            if unit.p_min_mw > unit.p_max_mw:
                raise ValueError(f"Pmin > Pmax for {unit.label}.")
            if unit.q_min_mvar > unit.q_max_mvar:
                raise ValueError(f"Qmin > Qmax for {unit.label}.")

        if self.slack.bus not in self.voltage_buses:
            raise ValueError(
                "The slack bus must be a voltage-controlled bus."
            )

        if self.base_mva <= 0:
            raise ValueError("base_mva must be positive.")

    # ------------------------------------------------------------------
    # Dimensions and index helpers
    # ------------------------------------------------------------------

    @property
    def n_voltage(self) -> int:
        return len(self.voltage_buses)

    @property
    def n_generator(self) -> int:
        return len(self.generators)

    @property
    def action_dim(self) -> int:
        """|B_v ∪ B_s| + |G / G_s| (paper, Table 1)."""
        return self.n_voltage + self.n_generator

    @property
    def voltage_action_indices(self) -> Tuple[int, ...]:
        return tuple(range(self.n_voltage))

    @property
    def generator_action_indices(self) -> Tuple[int, ...]:
        return tuple(range(self.n_voltage, self.action_dim))

    # ------------------------------------------------------------------
    # Bound vectors (convenient for the decoder / critic)
    # ------------------------------------------------------------------

    @property
    def generator_p_min_mw(self) -> np.ndarray:
        return np.array([u.p_min_mw for u in self.generators])

    @property
    def generator_p_max_mw(self) -> np.ndarray:
        return np.array([u.p_max_mw for u in self.generators])

    def describe(self) -> str:
        """Human-readable description of the action layout."""

        lines = [
            "D-HELM OPF action configuration (derived from network)",
            "",
            f"Action dimension : {self.n_voltage} voltage "
            f"+ {self.n_generator} generator P = {self.action_dim}",
            "",
            "Voltage setpoints:",
        ]

        for k, (bus, vmin, vmax) in enumerate(
            zip(self.voltage_buses, self.voltage_min, self.voltage_max)
        ):
            kind = "slack" if bus == self.slack.bus else "PV"
            lines.append(
                f"  a[{k}]  bus {bus} ({kind:5s})  "
                f"[{vmin:.2f}, {vmax:.2f}] p.u."
            )

        lines.append("")
        lines.append("Generator active-power setpoints:")

        for k, unit in zip(self.generator_action_indices, self.generators):
            lines.append(
                f"  a[{k}]  {unit.label:15s}  "
                f"[{unit.p_min_mw:.1f}, {unit.p_max_mw:.1f}] MW  "
                f"cost {unit.cp1:.1f} $/MWh"
            )

        lines.append("")
        lines.append(
            f"Slack (critic output, not an action): {self.slack.label}  "
            f"[{self.slack.p_min_mw:.1f}, {self.slack.p_max_mw:.1f}] MW  "
            f"cost {self.slack.cp1:.1f} $/MWh"
        )

        return "\n".join(lines)


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
        p_min_mw=_required(row, "min_p_mw", label),
        p_max_mw=_required(row, "max_p_mw", label),
        q_min_mvar=_required(row, "min_q_mvar", label),
        q_max_mvar=_required(row, "max_q_mvar", label),
        cp0=cp0,
        cp1=cp1,
        cp2=cp2,
    )


def is_wind_sgen(net: pp.pandapowerNet, index: int) -> bool:
    """True if sgen `index` is a wind unit (tagged by type)."""

    return str(net.sgen.at[index, "type"]) == WIND_SGEN_TYPE


def action_config_from_network(net: pp.pandapowerNet) -> ActionConfig:
    """
    Derive the action layout, limits and costs from the network.

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
    voltage_buses = sorted(
        {
            int(row["bus"])
            for _, row in net.gen.iterrows()
            if bool(row["in_service"])
        }
        | {slack.bus}
    )

    voltage_min = tuple(
        _required(net.bus.loc[bus], "min_vm_pu", f"bus{bus}")
        for bus in voltage_buses
    )
    voltage_max = tuple(
        _required(net.bus.loc[bus], "max_vm_pu", f"bus{bus}")
        for bus in voltage_buses
    )

    return ActionConfig(
        voltage_buses=tuple(voltage_buses),
        voltage_min=voltage_min,
        voltage_max=voltage_max,
        generators=tuple(units),
        slack=slack,
        base_mva=float(net.sn_mva),
    )
