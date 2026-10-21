"""
Decode the normalized policy action into physical OPF setpoints.

Paper basis
-----------
Table 1: policy output = |B_v ∪ B_s| + |G / G_s|  (4 + 4 = 8 for 5-bus).

The bounds are taken from the ActionConfig, which is derived from the
network data (see action_config.action_config_from_network), so no
limits are supplied or invented here.

Normalized action convention
----------------------------
Each policy output is in [-1, 1] and is mapped linearly to its physical
range:

    voltage   : [-1, 1] -> [V_min, V_max]   (p.u.)
    generator : [-1, 1] -> [P_min, P_max]   (MW)

The policy's tanh / bound enforcement will be connected later.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandapower as pp

from .action_config import ActionConfig


@dataclass(frozen=True)
class DecodedAction:
    """
    Physical setpoints for one policy action.

    voltage_setpoints[k]  belongs to config.voltage_buses[k]   (p.u.)
    generator_p_mw[k]     belongs to config.generators[k]      (MW)
    """

    voltage_setpoints: np.ndarray
    generator_p_mw: np.ndarray
    base_mva: float

    @property
    def generator_p_pu(self) -> np.ndarray:
        return self.generator_p_mw / self.base_mva

    @property
    def action_dim(self) -> int:
        return len(self.voltage_setpoints) + len(self.generator_p_mw)


class ActionDecoder:
    """Convert normalized policy outputs into physical OPF setpoints."""

    def __init__(self, config: ActionConfig) -> None:

        self.config = config

        self._v_min = np.asarray(config.voltage_min, dtype=float)
        self._v_max = np.asarray(config.voltage_max, dtype=float)

        self._p_min = config.generator_p_min_mw
        self._p_max = config.generator_p_max_mw

    def _validate_normalized_action(
        self,
        action: Sequence[float],
    ) -> np.ndarray:

        values = np.asarray(action, dtype=float)

        if values.shape != (self.config.action_dim,):
            raise ValueError(
                f"Expected an action of shape ({self.config.action_dim},), "
                f"received {values.shape}."
            )

        if not np.all(np.isfinite(values)):
            raise ValueError("Action contains NaN or infinite values.")

        if np.any(values < -1.0) or np.any(values > 1.0):
            raise ValueError(
                "Normalized action values must lie in [-1, 1]."
            )

        return values

    @staticmethod
    def _scale(value, lower, upper):
        """Map [-1, 1] to [lower, upper] (element-wise)."""

        return lower + 0.5 * (value + 1.0) * (upper - lower)

    def decode(self, action: Sequence[float]) -> DecodedAction:

        action = self._validate_normalized_action(action)

        v_part = action[list(self.config.voltage_action_indices)]
        p_part = action[list(self.config.generator_action_indices)]

        return DecodedAction(
            voltage_setpoints=self._scale(v_part, self._v_min, self._v_max),
            generator_p_mw=self._scale(p_part, self._p_min, self._p_max),
            base_mva=self.config.base_mva,
        )


def apply_action_to_network(
    net: pp.pandapowerNet,
    config: ActionConfig,
    decoded: DecodedAction,
) -> None:
    """
    Write decoded setpoints into the pandapower network.

    Voltage setpoints:
        slack bus -> ext_grid.vm_pu
        PV bus    -> vm_pu of every in-service gen on that bus

    Generator setpoints:
        p_mw of the corresponding gen / sgen row

    Used for validation against pandapower and to build HelmProblem
    instances.  The training critic will take these values directly as
    tensors instead of going through pandapower.
    """

    for bus, vm in zip(config.voltage_buses, decoded.voltage_setpoints):

        if bus == config.slack.bus:
            net.ext_grid.at[config.slack.index, "vm_pu"] = float(vm)
            continue

        on_bus = net.gen.index[
            (net.gen["bus"] == bus) & net.gen["in_service"]
        ]

        if len(on_bus) == 0:
            raise ValueError(
                f"No in-service gen regulates voltage bus {bus}."
            )

        net.gen.loc[on_bus, "vm_pu"] = float(vm)

    for unit, p_mw in zip(config.generators, decoded.generator_p_mw):
        net[unit.table].at[unit.index, "p_mw"] = float(p_mw)
