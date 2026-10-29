"""
Decode the normalized policy action a_hat into the physical action a.

Paper basis
-----------
Eq. (4):   a_t = (p_g, g in G / G_s;  |v_i|, i in B_v ∪ B_s)
Eq. (18):  a = a_min + (a_max - a_min) (tanh(logit) + 1) / 2
Table 1:   |B_v ∪ B_s| + |G / G_s| = 4 + 4 = 8 for 5-bus.

The policy outputs a_hat = tanh(logit) in [-1, 1]; this module applies
the Eq. (18) map with the bounds of the ActionConfig, which is derived
from the network data (see action_config.action_config_from_network),
so no limits are supplied or invented here:

    |v_i| : [-1, 1] -> [v_min, v_max]   (p.u.)
    p_g   : [-1, 1] -> [P_min, P_max]   (MW)

Notation follows the paper (see NOTATION.md).
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
    Physical action a for one policy output.

    v_sp[k]  |v_i|^sp of bus config.B_vs[k]   (p.u.)
    p_g[k]   p_g of unit config.G_ns[k]        (MW)
    """

    v_sp: np.ndarray
    p_g: np.ndarray
    base_mva: float

    @property
    def p_g_pu(self) -> np.ndarray:
        return self.p_g / self.base_mva

    @property
    def action_dim(self) -> int:
        return len(self.v_sp) + len(self.p_g)


class ActionDecoder:
    """Eq. (18): convert a_hat in [-1, 1] into the physical action a."""

    def __init__(self, config: ActionConfig) -> None:

        self.config = config

        self._v_min = np.asarray(config.v_min, dtype=float)
        self._v_max = np.asarray(config.v_max, dtype=float)

        self._P_min = config.P_min
        self._P_max = config.P_max

    def _validate_a_hat(
        self,
        a_hat: Sequence[float],
    ) -> np.ndarray:

        values = np.asarray(a_hat, dtype=float)

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
    def _scale(a_hat, a_min, a_max):
        """Eq. (18): a = a_min + (a_max - a_min) (a_hat + 1) / 2."""

        return a_min + 0.5 * (a_hat + 1.0) * (a_max - a_min)

    def decode(self, a_hat: Sequence[float]) -> DecodedAction:

        a_hat = self._validate_a_hat(a_hat)

        a_hat_v = a_hat[list(self.config.voltage_action_indices)]
        a_hat_p = a_hat[list(self.config.generator_action_indices)]

        return DecodedAction(
            v_sp=self._scale(a_hat_v, self._v_min, self._v_max),
            p_g=self._scale(a_hat_p, self._P_min, self._P_max),
            base_mva=self.config.base_mva,
        )


def apply_action_to_network(
    net: pp.pandapowerNet,
    config: ActionConfig,
    decoded: DecodedAction,
) -> None:
    """
    Write the physical action a into the pandapower network.

    |v_i|^sp:
        slack bus B_s -> ext_grid.vm_pu
        PV bus in B_v -> vm_pu of every in-service gen on that bus

    p_g:
        p_mw of the corresponding gen / sgen row

    Used for validation against pandapower and to build HelmProblem
    instances.  The training critic takes these values directly as
    tensors instead of going through pandapower.
    """

    for bus, v_sp_i in zip(config.B_vs, decoded.v_sp):

        if bus == config.G_s.bus:
            net.ext_grid.at[config.G_s.index, "vm_pu"] = float(v_sp_i)
            continue

        on_bus = net.gen.index[
            (net.gen["bus"] == bus) & net.gen["in_service"]
        ]

        if len(on_bus) == 0:
            raise ValueError(
                f"No in-service gen regulates voltage bus {bus}."
            )

        net.gen.loc[on_bus, "vm_pu"] = float(v_sp_i)

    for unit, p_g in zip(config.G_ns, decoded.p_g):
        net[unit.table].at[unit.index, "p_mw"] = float(p_g)
