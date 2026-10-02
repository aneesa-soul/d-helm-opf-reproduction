"""
Decode the normalized 8-dimensional policy action into physical OPF
setpoints.

Paper basis
-----------
For the 5-bus case, Table 1 gives:

    Policy output = |B_v ∪ B_s| + |G / G_s|
                  = 4 + 4
                  = 8

The policy therefore produces four voltage-related controls and four
non-slack-generator active-power controls.

The paper does not explicitly identify the individual buses/generators
corresponding to those eight outputs. Therefore, this decoder does not
silently assume generator identities or generator limits.

The physical generator bounds must be supplied explicitly by the caller.
"""

from dataclasses import dataclass
from typing import Sequence, Tuple

import numpy as np

from .action_config import ActionConfig, DEFAULT_ACTION_CONFIG


@dataclass(frozen=True)
class DecodedAction:
    """
    Physical action passed toward the D-HELM physics model.

    Attributes
    ----------
    voltage_setpoints:
        Four voltage-magnitude setpoints in p.u.

    generator_p_setpoints:
        Four active-power setpoints in p.u.
    """

    voltage_setpoints: Tuple[float, ...]
    generator_p_setpoints: Tuple[float, ...]

    @property
    def action_dim(self) -> int:
        return len(self.voltage_setpoints) + len(
            self.generator_p_setpoints
        )


class ActionDecoder:
    """
    Convert normalized policy outputs into physical OPF setpoints.

    Normalized action convention
    ----------------------------
    Each policy output is expected to be in [-1, 1].

    Voltage controls:
        [-1, 1] -> [V_min, V_max]

    Generator controls:
        [-1, 1] -> [P_min, P_max]

    This linear scaling is an implementation representation of the
    physical action bounds. The policy's tanh/bound-enforcement
    mechanism will be connected later.
    """

    def __init__(
        self,
        config: ActionConfig = DEFAULT_ACTION_CONFIG,
        generator_pmin: Sequence[float] | None = None,
        generator_pmax: Sequence[float] | None = None,
    ) -> None:

        self.config = config

        if generator_pmin is None or generator_pmax is None:
            raise ValueError(
                "generator_pmin and generator_pmax must be supplied "
                "explicitly for all 4 generator controls. "
                "Do not invent generator limits."
            )

        self.generator_pmin = tuple(
            float(value) for value in generator_pmin
        )
        self.generator_pmax = tuple(
            float(value) for value in generator_pmax
        )

        if len(self.generator_pmin) != 4:
            raise ValueError(
                "Exactly 4 generator Pmin values are required."
            )

        if len(self.generator_pmax) != 4:
            raise ValueError(
                "Exactly 4 generator Pmax values are required."
            )

        for pmin, pmax in zip(
            self.generator_pmin,
            self.generator_pmax,
        ):
            if pmin > pmax:
                raise ValueError(
                    f"Invalid generator bounds: "
                    f"Pmin={pmin} > Pmax={pmax}"
                )

    @staticmethod
    def _validate_normalized_action(
        action: Sequence[float],
    ) -> np.ndarray:
        """Validate and convert the policy action."""

        values = np.asarray(action, dtype=float)

        if values.ndim != 1:
            raise ValueError(
                "Action must be a one-dimensional vector."
            )

        if values.shape[0] != 8:
            raise ValueError(
                f"Expected an 8-dimensional action, "
                f"received shape {values.shape}."
            )

        if not np.all(np.isfinite(values)):
            raise ValueError(
                "Action contains NaN or infinite values."
            )

        if np.any(values < -1.0) or np.any(values > 1.0):
            raise ValueError(
                "Normalized action values must lie in [-1, 1]."
            )

        return values

    @staticmethod
    def _scale(
        value: float,
        lower: float,
        upper: float,
    ) -> float:
        """
        Map [-1, 1] to [lower, upper].
        """

        return lower + 0.5 * (value + 1.0) * (
            upper - lower
        )

    def decode(
        self,
        action: Sequence[float],
    ) -> DecodedAction:
        """
        Decode an 8-dimensional normalized policy action.
        """

        action = self._validate_normalized_action(action)

        # --------------------------------------------------------------
        # First four policy outputs -> voltage setpoints
        # --------------------------------------------------------------

        voltage_setpoints = tuple(
            self._scale(
                value=action[index],
                lower=self.config.voltage_min,
                upper=self.config.voltage_max,
            )
            for index in self.config.voltage_action_indices
        )

        # --------------------------------------------------------------
        # Last four policy outputs -> generator P setpoints
        # --------------------------------------------------------------

        generator_setpoints = tuple(
            self._scale(
                value=action[index],
                lower=self.generator_pmin[position],
                upper=self.generator_pmax[position],
            )
            for position, index in enumerate(
                self.config.generator_action_indices
            )
        )

        return DecodedAction(
            voltage_setpoints=voltage_setpoints,
            generator_p_setpoints=generator_setpoints,
        )

