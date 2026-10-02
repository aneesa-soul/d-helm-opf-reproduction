"""
Action configuration for the 5-bus D-HELM OPF reproduction.

Paper basis
-----------
For the 5-bus case, Table 1 defines the policy output dimension as:

    |B_v ∪ B_s| + |G / G_s| = 4 + 4 = 8

Therefore, the policy action contains:
    - 4 voltage-related controls
    - 4 active-power controls for non-slack generators

Important reproduction note
---------------------------
The paper does not explicitly enumerate the individual bus/generator
indices corresponding to these 4 + 4 controls.

Therefore, the specific mapping below is an explicit reproduction
assumption. It must not be interpreted as the author's exact mapping.

The configuration is kept separate from the D-HELM critic so that the
assumption can be changed later without modifying the physics model.
"""

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class ActionConfig:
    """
    Defines the 8-dimensional policy action for the IEEE 5-bus case.

    Action layout
    -------------
    action[0:4]:
        Voltage-related controls.

    action[4:8]:
        Active-power controls for non-slack generators.

    The exact mapping is a reproduction assumption because the paper
    does not enumerate the individual bus/generator indices.
    """

    # ------------------------------------------------------------------
    # Voltage-related controls
    # ------------------------------------------------------------------
    #
    # Reproduction assumption:
    # four controlled voltage buses.
    #
    # Bus numbering follows pandapower's zero-based indexing.
    voltage_buses: Tuple[int, ...] = (0, 1, 2, 4)

    # ------------------------------------------------------------------
    # Non-slack generator active-power controls
    # ------------------------------------------------------------------
    #
    # Reproduction assumption:
    # four non-slack generation controls.
    #
    # These are represented as generator identifiers rather than assuming
    # that the policy directly controls reactive power.
    generator_ids: Tuple[int, ...] = (0, 1, 2, 3)

    # ------------------------------------------------------------------
    # Action bounds
    # ------------------------------------------------------------------
    #
    # These are physical bounds used to interpret the normalized policy
    # output. The exact numerical limits should be populated from the
    # network/paper before using them for final experiments.
    #
    # None means that the physical limit has not yet been assigned here.
    voltage_min: float = 0.90
    voltage_max: float = 1.10

    # Placeholder for generator active-power bounds.
    # These are intentionally not hard-coded here because the exact
    # generator-specific limits need to come from the implemented case.
    generator_pmin: Tuple[float, ...] = ()
    generator_pmax: Tuple[float, ...] = ()

    def __post_init__(self) -> None:
        """Validate the action configuration."""

        if len(self.voltage_buses) != 4:
            raise ValueError(
                "The 5-bus paper configuration requires "
                "4 voltage-related action components."
            )

        if len(self.generator_ids) != 4:
            raise ValueError(
                "The 5-bus paper configuration requires "
                "4 non-slack generator action components."
            )

        if self.voltage_min >= self.voltage_max:
            raise ValueError(
                "voltage_min must be smaller than voltage_max."
            )

        if self.generator_pmin and len(self.generator_pmin) != 4:
            raise ValueError(
                "generator_pmin must contain exactly 4 values."
            )

        if self.generator_pmax and len(self.generator_pmax) != 4:
            raise ValueError(
                "generator_pmax must contain exactly 4 values."
            )

        if self.generator_pmin and self.generator_pmax:
            for pmin, pmax in zip(
                self.generator_pmin,
                self.generator_pmax,
            ):
                if pmin > pmax:
                    raise ValueError(
                        "Every generator Pmin must be <= Pmax."
                    )

    @property
    def action_dim(self) -> int:
        """Return the policy action dimension."""

        return len(self.voltage_buses) + len(self.generator_ids)

    @property
    def voltage_action_indices(self) -> Tuple[int, ...]:
        """Indices occupied by voltage-related controls."""

        return tuple(range(len(self.voltage_buses)))

    @property
    def generator_action_indices(self) -> Tuple[int, ...]:
        """Indices occupied by generator active-power controls."""

        start = len(self.voltage_buses)

        return tuple(
            range(
                start,
                start + len(self.generator_ids),
            )
        )

    def describe(self) -> str:
        """Return a human-readable description of the action layout."""

        lines = [
            "D-HELM OPF 5-bus action configuration",
            "",
            "Paper-defined structure:",
            "  Voltage-related controls : 4",
            "  Non-slack generator P    : 4",
            "  Total action dimension   : 8",
            "",
            "Reproduction-assumed mapping:",
            f"  Voltage buses            : {self.voltage_buses}",
            f"  Generator IDs            : {self.generator_ids}",
            "",
            f"Voltage bounds             : "
            f"[{self.voltage_min}, {self.voltage_max}]",
        ]

        return "\n".join(lines)


DEFAULT_ACTION_CONFIG = ActionConfig()
