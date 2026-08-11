"""
Paper-specific uncertainty model for the D-HELM OPF reproduction.

The paper defines the system state as:

    s_t = (
        p_d(i,t),
        q_d(i,t),
        p_w(i,t)
    )

where demand uncertainty is applied to every demand bus and
wind-power uncertainty is applied to the designated wind buses.

For the IEEE 5-bus case:

    |D| = 3
    |W| = 2

therefore:

    uncertainty dimension = 3*2 + 2 = 8

The paper specifies:

    demand scaling  : [0.60, 1.10]
    wind scaling    : [0.40, 1.00]

The exact wind-bus placement is kept configurable because the
paper refers to its electronic companion for those placements.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandapower as pp


@dataclass(frozen=True)
class UncertaintyConfig:
    """
    Configuration of the paper's uncertain operating conditions.
    """

    demand_min: float = 0.60
    demand_max: float = 1.10

    wind_min: float = 0.40
    wind_max: float = 1.00

    # The paper specifies two wind buses for the 5-bus system.
    # Their exact indices are supplied externally rather than guessed.
    wind_buses: tuple[int, ...] = ()


@dataclass(frozen=True)
class OperatingScenario:
    """
    One realized uncertain operating condition.

    Arrays are ordered according to the corresponding pandapower
    table entries / configured wind buses.
    """

    demand_p_scale: np.ndarray
    demand_q_scale: np.ndarray
    wind_scale: np.ndarray


def validate_uncertainty_config(
    net: pp.pandapowerNet,
    config: UncertaintyConfig,
) -> None:
    """
    Validate the uncertainty configuration against the network.

    For the IEEE 5-bus case, the paper requires:

        3 demand buses
        2 wind buses
        8 total uncertainty variables
    """

    if len(net.load) != 3:
        raise ValueError(
            "The paper's 5-bus configuration expects "
            f"3 demand entries, but the network has {len(net.load)}."
        )

    if len(config.wind_buses) != 2:
        raise ValueError(
            "The paper's 5-bus configuration requires exactly "
            "2 wind buses. Supply their exact indices from the "
            "electronic companion."
        )

    n_bus = len(net.bus)

    for bus in config.wind_buses:
        if bus < 0 or bus >= n_bus:
            raise ValueError(
                f"Wind bus index {bus} is outside the "
                f"5-bus network range [0, {n_bus - 1}]."
            )

    if not 0.0 <= config.demand_min <= config.demand_max:
        raise ValueError("Invalid demand uncertainty interval.")

    if not 0.0 <= config.wind_min <= config.wind_max:
        raise ValueError("Invalid wind uncertainty interval.")


def sample_scenario(
    config: UncertaintyConfig,
    rng: np.random.Generator,
) -> OperatingScenario:
    """
    Sample one realization of demand and wind uncertainty.

    Demand:
        [0.60, 1.10]

    Wind:
        [0.40, 1.00]
    """

    if len(config.wind_buses) != 2:
        raise ValueError(
            "Two exact wind buses must be configured before "
            "sampling a 5-bus scenario."
        )

    demand_p_scale = rng.uniform(
        config.demand_min,
        config.demand_max,
        size=3,
    )

    demand_q_scale = rng.uniform(
        config.demand_min,
        config.demand_max,
        size=3,
    )

    wind_scale = rng.uniform(
        config.wind_min,
        config.wind_max,
        size=2,
    )

    return OperatingScenario(
        demand_p_scale=demand_p_scale,
        demand_q_scale=demand_q_scale,
        wind_scale=wind_scale,
    )


def apply_demand_uncertainty(
    net: pp.pandapowerNet,
    scenario: OperatingScenario,
) -> None:
    """
    Apply realized active/reactive demand scaling to the network.
    """

    if len(net.load) != len(scenario.demand_p_scale):
        raise ValueError(
            "Number of demand scaling values does not match "
            "the number of loads."
        )

    net.load["p_mw"] = (
        net.load["p_mw"].to_numpy()
        * scenario.demand_p_scale
    )

    net.load["q_mvar"] = (
        net.load["q_mvar"].to_numpy()
        * scenario.demand_q_scale
    )


def apply_wind_output(
    net: pp.pandapowerNet,
    wind_buses: Sequence[int],
    wind_scale: Sequence[float],
    wind_rated_mw: Sequence[float],
) -> None:
    """
    Apply realized wind output.

    The paper's electronic companion determines the exact renewable
    placement and rated capacities. Therefore this function does not
    assume that an existing pandapower generator is automatically a
    wind generator.

    Parameters
    ----------
    net:
        Pandapower network.

    wind_buses:
        Exact wind-bus indices.

    wind_scale:
        Realized [0.40, 1.00] scaling factors.

    wind_rated_mw:
        Rated wind outputs for the configured wind buses.
    """

    if len(wind_buses) != len(wind_scale):
        raise ValueError(
            "wind_buses and wind_scale must have equal length."
        )

    if len(wind_buses) != len(wind_rated_mw):
        raise ValueError(
            "wind_buses and wind_rated_mw must have equal length."
        )

    for bus, scale, rated in zip(
        wind_buses,
        wind_scale,
        wind_rated_mw,
    ):
        if bus < 0 or bus >= len(net.bus):
            raise ValueError(f"Invalid wind bus: {bus}")

        if scale < 0.40 or scale > 1.00:
            raise ValueError(
                f"Wind scale {scale} violates the paper's "
                "[0.40, 1.00] range."
            )

        output_mw = float(scale * rated)

        # Wind units will be represented explicitly as generators
        # when the exact companion-system configuration is available.
        #
        # We intentionally do not mutate net.gen here because doing so
        # without knowing which generators are wind resources could
        # silently corrupt the paper's test system.

        print(
            f"Wind bus {bus}: "
            f"scale={scale:.4f}, "
            f"output={output_mw:.4f} MW"
        )


def scenario_to_state(
    net: pp.pandapowerNet,
    scenario: OperatingScenario,
) -> np.ndarray:
    """
    Construct the 8-dimensional uncertainty state.

    Ordering:

        [p_d(3), q_d(3), p_w(2)]

    The wind values are the normalized realized outputs.
    """

    if len(net.load) != 3:
        raise ValueError("Expected 3 loads for the 5-bus case.")

    if len(scenario.wind_scale) != 2:
        raise ValueError("Expected 2 wind values for the 5-bus case.")

    state = np.concatenate(
        [
            scenario.demand_p_scale,
            scenario.demand_q_scale,
            scenario.wind_scale,
        ]
    )

    if state.shape != (8,):
        raise RuntimeError(
            f"Expected an 8-dimensional uncertainty state, "
            f"got shape {state.shape}."
        )

    return state.astype(np.float64)


if __name__ == "__main__":
    from .case5 import build_case5

    net = build_case5()

    # Intentionally left empty until the exact two wind-bus
    # indices from the electronic companion are identified.
    config = UncertaintyConfig(
        wind_buses=(),
    )

    try:
        validate_uncertainty_config(net, config)
    except ValueError as exc:
        print("Configuration check:")
        print(exc)
        print()
        print(
            "This is intentional: the paper specifies two wind buses "
            "but refers to the electronic companion for their exact "
            "placement."
        )
