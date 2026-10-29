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


# Wind units are pandapower sgens tagged with this type label
# (pandapower's convention for wind power).  The tag is what separates
# them from controllable conventional sgens such as sgen0 at bus 0 in
# the IEEE 5-bus case.
WIND_SGEN_TYPE = "WP"


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
    
    # Replication assumption:
    # The paper specifies two wind buses, but their exact placement
    # is not available in the publicly accessible material.
    B_w: tuple[int, ...] = (1, 4)
    # Replication assumption: 100 MW rated wind capacity at each bus.
    p_w_rated_mw: tuple[float, ...] = (100.0, 100.0)


@dataclass(frozen=True)
class OperatingScenario:
    """
    One realized uncertain operating condition.

    Arrays are ordered according to the corresponding pandapower
    table entries / configured wind buses.
    """

    p_d_scale: np.ndarray
    q_d_scale: np.ndarray
    p_w_scale: np.ndarray


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

    if len(config.B_w) != 2:
        raise ValueError(
            "The paper's 5-bus configuration requires exactly "
            "2 wind buses. Supply their exact indices from the "
            "electronic companion."
        )

    n_bus = len(net.bus)

    for bus in config.B_w:
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

    if len(config.B_w) != 2:
        raise ValueError(
            "Two exact wind buses must be configured before "
            "sampling a 5-bus scenario."
        )

    p_d_scale = rng.uniform(
        config.demand_min,
        config.demand_max,
        size=3,
    )

    q_d_scale = rng.uniform(
        config.demand_min,
        config.demand_max,
        size=3,
    )

    p_w_scale = rng.uniform(
        config.wind_min,
        config.wind_max,
        size=2,
    )

    return OperatingScenario(
        p_d_scale=p_d_scale,
        q_d_scale=q_d_scale,
        p_w_scale=p_w_scale,
    )


def apply_demand_uncertainty(
    net: pp.pandapowerNet,
    scenario: OperatingScenario,
) -> None:
    """
    Apply realized active/reactive demand scaling to the network.
    """

    if len(net.load) != len(scenario.p_d_scale):
        raise ValueError(
            "Number of demand scaling values does not match "
            "the number of loads."
        )

    net.load["p_mw"] = (
        net.load["p_mw"].to_numpy()
        * scenario.p_d_scale
    )

    net.load["q_mvar"] = (
        net.load["q_mvar"].to_numpy()
        * scenario.q_d_scale
    )


def apply_wind_output(
    net: pp.pandapowerNet,
    B_w: Sequence[int],
    p_w_scale: Sequence[float],
    p_w_rated_mw: Sequence[float],
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

    B_w:
        Exact wind-bus indices.

    p_w_scale:
        Realized [0.40, 1.00] scaling factors.

    p_w_rated_mw:
        Rated wind outputs for the configured wind buses.
    """

    if len(B_w) != len(p_w_scale):
        raise ValueError(
            "B_w and p_w_scale must have equal length."
        )

    if len(B_w) != len(p_w_rated_mw):
        raise ValueError(
            "B_w and p_w_rated_mw must have equal length."
        )

    for bus, scale, rated in zip(
        B_w,
        p_w_scale,
        p_w_rated_mw,
    ):
        if bus < 0 or bus >= len(net.bus):
            raise ValueError(f"Invalid wind bus: {bus}")

        if scale < 0.40 or scale > 1.00:
            raise ValueError(
                f"Wind scale {scale} violates the paper's "
                "[0.40, 1.00] range."
            )

        output_mw = float(scale * rated)

        # Represent wind explicitly as a static generator (sgen)
        # tagged with type WIND_SGEN_TYPE and controllable=False.
        #
        # Only an existing WIND sgen on this bus is reused.  Matching
        # on bus alone would overwrite a conventional sgen (e.g. the
        # 170 MW sgen0 at bus 0 in the IEEE 5-bus case).

        existing = net.sgen.index[
            (net.sgen["bus"] == bus)
            & (net.sgen["type"] == WIND_SGEN_TYPE)
        ].tolist()

        if existing:
            idx = existing[0]

            net.sgen.at[idx, "p_mw"] = output_mw
            net.sgen.at[idx, "q_mvar"] = 0.0
            net.sgen.at[idx, "scaling"] = 1.0
            net.sgen.at[idx, "in_service"] = True

        else:
            pp.create_sgen(
                net,
                bus=bus,
                p_mw=output_mw,
                q_mvar=0.0,
                scaling=1.0,
                in_service=True,
                name=f"wind_bus_{bus}",
                type=WIND_SGEN_TYPE,
                controllable=False,
            )

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

    if len(scenario.p_w_scale) != 2:
        raise ValueError("Expected 2 wind values for the 5-bus case.")

    state = np.concatenate(
        [
            scenario.p_d_scale,
            scenario.q_d_scale,
            scenario.p_w_scale,
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

    config = UncertaintyConfig(
        B_w=(1, 4),
        p_w_rated_mw=(100.0, 100.0),
    )

    validate_uncertainty_config(net, config)

    rng = np.random.default_rng(42)

    scenario = sample_scenario(
        config=config,
        rng=rng,
    )

    apply_demand_uncertainty(
        net,
        scenario,
    )

    apply_wind_output(
        net=net,
        B_w=config.B_w,
        p_w_scale=scenario.p_w_scale,
        p_w_rated_mw=config.p_w_rated_mw,
    )

    state = scenario_to_state(
        net,
        scenario,
    )

    print("\nUncertainty state:")
    print(state)

    print("\nWind generators:")
    print(net.sgen[["bus", "p_mw", "q_mvar", "in_service"]])
