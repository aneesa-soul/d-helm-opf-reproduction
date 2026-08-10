"""
IEEE 5-bus network definition for the D-HELM OPF reproduction.

This module creates the base pandapower network used by the
reproduction. Paper-specific uncertainty and renewable scenarios
are handled separately.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandapower as pp
import pandapower.networks as pn


@dataclass(frozen=True)
class Case5Metadata:
    """Structural information about the IEEE 5-bus system."""

    n_bus: int = 5
    n_load: int = 3
    n_gen: int = 3
    n_line: int = 6

    # Dimensions reported for the 5-bus case in the paper.
    uncertainty_dim: int = 8
    policy_input_dim: int = 12
    policy_output_dim: int = 8
    helm_output_dim: int = 32


def build_case5() -> pp.pandapowerNet:
    """
    Construct the base IEEE 5-bus pandapower network.

    Returns
    -------
    pandapowerNet
        Fresh IEEE 5-bus network.

    Notes
    -----
    This function intentionally does not apply the paper's
    uncertainty model. That will be implemented separately so
    that the base network and stochastic operating conditions
    remain clearly distinguishable.
    """
    net = pn.case5()

    return net


def validate_case5(net: pp.pandapowerNet) -> None:
    """
    Validate the expected structural dimensions of the case.

    Raises
    ------
    ValueError
        If the loaded network does not have the expected
        IEEE 5-bus structure.
    """

    expected = Case5Metadata()

    actual = {
        "n_bus": len(net.bus),
        "n_load": len(net.load),
        "n_gen": len(net.gen),
        "n_line": len(net.line),
    }

    for name, expected_value in {
        "n_bus": expected.n_bus,
        "n_load": expected.n_load,
        "n_gen": expected.n_gen,
        "n_line": expected.n_line,
    }.items():

        if actual[name] != expected_value:
            raise ValueError(
                f"Unexpected case5 {name}: "
                f"expected {expected_value}, "
                f"got {actual[name]}"
            )


def run_reference_power_flow(net: pp.pandapowerNet) -> None:
    """
    Run a conventional AC power flow.

    This is a reference/diagnostic calculation only.

    The actual reproduction will use the differentiable HELM
    model for the physics-driven critic.
    """

    pp.runpp(net)


def get_ybus(net: pp.pandapowerNet):
    """
    Return the network bus-admittance matrix after a power flow.

    Parameters
    ----------
    net:
        A pandapower network on which runpp() has already been called.

    Returns
    -------
    scipy sparse matrix
        The complex bus-admittance matrix.
    """

    if "_ppc" not in net:
        raise RuntimeError(
            "Power-flow data are unavailable. "
            "Run run_reference_power_flow(net) first."
        )

    return net._ppc["internal"]["Ybus"]


if __name__ == "__main__":

    network = build_case5()

    validate_case5(network)

    run_reference_power_flow(network)

    ybus = get_ybus(network)

    metadata = Case5Metadata()

    print("IEEE 5-bus network validated.")
    print(f"Buses      : {metadata.n_bus}")
    print(f"Loads      : {metadata.n_load}")
    print(f"Generators : {metadata.n_gen}")
    print(f"Lines      : {metadata.n_line}")
    print(f"Ybus shape : {ybus.shape}")
