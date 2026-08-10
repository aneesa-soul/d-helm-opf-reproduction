"""
Convert a pandapower network into the mathematical problem used by HELM.

This module is the bridge between:

    pandapower network
            ↓
        Ybus + Sbus
            ↓
        HelmProblem

The conversion is deliberately kept separate from the HELM recursion.
"""

from __future__ import annotations

import numpy as np
import pandapower as pp

from dhelm_opf.helm.embedding import HelmProblem


def extract_ybus(net: pp.pandapowerNet) -> np.ndarray:
    """
    Extract the complex bus-admittance matrix from pandapower.

    A power flow must be executed before calling this function because
    pandapower constructs the internal PPC representation during the
    power-flow preparation stage.
    """

    if "_ppc" not in net:
        raise RuntimeError(
            "Pandapower internal data are unavailable. "
            "Run pp.runpp(net) before extracting Ybus."
        )

    ybus = net._ppc["internal"]["Ybus"]

    # Pandapower may store Ybus as a sparse matrix.
    if hasattr(ybus, "toarray"):
        ybus = ybus.toarray()

    return np.asarray(ybus, dtype=np.complex128)


def get_slack_bus(net: pp.pandapowerNet) -> int:
    """
    Return the pandapower bus index used as the external-grid bus.

    The IEEE 5-bus case has one external grid.
    """

    if len(net.ext_grid) != 1:
        raise ValueError(
            "This reproduction currently expects exactly "
            "one external-grid/slack bus."
        )

    return int(net.ext_grid.iloc[0]["bus"])


def get_slack_voltage(net: pp.pandapowerNet) -> complex:
    """
    Return the specified complex slack voltage.

    pandapower stores the external-grid voltage magnitude and
    angle separately.
    """

    row = net.ext_grid.iloc[0]

    vm = float(row["vm_pu"])
    va_deg = float(row["va_degree"])

    va_rad = np.deg2rad(va_deg)

    return vm * np.exp(1j * va_rad)


def build_sbus(net: pp.pandapowerNet) -> np.ndarray:
    """
    Construct the net complex power injection vector.

    Positive values represent net injection into the network.

    The resulting vector is:

        S_bus = P_bus + j Q_bus

    This routine uses pandapower's solved bus power-balance results
    rather than attempting to reconstruct generator/load injections
    manually.
    """

    if not hasattr(net, "res_bus"):
        raise RuntimeError(
            "Power-flow results are unavailable. "
            "Run pp.runpp(net) first."
        )

    p = net.res_bus["p_mw"].to_numpy(dtype=float)
    q = net.res_bus["q_mvar"].to_numpy(dtype=float)

    return p + 1j * q


def build_helm_problem_from_network(
    net: pp.pandapowerNet,
) -> HelmProblem:
    """
    Convert a solved pandapower network into a HelmProblem.

    Parameters
    ----------
    net:
        Solved pandapower network.

    Returns
    -------
    HelmProblem
        Mathematical representation required by the HELM solver.
    """

    ybus = extract_ybus(net)
    sbus = build_sbus(net)

    slack_bus = get_slack_bus(net)
    slack_voltage = get_slack_voltage(net)

    return HelmProblem(
        ybus=ybus,
        s_spec=sbus,
        v_slack=slack_voltage,
        slack_bus=slack_bus,
    )


def inspect_helm_problem(problem: HelmProblem) -> None:
    """
    Print a compact diagnostic summary.
    """

    print("=" * 70)
    print("HELM problem")
    print("=" * 70)

    print(f"Number of buses : {problem.ybus.shape[0]}")
    print(f"Slack bus       : {problem.slack_bus}")
    print(f"Slack voltage   : {problem.v_slack}")
    print(f"Ybus shape      : {problem.ybus.shape}")
    print(f"Sbus shape      : {problem.s_spec.shape}")

    print("\nSpecified complex bus injections:")
    for bus, value in enumerate(problem.s_spec):
        print(
            f"  bus {bus}: "
            f"P={value.real:.6f} MW, "
            f"Q={value.imag:.6f} MVAr"
        )


if __name__ == "__main__":

    import pandapower.networks as pn

    print("Loading IEEE 5-bus network...")

    net = pn.case5()

    print("Running reference AC power flow...")

    pp.runpp(net)

    problem = build_helm_problem_from_network(net)

    inspect_helm_problem(problem)
