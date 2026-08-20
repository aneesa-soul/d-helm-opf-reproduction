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
    Return the specified complex voltage of the external-grid
    slack bus.

    Pandapower stores the slack voltage as magnitude and angle:

        V = vm_pu * exp(j * va_degree)

    For the IEEE 5-bus case:

        vm_pu     = 1.0
        va_degree = 0.0

    therefore:

        V_slack = 1 + 0j
    """

    if len(net.ext_grid) != 1:
        raise ValueError(
            "This reproduction currently expects exactly "
            "one external-grid/slack bus."
        )

    row = net.ext_grid.iloc[0]

    vm = float(row["vm_pu"])
    va_deg = float(row["va_degree"])

    return vm * np.exp(1j * np.deg2rad(va_deg))

def get_pv_buses(net: pp.pandapowerNet) -> tuple[int, ...]:
    """
    Return buses containing in-service conventional generators.

    In the pandapower IEEE 5-bus network, these correspond to
    PV-type buses because their active power and voltage magnitude
    are specified.
    """

    pv_buses = []

    for _, row in net.gen.iterrows():
        if bool(row["in_service"]):
            pv_buses.append(int(row["bus"]))

    return tuple(sorted(set(pv_buses)))


def get_pq_buses(
    net: pp.pandapowerNet,
    slack_bus: int,
    pv_buses: tuple[int, ...],
) -> tuple[int, ...]:
    """
    Identify buses that are neither slack nor PV buses.
    """

    excluded = {
        slack_bus,
        *pv_buses,
    }

    pq_buses = tuple(
        bus
        for bus in range(len(net.bus))
        if bus not in excluded
    )

    return pq_buses


def get_voltage_setpoints(
    net: pp.pandapowerNet,
    slack_bus: int,
    pv_buses: tuple[int, ...],
) -> np.ndarray:
    """
    Construct the specified voltage-magnitude vector.

    NaN indicates that no voltage magnitude is prescribed.

    Slack voltage comes from ext_grid.vm_pu.

    PV voltage magnitudes come from gen.vm_pu.
    """

    n_bus = len(net.bus)

    setpoints = np.full(
        n_bus,
        np.nan,
        dtype=float,
    )

    # Slack voltage.
    ext_grid_row = net.ext_grid.iloc[0]
    setpoints[slack_bus] = float(
        ext_grid_row["vm_pu"]
    )

    # PV-bus voltage setpoints.
    for _, row in net.gen.iterrows():
        if bool(row["in_service"]):
            bus = int(row["bus"])

            setpoints[bus] = float(
                row["vm_pu"]
            )

    return setpoints

def build_sbus(net: pp.pandapowerNet) -> np.ndarray:
    """
    Construct the specified complex bus injection vector directly
    from the network operating point.

    Positive values represent net generation/injection.
    Negative values represent net demand.

    S_bus = (P_gen - P_load) + j(Q_gen - Q_load)

    This function intentionally does NOT use net.res_bus because
    HELM must calculate the power-flow solution independently.
    """

    n_bus = len(net.bus)

    p = np.zeros(n_bus, dtype=float)
    q = np.zeros(n_bus, dtype=float)

    # ------------------------------------------------------------
    # Conventional generators
    # ------------------------------------------------------------
    if len(net.gen) > 0:
        for _, row in net.gen.iterrows():
            if bool(row["in_service"]):
                bus = int(row["bus"])
                p[bus] += float(row["p_mw"])

                # Generator Q is not necessarily specified before
                # an AC power flow. For now, use q_mvar if available.
                if "q_mvar" in row.index and np.isfinite(row["q_mvar"]):
                    q[bus] += float(row["q_mvar"])

    # ------------------------------------------------------------
    # External grid / slack generation
    #
    # For the HELM formulation the slack bus voltage is fixed,
    # so we do not need to prescribe its active/reactive power.
    # ------------------------------------------------------------

    # ------------------------------------------------------------
    # Loads
    # ------------------------------------------------------------
    if len(net.load) > 0:
        for _, row in net.load.iterrows():
            if bool(row["in_service"]):
                bus = int(row["bus"])
                p[bus] -= float(row["p_mw"])
                q[bus] -= float(row["q_mvar"])

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

    pv_buses = get_pv_buses(net)

    pq_buses = get_pq_buses(
        net,
        slack_bus,
        pv_buses,
    )

    voltage_setpoints = get_voltage_setpoints(
        net,
        slack_bus,
        pv_buses,
    )

    return HelmProblem(
        ybus=ybus,
        s_spec=sbus,
        v_slack=slack_voltage,
        slack_bus=slack_bus,
        pv_buses=pv_buses,
        pq_buses=pq_buses,
        voltage_setpoints=voltage_setpoints,
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
