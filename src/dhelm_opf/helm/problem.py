"""
Convert a pandapower network into the mathematical problem used by HELM.

This module is the bridge between:

    pandapower network
            ↓
        Y + e_sp
            ↓
        HelmProblem

The conversion is deliberately kept separate from the HELM recursion.
Notation follows the paper (see NOTATION.md).
"""

from __future__ import annotations

import numpy as np
import pandapower as pp

from dhelm_opf.helm.embedding import HelmProblem


def extract_Y(net: pp.pandapowerNet) -> np.ndarray:
    """
    Extract the complex bus-admittance matrix Y from pandapower.

    A power flow must be executed before calling this function because
    pandapower constructs the internal PPC representation during the
    power-flow preparation stage.
    """

    if "_ppc" not in net:
        raise RuntimeError(
            "Pandapower internal data are unavailable. "
            "Run pp.runpp(net) before extracting Y."
        )

    Y = net._ppc["internal"]["Ybus"]

    # Pandapower may store Ybus as a sparse matrix.
    if hasattr(Y, "toarray"):
        Y = Y.toarray()

    return np.asarray(Y, dtype=np.complex128)


def get_B_s(net: pp.pandapowerNet) -> int:
    """
    Return the slack bus B_s (the external-grid bus).

    The IEEE 5-bus case has one external grid.
    """

    if len(net.ext_grid) != 1:
        raise ValueError(
            "This reproduction currently expects exactly "
            "one external-grid/slack bus."
        )

    return int(net.ext_grid.iloc[0]["bus"])


def get_v_s(net: pp.pandapowerNet) -> complex:
    """
    Return the specified complex voltage of the slack bus.

    Pandapower stores the slack voltage as magnitude and angle:

        v_s = vm_pu * exp(j * va_degree)

    For the IEEE 5-bus case:

        vm_pu     = 1.0
        va_degree = 0.0

    therefore:

        v_s = 1 + 0j
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


def get_B_v(net: pp.pandapowerNet) -> tuple[int, ...]:
    """
    Return the PV buses B_v: buses with in-service conventional
    generators.

    In the pandapower IEEE 5-bus network, these correspond to
    PV-type buses because their active power and voltage magnitude
    are specified.
    """

    B_v = []

    for _, row in net.gen.iterrows():
        if bool(row["in_service"]):
            B_v.append(int(row["bus"]))

    return tuple(sorted(set(B_v)))


def get_B_q(
    net: pp.pandapowerNet,
    B_s: int,
    B_v: tuple[int, ...],
) -> tuple[int, ...]:
    """
    Return the PQ buses B_q: buses that are neither slack nor PV.
    """

    excluded = {
        B_s,
        *B_v,
    }

    B_q = tuple(
        bus
        for bus in range(len(net.bus))
        if bus not in excluded
    )

    return B_q


def get_v_sp(
    net: pp.pandapowerNet,
    B_s: int,
    B_v: tuple[int, ...],
) -> np.ndarray:
    """
    Construct the specified voltage-magnitude vector |v_i|^sp.

    NaN indicates that no voltage magnitude is prescribed.

    Slack voltage comes from ext_grid.vm_pu.

    PV voltage magnitudes come from gen.vm_pu.
    """

    n_bus = len(net.bus)

    v_sp = np.full(
        n_bus,
        np.nan,
        dtype=float,
    )

    # Slack voltage.
    ext_grid_row = net.ext_grid.iloc[0]
    v_sp[B_s] = float(
        ext_grid_row["vm_pu"]
    )

    # PV-bus voltage setpoints.
    for _, row in net.gen.iterrows():
        if bool(row["in_service"]):
            bus = int(row["bus"])

            v_sp[bus] = float(
                row["vm_pu"]
            )

    return v_sp


def build_e_sp(net: pp.pandapowerNet) -> np.ndarray:
    """
    Construct the specified complex injection e_i^sp = p_i + j q_i
    (Eq. 8) for HELM:

        p_i = sum_{g in G_i} p_g + p_i^w - p_i^d
        q_i = (fixed sgen Q) - q_i^d

    All power quantities are converted from pandapower's MW/MVAr
    representation into per-unit using the network base power.

    Positive values represent net injection.
    Negative values represent net demand.

    For HELM:
        - PQ-bus P and Q are specified.
        - PV-bus P is specified, while Q is solved recursively.
        - Slack-bus P and Q are not specified because its voltage
          is fixed.
    """

    n_bus = len(net.bus)

    if net.sn_mva <= 0:
        raise ValueError(
            "Network base power sn_mva must be positive."
        )

    base_mva = float(net.sn_mva)

    p = np.zeros(n_bus, dtype=float)
    q = np.zeros(n_bus, dtype=float)

    # ------------------------------------------------------------
    # Generator active power p_g
    # ------------------------------------------------------------
    if len(net.gen) > 0:
        for _, row in net.gen.iterrows():
            if bool(row["in_service"]):
                bus = int(row["bus"])
                p[bus] += float(row["p_mw"]) / base_mva

    # Static generators (conventional sgens and wind p^w)
    if len(net.sgen) > 0:
        for _, row in net.sgen.iterrows():
            if bool(row["in_service"]):
                bus = int(row["bus"])
                p[bus] += (
                    float(row["p_mw"])
                    * float(row.get("scaling", 1.0))
                    / base_mva
                )
                q[bus] += (
                    float(row["q_mvar"])
                    * float(row.get("scaling", 1.0))
                    / base_mva
                )

    # ------------------------------------------------------------
    # Loads p^d, q^d
    # ------------------------------------------------------------
    if len(net.load) > 0:
        for _, row in net.load.iterrows():
            if bool(row["in_service"]):
                bus = int(row["bus"])

                p[bus] -= float(row["p_mw"]) / base_mva
                q[bus] -= float(row["q_mvar"]) / base_mva

    # ------------------------------------------------------------
    # Important:
    #
    # Do NOT add generator Q here.
    #
    # For PV buses, reactive power is an unknown HELM coefficient
    # Q_{i,n} and is solved through Eqs. (12)-(13) of the paper.
    #
    # Therefore q represents the specified PQ-bus reactive
    # injection only.
    # ------------------------------------------------------------

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

    Y = extract_Y(net)
    e_sp = build_e_sp(net)

    B_s = get_B_s(net)
    v_s = get_v_s(net)

    B_v = get_B_v(net)

    B_q = get_B_q(
        net,
        B_s,
        B_v,
    )

    v_sp = get_v_sp(
        net,
        B_s,
        B_v,
    )

    return HelmProblem(
        Y=Y,
        e_sp=e_sp,
        v_s=v_s,
        B_s=B_s,
        B_v=B_v,
        B_q=B_q,
        v_sp=v_sp,
    )


def inspect_helm_problem(problem: HelmProblem) -> None:
    """
    Print a compact diagnostic summary.
    """

    print("=" * 70)
    print("HELM problem")
    print("=" * 70)

    print(f"Number of buses : {problem.Y.shape[0]}")
    print(f"Slack bus B_s   : {problem.B_s}")
    print(f"Slack voltage   : {problem.v_s}")
    print(f"Y shape         : {problem.Y.shape}")
    print(f"e_sp shape      : {problem.e_sp.shape}")

    print("\nSpecified complex bus injections e_sp:")
    for bus, value in enumerate(problem.e_sp):
        print(
            f"  bus {bus}: "
            f"p={value.real:.6f} p.u., "
            f"q={value.imag:.6f} p.u."
        )


if __name__ == "__main__":

    import pandapower.networks as pn

    print("Loading IEEE 5-bus network...")

    net = pn.case5()

    print("Running reference AC power flow...")

    pp.runpp(net)

    problem = build_helm_problem_from_network(net)

    inspect_helm_problem(problem)
