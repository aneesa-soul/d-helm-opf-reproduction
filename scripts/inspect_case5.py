"""
Inspect the IEEE 5-bus pandapower network.

This script does NOT train an RL agent and does NOT modify the network.

Its purpose is to establish the network quantities that we need
for the D-HELM/OPF implementation.
"""

from __future__ import annotations

import numpy as np
import pandapower as pp
import pandapower.networks as pn


def print_section(title: str) -> None:
    """Print a clearly separated section heading."""
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def main() -> None:
    # ---------------------------------------------------------------
    # 1. Load the IEEE 5-bus test system
    # ---------------------------------------------------------------
    print_section("1. Loading IEEE 5-bus network")

    net = pn.case5()

    print(f"Number of buses      : {len(net.bus)}")
    print(f"Number of loads      : {len(net.load)}")
    print(f"Number of generators : {len(net.gen)}")
    print(f"Number of lines      : {len(net.line)}")
    print(f"Number of ext_grids  : {len(net.ext_grid)}")

    # ---------------------------------------------------------------
    # 2. Display bus information
    # ---------------------------------------------------------------
    print_section("2. Bus data")

    print(net.bus)

    # ---------------------------------------------------------------
    # 3. Display load information
    # ---------------------------------------------------------------
    print_section("3. Load data")

    if len(net.load) > 0:
        print(net.load[
            [
                "bus",
                "p_mw",
                "q_mvar",
                "scaling",
            ]
        ])
    else:
        print("No loads found.")

    # ---------------------------------------------------------------
    # 4. Display generator information
    # ---------------------------------------------------------------
    print_section("4. Generator data")

    if len(net.gen) > 0:
        generator_columns = [
            column
            for column in [
                "bus",
                "p_mw",
                "vm_pu",
                "min_p_mw",
                "max_p_mw",
                "min_q_mvar",
                "max_q_mvar",
            ]
            if column in net.gen.columns
        ]

        print(net.gen[generator_columns])
    else:
        print("No generators found.")

    # ---------------------------------------------------------------
    # 5. Display external-grid/slack information
    # ---------------------------------------------------------------
    print_section("5. External grid / slack bus")

    if len(net.ext_grid) > 0:
        print(net.ext_grid)
    else:
        print("No external grid found.")

    # ---------------------------------------------------------------
    # 6. Display line information
    # ---------------------------------------------------------------
    print_section("6. Line data")

    if len(net.line) > 0:
        line_columns = [
            column
            for column in [
                "from_bus",
                "to_bus",
                "length_km",
                "r_ohm_per_km",
                "x_ohm_per_km",
                "max_i_ka",
            ]
            if column in net.line.columns
        ]

        print(net.line[line_columns])
    else:
        print("No lines found.")

    # ---------------------------------------------------------------
    # 7. Run the conventional AC power flow
    #
    # This is only a diagnostic/reference calculation.
    # The eventual paper implementation will use HELM.
    # ---------------------------------------------------------------
    print_section("7. Conventional pandapower AC power flow")

    try:
        pp.runpp(net)

        print("Power flow converged.")

        print("\nBus voltages:")
        print(net.res_bus[["vm_pu", "va_degree"]])

        if len(net.res_line) > 0:
            print("\nLine results:")
            print(
                net.res_line[
                    [
                        "p_from_mw",
                        "q_from_mvar",
                        "p_to_mw",
                        "q_to_mvar",
                        "loading_percent",
                    ]
                ]
            )

        if len(net.res_gen) > 0:
            print("\nGenerator results:")
            print(net.res_gen)

        if len(net.res_ext_grid) > 0:
            print("\nExternal-grid results:")
            print(net.res_ext_grid)

    except Exception as exc:
        print(f"Power flow failed: {exc}")

    # ---------------------------------------------------------------
    # 8. Build the network admittance matrix
    #
    # pandapower internally builds Ybus for the AC power-flow
    # calculation. We expose it here because Ybus is fundamental
    # to the HELM formulation.
    # ---------------------------------------------------------------
    print_section("8. Network admittance matrix")

    try:
        # Internal PYPOWER representation is created by runpp().
        # Accessing it lets us inspect the numerical network model.
        Ybus = net._ppc["internal"]["Ybus"]

        print(f"Ybus shape: {Ybus.shape}")

        # Convert to a dense array only for this small diagnostic case.
        Ybus_dense = Ybus.toarray()

        print("\nYbus:")
        np.set_printoptions(precision=6, suppress=True)
        print(Ybus_dense)

    except Exception as exc:
        print(f"Could not access Ybus: {exc}")

    # ---------------------------------------------------------------
    # 9. Paper-specific dimensions
    #
    # These are taken from Table 1 of the paper.
    # They are NOT inferred from the pandapower network.
    # ---------------------------------------------------------------
    print_section("9. Paper reference dimensions")

    paper_uncertainty_dim = 3 * 2 + 2
    paper_policy_input_dim = 8 + 4
    paper_policy_output_dim = 4 + 4
    paper_helm_output_dim = 32

    print(f"Paper uncertainty dimension : {paper_uncertainty_dim}")
    print(f"Paper policy input          : {paper_policy_input_dim}")
    print(f"Paper policy output         : {paper_policy_output_dim}")
    print(f"Paper HELM output           : {paper_helm_output_dim}")

    # ---------------------------------------------------------------
    # 10. Summary
    # ---------------------------------------------------------------
    print_section("10. Summary")

    print("The network has been inspected successfully.")
    print(
        "Next we will compare this base pandapower system "
        "with the paper-specific uncertainty and renewable setup."
    )


if __name__ == "__main__":
    main()
