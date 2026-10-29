"""
Check that the action layout derived from the IEEE 5-bus network matches
the paper's Table 1 definition: |B_v ∪ B_s| + |G / G_s| = 4 + 4 = 8.
"""

from dhelm_opf.environment.case5 import build_case5
from dhelm_opf.models.action_config import action_config_from_network


def main() -> None:

    config = action_config_from_network(build_case5())

    print(config.describe())
    print()

    # Dimensions (paper, Table 1)
    assert config.n_v == 4
    assert config.n_g == 4
    assert config.action_dim == 8

    # Voltage-controlled buses: PV {0, 2, 4} + slack {3}
    assert config.B_vs == (0, 2, 3, 4)
    assert config.G_s.bus == 3
    assert config.v_min == (0.9, 0.9, 0.9, 0.9)
    assert config.v_max == (1.1, 1.1, 1.1, 1.1)

    # Non-slack generators, ordered by bus (gen before sgen)
    labels = [unit.label for unit in config.G_ns]
    assert labels == [
        "gen0@bus0",
        "sgen0@bus0",
        "gen1@bus2",
        "gen2@bus4",
    ], labels

    # Limits and costs come from the case data
    assert [u.P_max for u in config.G_ns] == [40, 170, 520, 600]
    assert [u.cp1 for u in config.G_ns] == [14, 15, 30, 10]
    assert (config.G_s.P_max, config.G_s.cp1) == (200, 40)

    print("All action-config tests passed.")


if __name__ == "__main__":
    main()
