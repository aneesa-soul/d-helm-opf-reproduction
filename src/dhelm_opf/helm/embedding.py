"""
Holomorphic embedding data structures for D-HELM.

This module defines the mathematical problem passed to the HELM
coefficient recursion.

The representation keeps the network equations separate from the
reinforcement-learning layer.  This allows the HELM solution to be
validated independently against a conventional AC power-flow solution.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class HelmProblem:
    """
    Representation of the AC power-flow problem used by HELM.

    Parameters
    ----------
    ybus:
        Complex bus-admittance matrix.

    s_spec:
        Complex specified net power injection at each bus.

        Positive P/Q means net injection into the network.
        Negative P/Q means net demand.

    v_slack:
        Complex voltage of the reference/slack bus.

    slack_bus:
        Index of the reference/slack bus.

    pv_buses:
        Indices of PV buses.

    pq_buses:
        Indices of PQ buses.

    voltage_setpoints:
        Specified voltage magnitudes for buses where a voltage
        magnitude is prescribed.  Entries may be NaN for buses
        without a specified voltage magnitude.
    """

    ybus: np.ndarray
    s_spec: np.ndarray
    v_slack: complex
    slack_bus: int

    pv_buses: tuple[int, ...] = ()
    pq_buses: tuple[int, ...] = ()
    voltage_setpoints: np.ndarray | None = None

    def __post_init__(self) -> None:
        """Validate the HELM problem dimensions and bus types."""

        self.ybus = np.asarray(
            self.ybus,
            dtype=np.complex128,
        )

        self.s_spec = np.asarray(
            self.s_spec,
            dtype=np.complex128,
        )

        if self.ybus.ndim != 2:
            raise ValueError(
                "Ybus must be a two-dimensional matrix."
            )

        if self.ybus.shape[0] != self.ybus.shape[1]:
            raise ValueError(
                "Ybus must be square."
            )

        n_bus = self.ybus.shape[0]

        if self.s_spec.shape != (n_bus,):
            raise ValueError(
                "s_spec must contain exactly one value per bus."
            )

        if not 0 <= self.slack_bus < n_bus:
            raise ValueError(
                f"Invalid slack bus {self.slack_bus} "
                f"for a {n_bus}-bus system."
            )

        # Convert bus collections to immutable integer tuples.
        self.pv_buses = tuple(int(bus) for bus in self.pv_buses)
        self.pq_buses = tuple(int(bus) for bus in self.pq_buses)

        # Validate PV bus indices.
        for bus in self.pv_buses:
            if not 0 <= bus < n_bus:
                raise ValueError(
                    f"Invalid PV bus {bus} "
                    f"for a {n_bus}-bus system."
                )

        # Validate PQ bus indices.
        for bus in self.pq_buses:
            if not 0 <= bus < n_bus:
                raise ValueError(
                    f"Invalid PQ bus {bus} "
                    f"for a {n_bus}-bus system."
                )

        if self.slack_bus in self.pv_buses:
            raise ValueError(
                "The slack bus must not also be listed as a PV bus."
            )

        if self.slack_bus in self.pq_buses:
            raise ValueError(
                "The slack bus must not also be listed as a PQ bus."
            )

        if set(self.pv_buses) & set(self.pq_buses):
            raise ValueError(
                "A bus cannot simultaneously be PV and PQ."
            )

        # If voltage setpoints are not supplied, create an array
        # containing NaN for every bus.
        if self.voltage_setpoints is None:
            self.voltage_setpoints = np.full(
                n_bus,
                np.nan,
                dtype=float,
            )
        else:
            self.voltage_setpoints = np.asarray(
                self.voltage_setpoints,
                dtype=float,
            )

            if self.voltage_setpoints.shape != (n_bus,):
                raise ValueError(
                    "voltage_setpoints must contain exactly "
                    "one value per bus."
                )

        # The slack voltage magnitude is always known.
        self.voltage_setpoints[self.slack_bus] = abs(
            self.v_slack
        )


def make_helm_problem(
    ybus: np.ndarray,
    s_spec: np.ndarray,
    slack_bus: int,
    v_slack: complex = 1.0 + 0.0j,
    pv_buses: tuple[int, ...] = (),
    pq_buses: tuple[int, ...] = (),
    voltage_setpoints: np.ndarray | None = None,
) -> HelmProblem:
    """
    Construct a validated HELM problem.
    """

    return HelmProblem(
        ybus=ybus,
        s_spec=s_spec,
        v_slack=v_slack,
        slack_bus=slack_bus,
        pv_buses=pv_buses,
        pq_buses=pq_buses,
        voltage_setpoints=voltage_setpoints,
    )


def power_injection_from_voltage(
    ybus: np.ndarray,
    voltage: np.ndarray,
) -> np.ndarray:
    """
    Calculate complex bus power injections.

    The AC relationship is

        I = Y V

    and

        S = V * conjugate(I).
    """

    ybus = np.asarray(
        ybus,
        dtype=np.complex128,
    )

    voltage = np.asarray(
        voltage,
        dtype=np.complex128,
    )

    current = ybus @ voltage

    return voltage * np.conjugate(current)


def voltage_magnitudes(
    voltage: np.ndarray,
) -> np.ndarray:
    """Return voltage magnitudes."""

    return np.abs(
        np.asarray(
            voltage,
            dtype=np.complex128,
        )
    )


def voltage_angles(
    voltage: np.ndarray,
) -> np.ndarray:
    """Return voltage phase angles in radians."""

    return np.angle(
        np.asarray(
            voltage,
            dtype=np.complex128,
        )
    )