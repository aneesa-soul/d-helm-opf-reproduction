"""
Holomorphic embedding data structures for D-HELM.

This module provides the basic representation used by the
holomorphic embedding method.  The actual recursive coefficient
calculation is implemented separately in recursion.py.

The purpose of keeping this separate is to make the physics solver
testable independently from the RL policy and CPO algorithm.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class HelmProblem:
    """
    Representation of an AC power-flow problem for HELM.

    Parameters
    ----------
    ybus:
        Complex bus-admittance matrix.

    s_spec:
        Complex specified power injection at each bus.

    v_slack:
        Complex voltage at the reference/slack bus.

    slack_bus:
        Index of the reference bus.
    """

    ybus: np.ndarray
    s_spec: np.ndarray
    v_slack: complex
    slack_bus: int

    def __post_init__(self) -> None:
        """Validate the problem dimensions."""

        self.ybus = np.asarray(self.ybus, dtype=np.complex128)
        self.s_spec = np.asarray(self.s_spec, dtype=np.complex128)

        if self.ybus.ndim != 2:
            raise ValueError("Ybus must be a two-dimensional matrix.")

        if self.ybus.shape[0] != self.ybus.shape[1]:
            raise ValueError("Ybus must be square.")

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


def make_helm_problem(
    ybus: np.ndarray,
    s_spec: np.ndarray,
    slack_bus: int,
    v_slack: complex = 1.0 + 0.0j,
) -> HelmProblem:
    """
    Construct a validated HELM problem.

    Parameters
    ----------
    ybus:
        Complex network admittance matrix.

    s_spec:
        Complex specified power injections.

    slack_bus:
        Reference bus index.

    v_slack:
        Reference-bus voltage.

    Returns
    -------
    HelmProblem
        Validated problem representation.
    """

    return HelmProblem(
        ybus=ybus,
        s_spec=s_spec,
        v_slack=v_slack,
        slack_bus=slack_bus,
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

    Parameters
    ----------
    ybus:
        Complex bus-admittance matrix.

    voltage:
        Complex bus-voltage vector.

    Returns
    -------
    numpy.ndarray
        Complex power injection vector.
    """

    ybus = np.asarray(ybus, dtype=np.complex128)
    voltage = np.asarray(voltage, dtype=np.complex128)

    current = ybus @ voltage

    return voltage * np.conjugate(current)


def voltage_magnitudes(
    voltage: np.ndarray,
) -> np.ndarray:
    """
    Return voltage magnitudes in per-unit.
    """

    return np.abs(np.asarray(voltage, dtype=np.complex128))


def voltage_angles(
    voltage: np.ndarray,
) -> np.ndarray:
    """
    Return voltage phase angles in radians.
    """

    return np.angle(np.asarray(voltage, dtype=np.complex128))
