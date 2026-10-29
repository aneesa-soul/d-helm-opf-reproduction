"""
Holomorphic embedding data structures for D-HELM.

This module defines the mathematical problem passed to the HELM
coefficient recursion.

The representation keeps the network equations separate from the
reinforcement-learning layer.  This allows the HELM solution to be
validated independently against a conventional AC power-flow solution.

Notation follows the paper (see NOTATION.md):

    Y          bus admittance matrix
    e_sp       specified complex injection e_i^sp = p_i + j q_i  (Eq. 8)
    v_sp       voltage-magnitude setpoints |v_i|^sp
    B_s        slack bus;  B_v PV buses;  B_q PQ buses
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
    Y:
        Complex bus-admittance matrix Y_ij.

    e_sp:
        Specified complex net power injection e_i^sp = p_i + j q_i at
        each bus (Eq. 8).

        Positive P/Q means net injection into the network.
        Negative P/Q means net demand.

    v_s:
        Complex voltage of the slack bus (|v_i|^sp at angle 0 in the
        paper).

    B_s:
        Index of the slack bus.

    B_v:
        Indices of PV buses.

    B_q:
        Indices of PQ buses.

    v_sp:
        Specified voltage magnitudes |v_i|^sp for buses where a voltage
        magnitude is prescribed.  Entries may be NaN for buses without
        a specified voltage magnitude.
    """

    Y: np.ndarray
    e_sp: np.ndarray
    v_s: complex
    B_s: int

    B_v: tuple[int, ...] = ()
    B_q: tuple[int, ...] = ()
    v_sp: np.ndarray | None = None

    def __post_init__(self) -> None:
        """Validate the HELM problem dimensions and bus types."""

        self.Y = np.asarray(
            self.Y,
            dtype=np.complex128,
        )

        self.e_sp = np.asarray(
            self.e_sp,
            dtype=np.complex128,
        )

        if self.Y.ndim != 2:
            raise ValueError(
                "Y must be a two-dimensional matrix."
            )

        if self.Y.shape[0] != self.Y.shape[1]:
            raise ValueError(
                "Y must be square."
            )

        n_bus = self.Y.shape[0]

        if self.e_sp.shape != (n_bus,):
            raise ValueError(
                "e_sp must contain exactly one value per bus."
            )

        if not 0 <= self.B_s < n_bus:
            raise ValueError(
                f"Invalid slack bus {self.B_s} "
                f"for a {n_bus}-bus system."
            )

        # Convert bus collections to immutable integer tuples.
        self.B_v = tuple(int(bus) for bus in self.B_v)
        self.B_q = tuple(int(bus) for bus in self.B_q)

        # Validate PV bus indices.
        for bus in self.B_v:
            if not 0 <= bus < n_bus:
                raise ValueError(
                    f"Invalid PV bus {bus} "
                    f"for a {n_bus}-bus system."
                )

        # Validate PQ bus indices.
        for bus in self.B_q:
            if not 0 <= bus < n_bus:
                raise ValueError(
                    f"Invalid PQ bus {bus} "
                    f"for a {n_bus}-bus system."
                )

        if self.B_s in self.B_v:
            raise ValueError(
                "The slack bus must not also be listed as a PV bus."
            )

        if self.B_s in self.B_q:
            raise ValueError(
                "The slack bus must not also be listed as a PQ bus."
            )

        if set(self.B_v) & set(self.B_q):
            raise ValueError(
                "A bus cannot simultaneously be PV and PQ."
            )

        # If voltage setpoints are not supplied, create an array
        # containing NaN for every bus.
        if self.v_sp is None:
            self.v_sp = np.full(
                n_bus,
                np.nan,
                dtype=float,
            )
        else:
            self.v_sp = np.asarray(
                self.v_sp,
                dtype=float,
            )

            if self.v_sp.shape != (n_bus,):
                raise ValueError(
                    "v_sp must contain exactly "
                    "one value per bus."
                )

        # The slack voltage magnitude is always known.
        self.v_sp[self.B_s] = abs(
            self.v_s
        )


def make_helm_problem(
    Y: np.ndarray,
    e_sp: np.ndarray,
    B_s: int,
    v_s: complex = 1.0 + 0.0j,
    B_v: tuple[int, ...] = (),
    B_q: tuple[int, ...] = (),
    v_sp: np.ndarray | None = None,
) -> HelmProblem:
    """
    Construct a validated HELM problem.
    """

    return HelmProblem(
        Y=Y,
        e_sp=e_sp,
        v_s=v_s,
        B_s=B_s,
        B_v=B_v,
        B_q=B_q,
        v_sp=v_sp,
    )


def power_injection_from_voltage(
    Y: np.ndarray,
    v: np.ndarray,
) -> np.ndarray:
    """
    Calculate the complex bus power injections e (Eq. 8, Algorithm 1):

        e_i = v_i * sum_j conj(Y_ij) conj(v_j) = v * conj(Y v)
    """

    Y = np.asarray(
        Y,
        dtype=np.complex128,
    )

    v = np.asarray(
        v,
        dtype=np.complex128,
    )

    return v * np.conjugate(Y @ v)


def voltage_magnitudes(
    v: np.ndarray,
) -> np.ndarray:
    """Return |v_i|."""

    return np.abs(
        np.asarray(
            v,
            dtype=np.complex128,
        )
    )


def voltage_angles(
    v: np.ndarray,
) -> np.ndarray:
    """Return theta_i (radians)."""

    return np.angle(
        np.asarray(
            v,
            dtype=np.complex128,
        )
    )
