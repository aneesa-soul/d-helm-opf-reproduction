"""
D-HELM physics-driven critic.

Paper basis
-----------
The critic accepts the operating state and policy action and evaluates
the physical operating point through the D-HELM equations.

For the 5-bus case:

    state dimension  = 8
    policy action    = 8
    HELM output      = 32

The paper's critic sequence is:

    1. assign action/state quantities
    2. calculate HELM coefficients
    3. reconstruct bus voltages and injections
    4. calculate generator/voltage/current quantities
    5. calculate constraint violations
    6. calculate reward

This first implementation deliberately separates the physical
quantities and violation calculation from the generator cost/reward
calculation because the paper does not provide the numerical generator
cost coefficients.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class CriticOutput:
    """
    Physical quantities returned by the D-HELM critic.

    All power quantities are expressed in p.u. unless otherwise stated.
    """

    voltage: np.ndarray
    voltage_magnitude: np.ndarray
    voltage_angle: np.ndarray

    bus_injection: np.ndarray

    generator_p: np.ndarray
    generator_q: np.ndarray

    line_current_from: Optional[np.ndarray]
    line_current_to: Optional[np.ndarray]

    violations: np.ndarray

    converged: bool
    series_order: int
    residual: float

    @property
    def total_violation(self) -> float:
        """
        Sum of normalized constraint violations.
        """
        return float(np.sum(self.violations))


class DHELMViolations:
    """
    Constraint-violation calculation following Eq. (6).

    For a bounded quantity x:

        z = eta * (
            max(0, x - xmax)
            + max(0, xmin - x)
        )

    The implementation is kept as a separate component because the
    normalization factors eta_n are system/configuration dependent.
    """

    @staticmethod
    def bounded(
        value: np.ndarray,
        lower: np.ndarray,
        upper: np.ndarray,
        eta: np.ndarray,
    ) -> np.ndarray:
        """
        Calculate Eq. (6) violations for bounded quantities.
        """

        value = np.asarray(value, dtype=float)
        lower = np.asarray(lower, dtype=float)
        upper = np.asarray(upper, dtype=float)
        eta = np.asarray(eta, dtype=float)

        if not (
            value.shape
            == lower.shape
            == upper.shape
            == eta.shape
        ):
            raise ValueError(
                "value, lower, upper, and eta must have "
                "identical shapes."
            )

        return eta * (
            np.maximum(0.0, value - upper)
            + np.maximum(0.0, lower - value)
        )


@dataclass(frozen=True)
class CriticLimits:
    """
    Physical operating limits required by the critic.

    Paper-derived:
        Voltage limits are [0.9, 1.1] p.u. for the examined systems.

    The remaining limits must be supplied from the implemented network
    or explicitly documented reproduction assumptions.
    """

    voltage_min: float = 0.90
    voltage_max: float = 1.10

    # These remain optional until the corresponding system limits are
    # explicitly configured.
    generator_p_min: Optional[np.ndarray] = None
    generator_p_max: Optional[np.ndarray] = None

    generator_q_min: Optional[np.ndarray] = None
    generator_q_max: Optional[np.ndarray] = None

    current_max_from: Optional[np.ndarray] = None
    current_max_to: Optional[np.ndarray] = None


class DHELMViolations5Bus:
    """
    IEEE 5-bus violation evaluator.

    This class currently implements the voltage constraints because the
    paper explicitly reports the voltage range [0.9, 1.1].

    Generator/reactive/current constraints are added only when their
    limits are explicitly configured.
    """

    def __init__(
        self,
        limits: CriticLimits,
        voltage_eta: float = 1.0,
    ) -> None:

        self.limits = limits
        self.voltage_eta = float(voltage_eta)

    def evaluate(
        self,
        voltage_magnitude: np.ndarray,
        generator_p: Optional[np.ndarray] = None,
        generator_q: Optional[np.ndarray] = None,
        line_current_from: Optional[np.ndarray] = None,
        line_current_to: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Return the complete currently configured violation vector.
        """

        voltage_magnitude = np.asarray(
            voltage_magnitude,
            dtype=float,
        )

        voltage_lower = np.full_like(
            voltage_magnitude,
            self.limits.voltage_min,
        )

        voltage_upper = np.full_like(
            voltage_magnitude,
            self.limits.voltage_max,
        )

        voltage_eta = np.full_like(
            voltage_magnitude,
            self.voltage_eta,
        )

        violations = [
            DHELMViolations.bounded(
                value=voltage_magnitude,
                lower=voltage_lower,
                upper=voltage_upper,
                eta=voltage_eta,
            )
        ]

        if generator_p is not None:
            if (
                self.limits.generator_p_min is not None
                and self.limits.generator_p_max is not None
            ):
                generator_p = np.asarray(
                    generator_p,
                    dtype=float,
                )

                p_eta = np.ones_like(generator_p)

                violations.append(
                    DHELMViolations.bounded(
                        value=generator_p,
                        lower=self.limits.generator_p_min,
                        upper=self.limits.generator_p_max,
                        eta=p_eta,
                    )
                )

        if generator_q is not None:
            if (
                self.limits.generator_q_min is not None
                and self.limits.generator_q_max is not None
            ):
                generator_q = np.asarray(
                    generator_q,
                    dtype=float,
                )

                q_eta = np.ones_like(generator_q)

                violations.append(
                    DHELMViolations.bounded(
                        value=generator_q,
                        lower=self.limits.generator_q_min,
                        upper=self.limits.generator_q_max,
                        eta=q_eta,
                    )
                )

        if line_current_from is not None:
            if self.limits.current_max_from is not None:
                line_current_from = np.asarray(
                    line_current_from,
                    dtype=float,
                )

                current_lower = -np.asarray(
                    self.limits.current_max_from,
                    dtype=float,
                )

                current_upper = np.asarray(
                    self.limits.current_max_from,
                    dtype=float,
                )

                violations.append(
                    DHELMViolations.bounded(
                        value=line_current_from,
                        lower=current_lower,
                        upper=current_upper,
                        eta=np.ones_like(line_current_from),
                    )
                )

        if line_current_to is not None:
            if self.limits.current_max_to is not None:
                line_current_to = np.asarray(
                    line_current_to,
                    dtype=float,
                )

                current_lower = -np.asarray(
                    self.limits.current_max_to,
                    dtype=float,
                )

                current_upper = np.asarray(
                    self.limits.current_max_to,
                    dtype=float,
                )

                violations.append(
                    DHELMViolations.bounded(
                        value=line_current_to,
                        lower=current_lower,
                        upper=current_upper,
                        eta=np.ones_like(line_current_to),
                    )
                )

        return np.concatenate(violations)


class DHELMPhysicCritic:
    """
    Interface for the D-HELM physics-driven critic.

    The actual HELM calculation is injected through `helm_solver`.
    This keeps the critic independent of the current solver API while
    we connect the already validated HELM implementation.

    The callable must eventually return a physical solution containing:

        voltage
        voltage_magnitude
        voltage_angle
        bus_injection
        generator_p
        generator_q
        line_current_from
        line_current_to
        converged
        series_order
        residual
    """

    STATE_DIM = 8
    ACTION_DIM = 8
    HELM_OUTPUT_DIM = 32

    def __init__(
        self,
        helm_solver,
        limits: Optional[CriticLimits] = None,
    ) -> None:

        self.helm_solver = helm_solver

        if limits is None:
            limits = CriticLimits()

        self.limits = limits
        self.violation_model = DHELMViolations5Bus(limits)

    def evaluate_physics(
        self,
        state: np.ndarray,
        action: np.ndarray,
    ) -> CriticOutput:
        """
        Evaluate one state/action pair through the D-HELM physics model.

        This method intentionally does not call pandapower.runpp().
        """

        state = np.asarray(state, dtype=float)
        action = np.asarray(action, dtype=float)

        if state.shape != (self.STATE_DIM,):
            raise ValueError(
                f"Expected state shape ({self.STATE_DIM},), "
                f"received {state.shape}."
            )

        if action.shape != (self.ACTION_DIM,):
            raise ValueError(
                f"Expected action shape ({self.ACTION_DIM},), "
                f"received {action.shape}."
            )

        # The exact call into HelmSolver will be connected next.
        #
        # We deliberately fail here rather than silently using an
        # incompatible solver API.
        raise NotImplementedError(
            "Connect evaluate_physics() to the validated HelmSolver "
            "interface before running the D-HELM critic."
        )
