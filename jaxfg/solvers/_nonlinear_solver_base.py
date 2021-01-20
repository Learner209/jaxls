import abc
import dataclasses
from typing import TYPE_CHECKING

import jax
from jax import numpy as jnp

from .. import utils
from ..core._variable_assignments import VariableAssignments

if TYPE_CHECKING:
    from ..core._prepared_factor_graph import PreparedFactorGraph


@utils.register_dataclass_pytree
@dataclasses.dataclass(frozen=True)
class _NonlinearSolverBase:
    # For why we have two classes:
    # https://github.com/python/mypy/issues/5374#issuecomment-650656381

    """Nonlinear solver interface. """

    max_iterations: int = 100
    """Maximum number of iterations."""

    verbose: bool = True
    """Set to `True` to enable printing."""

    def _print(self, *args, **kwargs):
        """Prefixed printing helper. No-op if `verbose` is set to `False`."""
        if self.verbose:
            print(f"[{type(self).__name__}]", *args, **kwargs)


class NonlinearSolverBase(_NonlinearSolverBase, abc.ABC):
    @abc.abstractmethod
    def solve(
        self,
        graph: "PreparedFactorGraph",
        initial_assignments: "VariableAssignments",
    ) -> "VariableAssignments":
        """Run MAP inference on a factor graph."""


@utils.register_dataclass_pytree
@dataclasses.dataclass(frozen=True)
class _NonlinearSolverState:
    """Standard state passed between nonlinear solve iterations."""

    iterations: int
    assignments: "VariableAssignments"
    cost: float
    residual_vector: jnp.ndarray
    done: bool


@dataclasses.dataclass(frozen=True)
class _InexactStepSolverMixin:
    """Mixin for inexact Newton steps. Currently used by all solvers."""

    inexact_step_eta: float = 1e-1
    """Forcing sequence parameter for inexact Newton steps. CG tolerance is set to
    `eta / iteration #`.

    For reference, see AN INEXACT LEVENBERG-MARQUARDT METHOD FOR LARGE SPARSE NONLINEAR
    LEAST SQUARES, Wright & Holt 1983."""

    @jax.jit
    def inexact_step_forcing_sequence(self, iterations: int) -> float:
        """Get CGLS tolerance from zero-indexed iteration count."""
        return self.inexact_step_eta / (iterations + 1)


@dataclasses.dataclass(frozen=True)
class _TerminationCriteriaMixin:
    """Mixin for Ceres-style termination criteria."""

    cost_tolerance: float = 1e-5
    """We terminate if `|cost change| / cost < cost_tolerance`."""

    gradient_tolerance: float = 1e-9
    """We terminate if `norm_inf(x - rplus(x, linear delta)) < gradient_tolerance`."""

    gradient_tolerance_start_step: int = 10
    """When to start checking the gradient tolerance condition. Helps solve precision
    issues caused by inexact Newton steps."""

    parameter_tolerance: float = 1e-7
    """We terminate if `norm_2(linear delta) < (norm2(x) + parameter_tolerance) * parameter_tolerance`."""

    @jax.jit
    def check_convergence(
        self,
        state_prev: _NonlinearSolverState,
        cost_updated: float,
        local_delta_assignments: VariableAssignments,
        negative_gradient: jnp.ndarray,
    ):
        """Check for convergence!"""
        # Cost tolerance
        converged_cost = (
            jnp.abs(cost_updated - state_prev.cost) / state_prev.cost
            < self.cost_tolerance
        )

        # Gradient tolerance
        converged_gradient = jnp.where(
            state_prev.iterations >= self.gradient_tolerance_start_step,
            jnp.max(
                state_prev.assignments.storage
                - state_prev.assignments.apply_local_deltas(
                    VariableAssignments(
                        storage=negative_gradient,
                        storage_metadata=local_delta_assignments.storage_metadata,
                    ),
                ).storage
            )
            < self.gradient_tolerance,
            False,
        )

        # Parameter tolerance
        converged_parameters = (
            jnp.linalg.norm(jnp.abs(local_delta_assignments.storage))
            < (
                jnp.linalg.norm(state_prev.assignments.storage)
                + self.parameter_tolerance
            )
            * self.parameter_tolerance
        )

        return jnp.logical_or(
            converged_cost,
            jnp.logical_or(
                converged_gradient,
                converged_parameters,
            ),
        )
