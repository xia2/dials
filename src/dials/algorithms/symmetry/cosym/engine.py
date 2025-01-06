"""LBFGS refinement engine for cosym analysis."""

from __future__ import annotations

import logging

import scipy.optimize

import scitbx.lbfgs
from scitbx.array_family import flex

logger = logging.getLogger(__name__)


class lbfgs_with_curvs:
    """Minimise a target function using the LBFGS minimiser.

    Implementation of an LBFGS minimiser using curvature information, according
    to the interface defined by :mod:`scitbx.lbfgs`.
    """

    def __init__(self, target, coords, use_curvatures=True, termination_params=None):
        """Initialise an lbfgs_with_curvs object.

        Args:
          target (dials.algorithms.target.Target): The target function to minimise.
          coords (np.ndarray): The starting coordinates for minimisation.
          use_curvatures (bool): Whether or not to use curvature information in the
            minimisation. Defaults to True.
          termination_params (scitbx.lbfgs.termination_parameters):
            Override the default termination parameters for the minimisation.
        """
        self.target = target

        self.x = flex.double(coords)
        self.f = None
        self.g = None

        if use_curvatures:
            self.diag_mode = "always"
        else:
            self.diag_mode = None

        self.minimizer = scitbx.lbfgs.run(
            target_evaluator=self, termination_params=termination_params
        )
        self.coords = self.x.as_numpy_array()

    def compute_functional_gradients_diag(self):
        """Compute the functional, gradients and diagonal.

        Returns:
          tuple: A tuple of the functional, gradients and diagonal, where the
          diagonal is the reciprocal of the curvatures.
        """
        f, g, curvs = self.compute_functional_gradients_and_curvatures()

        # Curvatures of zero will cause a crash, because their inverse is taken.
        if not curvs.all_gt(0.0):
            raise RuntimeError(
                "Zero curvature(s) encountered during minimization, minimization stopped."
            )
        diags = 1.0 / curvs
        return f, g, diags

    def compute_functional_gradients_and_curvatures(self):
        """Compute the functional, gradients and curvatures.

        Returns:
          tuple: A tuple of the functional, gradients and curvatures.
        """
        x = self.x.as_numpy_array()
        self.f = self.target.compute_functional(x)
        self.g = self.target.compute_gradients(x)
        self.c = self.target.curvatures(x)
        return self.f, flex.double(self.g), flex.double(self.c)

    def compute_functional_and_gradients(self):
        """Compute the functional and gradients.

        Returns:
          tuple: A tuple of the functional and gradients.
        """
        x = self.x.as_numpy_array()
        self.f = self.target.compute_functional(x)
        self.g = self.target.compute_gradients(x)
        return self.f, flex.double(self.g)

    def callback_after_step(self, minimizer):
        """Log progress after each successful step of the minimisation."""
        logger.debug("minimization step: f, iter, nfun:")
        logger.debug(f"{self.f} {minimizer.iter()} {minimizer.nfun()}")


def minimize_scitbx_lbfgs(
    target, coords, use_curvatures=True, max_iterations=100, max_calls=None
):
    termination_params = scitbx.lbfgs.termination_parameters(
        max_iterations=max_iterations,
        max_calls=max_calls,
        traditional_convergence_test=True,
        traditional_convergence_test_eps=0.001,
        drop_convergence_test_n_test_points=5,
        drop_convergence_test_max_drop_eps=1e-7,
        drop_convergence_test_iteration_coefficient=2,
    )
    result = lbfgs_with_curvs(
        target,
        coords,
        use_curvatures=use_curvatures,
        termination_params=termination_params,
    )
    return scipy.optimize.OptimizeResult(
        fun=result.f, jac=result.g, x=result.coords, nfev=result.minimizer.nfun()
    )


def minimize_scipy(
    target, coords, method="L-BFGS-B", max_iterations=None, max_calls=None
):
    """Thin wrapper around scipy.optimize.minimize.

    Args:
      target (dials.algorithms.target.Target): The target function to minimise.
      coords (np.array): The starting coordinates for
        minimisation.
    """
    options = {}
    if max_iterations:
        options.update(maxiter=max_iterations)
    if max_calls:
        options.update(maxfun=max_calls)
    return scipy.optimize.minimize(
        fun=target.compute_functional,
        x0=coords,
        jac=target.compute_gradients,
        method=method,
        options=options,
    )
