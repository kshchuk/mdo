"""Steady flow around a contour with a prescribed circulation Γ0 (Lab 1)."""
from dataclasses import dataclass

import numpy as np

from . import fields
from .contour import Contour
from .kernels import vortex_velocity


def influence_matrix(contour: Contour):
    """System matrix: A_kj = n_k · V_j(x_k, y_k) for k < M, the last row is all ones.

    No regularisation is needed here: collocation points never coincide with vortices.
    """
    c = contour
    U, V = vortex_velocity(c.xc[:, None], c.yc[:, None], c.x0[None, :], c.y0[None, :])
    A = np.empty((c.M, c.M))
    A[:-1] = c.nx[:, None] * U + c.ny[:, None] * V
    A[-1] = 1.0
    return A


def solve_steady(contour: Contour, v_inf=(1.0, 0.0), gamma0=0.0):
    """Vortex strengths Γ1..ΓM: no-penetration at collocation points and ΣΓ = Γ0."""
    c = contour
    v_inf = np.asarray(v_inf, float)
    b = np.empty(c.M)
    b[:-1] = -(c.nx * v_inf[0] + c.ny * v_inf[1])
    b[-1] = gamma0
    return np.linalg.solve(influence_matrix(c), b)


@dataclass
class SteadySolution:
    """Steady solution: contour, free stream, Γ0 and the vortex strengths Γj; fields on any grid."""
    contour: Contour
    v_inf: np.ndarray
    gamma0: float
    G: np.ndarray

    @classmethod
    def solve(cls, contour, v_inf=(1.0, 0.0), gamma0=0.0):
        v_inf = np.asarray(v_inf, float)
        return cls(contour, v_inf, gamma0, solve_steady(contour, v_inf, gamma0))

    def _delta(self, delta):
        return self.contour.delta if delta is None else delta

    def velocity(self, X, Y, delta=None):
        """Velocity field with regularisation R = max(δ, r)."""
        c = self.contour
        return fields.velocity(X, Y, c.x0, c.y0, self.G, self.v_inf, self._delta(delta))

    def psi(self, X, Y, delta=None, transformed=False):
        """Stream function: direct sum of vortices or the transformed (dipole) form."""
        c = self.contour
        if transformed:
            return fields.chain_transformed(X, Y, c.x0, c.y0, self.G, self.v_inf,
                                            self._delta(delta))[1]
        return fields.psi(X, Y, c.x0, c.y0, self.G, self.v_inf, self._delta(delta))

    def phi(self, X, Y, delta=None, transformed=True):
        """Potential: transformed (single branch cut) or direct (one cut per vortex)."""
        c = self.contour
        if transformed:
            return fields.chain_transformed(X, Y, c.x0, c.y0, self.G, self.v_inf,
                                            self._delta(delta))[0]
        return fields.phi_direct(X, Y, c.x0, c.y0, self.G, self.v_inf)

    def residual(self):
        """max |V·n| at collocation points (no regularisation); should be ~1e-15."""
        c = self.contour
        u, v = fields.velocity(c.xc, c.yc, c.x0, c.y0, self.G, self.v_inf, 0.0)
        return np.max(np.abs(u * c.nx + v * c.ny))
