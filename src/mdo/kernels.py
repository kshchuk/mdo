"""Kernels of the method of discrete singularities.

Velocity of a unit point vortex with regularisation R = max(δ, r), the summed velocity of many vortices,
and the circular arctangent θ = arg(z − ω)/2π ∈ [0, 1).
"""
import numpy as np


def vortex_velocity(x, y, x0j, y0j, delta=0.0):
    """(u, v) induced at (x, y) by a vortex of unit strength at (x0j, y0j), with R ≥ delta."""
    dx = x - x0j
    dy = y - y0j
    R2 = np.maximum(dx**2 + dy**2, delta**2)
    return -dy / (2 * np.pi * R2), dx / (2 * np.pi * R2)


def induced(x, y, xs, ys, gs, delta, chunk=4000):
    """Total velocity at points (x, y) induced by vortices (xs, ys) with strengths gs.

    x, y may have any shape (the result has the same shape). Points are processed in chunks so that the
    "point × vortex" table does not exhaust memory.
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    shape = x.shape
    xf, yf = x.ravel(), y.ravel()
    xs, ys, gs = (np.asarray(a, float).ravel() for a in (xs, ys, gs))
    u = np.zeros(len(xf))
    v = np.zeros(len(xf))
    if len(gs):
        for s in range(0, len(xf), chunk):
            U, V = vortex_velocity(xf[s:s + chunk, None], yf[s:s + chunk, None],
                                   xs[None, :], ys[None, :], delta)
            u[s:s + chunk] = U @ gs
            v[s:s + chunk] = V @ gs
    return u.reshape(shape), v.reshape(shape)


def theta(x, y, xs, ys):
    """Circular arctangent arg(z − ω)/2π in [0, 1); the branch cut goes to the right of ω."""
    return np.mod(np.arctan2(y - ys, x - xs), 2 * np.pi) / (2 * np.pi)
