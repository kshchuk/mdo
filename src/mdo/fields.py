"""Flow fields of a vortex system: velocity, stream function ψ, potential φ, helper grids."""
import numpy as np

from .kernels import induced, theta


def velocity(X, Y, xs, ys, gs, v_inf, delta):
    """Velocity: free stream + Σ γ · V(unit vortex)."""
    u, v = induced(X, Y, xs, ys, gs, delta)
    return v_inf[0] + u, v_inf[1] + v


def psi(X, Y, xs, ys, gs, v_inf, delta):
    """Stream function: y·u∞ − x·v∞ − Σ γ/(2π)·ln R, R = max(δ, r). Single-valued."""
    out = Y * v_inf[0] - X * v_inf[1]
    for x0, y0, g in zip(np.ravel(xs), np.ravel(ys), np.ravel(gs)):
        out = out - g / (2 * np.pi) * np.log(np.maximum(np.hypot(X - x0, Y - y0), delta))
    return out


def phi_direct(X, Y, xs, ys, gs, v_inf):
    """Potential by the direct formula: x·u∞ + y·v∞ + Σ γ θ — one branch cut per vortex."""
    out = X * v_inf[0] + Y * v_inf[1]
    for x0, y0, g in zip(np.ravel(xs), np.ravel(ys), np.ravel(gs)):
        out = out + g * theta(X, Y, x0, y0)
    return out


def chain_transformed(X, Y, x0, y0, G, v_inf, delta):
    """Transformed (dipole) formulas for the chain of contour vortices.

    The vortices are rewritten as dipoles at the segment midpoints with moments S_j = Γ1 + … + Γj plus a
    single vortex Γ0 at the last point, so the potential has a single branch cut (none if Γ0 = 0).
    Returns (φ, ψ).
    """
    S = np.cumsum(G)[:-1]
    ddx, ddy = np.diff(x0), np.diff(y0)
    xm, ym = (x0[:-1] + x0[1:]) / 2, (y0[:-1] + y0[1:]) / 2
    g0 = G.sum()
    RM = np.maximum(np.hypot(X - x0[-1], Y - y0[-1]), delta)
    phi = X * v_inf[0] + Y * v_inf[1] + g0 * theta(X, Y, x0[-1], y0[-1])
    ps = Y * v_inf[0] - X * v_inf[1] - g0 / (2 * np.pi) * np.log(RM)
    for j in range(len(S)):
        Xj, Yj = X - xm[j], Y - ym[j]
        R2 = np.maximum(Xj**2 + Yj**2, delta**2)
        phi = phi + S[j] / (2 * np.pi) * (ddy[j] * Xj - ddx[j] * Yj) / R2
        ps = ps - S[j] / (2 * np.pi) * (ddx[j] * Xj + ddy[j] * Yj) / R2
    return phi, ps


def flow_box(v_inf, back, ahead, half):
    """Bounding box of the strip from −back to ahead along the flow and ±half across it."""
    e = np.array([-back, ahead])[:, None] * np.asarray(v_inf)
    p = np.array([-v_inf[1], v_inf[0]]) * half
    pts = np.r_[e + p, e - p]
    return (pts[:, 0].min(), pts[:, 0].max()), (pts[:, 1].min(), pts[:, 1].max())


def grid(xlim, ylim, nx=300):
    """Uniform grid with the same spacing in x and y."""
    ny = max(2, int(round(nx * (ylim[1] - ylim[0]) / (xlim[1] - xlim[0]))))
    return np.meshgrid(np.linspace(*xlim, nx), np.linspace(*ylim, ny))
