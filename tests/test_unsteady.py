"""Unsteady problem: step invariants and reproduction of known results."""
import numpy as np
import pytest

from mdo import Contour, Simulation

from test_contour import digit_five


def test_initial_state_is_lab1_with_zero_circulation():
    sim = Simulation(Contour.semicircle(24))
    assert abs(sim.G.sum()) < 1e-14


def test_short_run_invariants():
    """Kelvin, no-penetration, τ = δ/max|V|, no free vortex closer than δ to the arc."""
    c = Contour.semicircle(24)
    sim = Simulation(c).run(3.0, progress=False)
    assert sim.kelvin_residual() < 1e-12
    u, v = sim.velocity(c.xc, c.yc)
    assert np.max(np.abs(u * c.nx + v * c.ny)) < 1e-10
    cx, cy, R, *_ = c.arc
    on_arc = sim.fx <= cx
    assert np.all(np.abs(np.hypot(sim.fx - cx, sim.fy - cy)[on_arc] - R) >= c.delta - 1e-12)
    t, *_ = sim.forces()
    assert np.all(np.diff(np.r_[0, t]) <= c.delta / 1.0 + 1e-12)   # max|V| ≥ |V∞| = 1


def test_symmetric_until_breakdown():
    """At α = 0 the side force is zero until the symmetry breaks."""
    sim = Simulation(Contour.semicircle(24)).run(8.0, progress=False)
    t, Cx, Cy, _ = sim.forces()
    assert np.max(np.abs(Cy[t > 3])) < 1e-3                     # round-off only


def test_any_angle_removal_along_flow():
    """At α = 1 rad vortices are removed along the flow, so their number stays bounded."""
    a = 1.0
    sim = Simulation(Contour.semicircle(24), v_inf=(np.cos(a), np.sin(a)), x_max=3.0)
    sim.run(8.0, progress=False)
    proj = sim.fx * np.cos(a) + sim.fy * np.sin(a)
    assert np.all(proj <= 3.0)


def test_digit_five_with_absorption():
    """Digit "5" with 4 separation points and absorption in the cavity: Kelvin holds."""
    c = digit_five()
    sim = Simulation(c, absorb=((0.0, -0.13), 0.25)).run(3.0, progress=False)
    assert sim.kelvin_residual() < 1e-12
    assert len(sim.fg) > 0


@pytest.mark.slow
def test_reproduce_lab2():
    """Full semicircle run to t = 80: 4180 steps, Cx ≈ 1.36, St ≈ 0.25."""
    sim = Simulation(Contour.semicircle(24)).run(80.0, progress=False)
    assert abs(sim.n_steps - 4180) <= 50
    s = sim.summary()
    assert abs(s["Cx_mean"] - 1.36) < 0.15
    assert abs(s["St"] - 0.25) < 0.03
