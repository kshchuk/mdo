# mdo — method of discrete singularities

A small Python library implementing the **method of discrete singularities** (MDS, Ukrainian «МДО»), also
known as the discrete vortex method, for 2D potential flow of an ideal incompressible fluid around a thin
obstacle (an open contour). It covers two problems from the course «Technologies of Numerical Modelling»
(Taras Shevchenko National University of Kyiv):

- **Lab 1 — steady flow** around a contour with a prescribed circulation Γ0: bound vortices on the contour,
  collocation points between them, a linear system for the vortex strengths, velocity, potential φ and
  stream function ψ fields.
- **Lab 2 — unsteady separated flow**: at every time step the end (and corner) vortices separate and become
  free vortices that are carried by the flow; the wake develops into a Kármán vortex street.

## Installation

The repository is private, so the machine needs GitHub access (SSH key or token).

```bash
pip install "git+ssh://git@github.com/kshchuk/mdo.git"
# requirements.txt / pyproject.toml:  mdo @ git+ssh://git@github.com/kshchuk/mdo.git
```

For development:

```bash
git clone git@github.com:kshchuk/mdo.git && cd mdo
pip install -e ".[test]"
pytest -m "not slow"     # fast tests (~3 s)
pytest -m slow           # full reproduction of Lab 2 (~2 min)
```

Requires Python ≥ 3.9, NumPy and Matplotlib.

## Quick start

```python
import numpy as np
from mdo import Contour, SteadySolution, Simulation, plot_contour, plot_four, plot_wake

# Lab 1: semicircle of size 1 ("C" variant), 80 vortices, circulation Γ0 = 1
sol = SteadySolution.solve(Contour.semicircle(80), v_inf=(1, 0), gamma0=1.0)
print(sol.G.sum(), sol.residual())          # Γ0 and the no-penetration residual (~1e-16)
plot_four(sol, delta=sol.contour.h / 2)     # V on |V|, |V|, φ (transformed formula), ψ

# Lab 2: vortex shedding and the Kármán street
sim = Simulation(Contour.semicircle(24)).run(80)
print(sim.summary())                        # mean drag coefficient Cx ≈ 1.36, Strouhal number St ≈ 0.25
plot_wake(sim, [5, 20, 40, 80])
plot_four(sim)

# Any flow angle and any contour
a = 1.0                                      # radians
sim = Simulation(Contour.semicircle(24), v_inf=(np.cos(a), np.sin(a))).run(40)
L_shape = Contour.polyline([(0.5, 0.5), (-0.5, 0.5), (-0.5, -0.5)], 31)   # separates at ends and corner
```

A complete, executed example is in [`examples/demo.ipynb`](examples/demo.ipynb).

## API overview

| Module | Contents |
|---|---|
| `contour.py` | `Contour`: circular arc / semicircle / polyline / arbitrary points; collocation points, normals, separation points, regularisation radius δ, `keep_out` (free vortices may not cross the contour) |
| `kernels.py` | velocity of a unit vortex with regularisation R = max(δ, r); summed velocity of many vortices (chunked); circular arctangent |
| `fields.py` | velocity, stream function ψ, potential φ (direct and "transformed" dipole form with a single branch cut), grids |
| `steady.py` | Lab 1: influence matrix, `solve_steady`, `SteadySolution` |
| `unsteady.py` | Lab 2: `Simulation` — separation, wake advection, time step, linear system with Kelvin's circulation condition, forces |
| `plotting.py` | contour discretisation, the four field plots, wake snapshots |

## Method in brief

- **Discretisation.** M bound vortices Γj at points ω0j along the contour and M − 1 collocation points
  between them (on the arc itself for a circular arc, chord midpoints otherwise) with unit normals nk.
- **Linear system.** No-penetration n·V = 0 at every collocation point plus one circulation equation:
  ΣΓj = Γ0 (steady case) or ΣΓj = −Σγi (unsteady case, Kelvin's theorem — the flow starts from rest).
- **Regularisation.** Vortex velocities use R = max(δ, r), δ = L/(2M) by default (L — contour length).
- **Time step (Lab 2).** At step n the vortices at the separation points become free with
  γ = Γp(tn) (Kutta–Joukowski hypothesis); all free vortices, including the new ones, move with the local
  velocity (explicit Euler); τn = δ / max|V| taken over the free vortices; vortices that cross the contour
  or come closer than δ are reflected back to their side; then the system is solved for Γj(tn+1).
- **Forces.** Cx, Cy are computed from the hydrodynamic impulse P = ρ Σ Γk (yk, −xk), F = −dP/dt
  (Wu, 1981). Pressure (Cp) is not implemented.
- **Cavities.** `Simulation(..., absorb=(centre, radius))` absorbs free vortices trapped inside a nearly
  closed cavity, transferring their circulation back to the contour (needed e.g. for a "5"-shaped contour).

## Validation

The test suite reproduces results checked in the course work:

- Lab 1 control example (segment, M = 5): vortex strengths and velocity/φ/ψ at a probe point;
- semicircle, M = 80: ΣΓ = Γ0 and no-penetration residual below 1e-12;
- flat plate: error versus the exact solution decreases as 1/M;
- circle with a 4° gap versus the exact cylinder solution: error < 1e-5 (Γ0 = 0);
- Lab 2 invariants: Kelvin's theorem to 1e-12, no penetration, no free vortex closer than δ to the contour;
- full Lab 2 run (`-m slow`): 4180 steps to t = 80, mean Cx ≈ 1.36, St ≈ 0.25.

## References

1. Dovhyi S.O., Trotsenko Ya.P., Chernii D.I. *Technologies of Numerical Modelling. Laboratory Practicum*
   (in Ukrainian). Kyiv, 2024.
2. Wu J.C. Theory for aerodynamic force and moment in viscous flows. *AIAA Journal*, 19(4):432–441, 1981.
3. Cottet G.-H., Koumoutsakos P. *Vortex Methods: Theory and Practice*. Cambridge University Press.
