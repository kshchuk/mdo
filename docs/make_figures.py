"""Regenerate the README figures: python docs/make_figures.py (takes about a minute)."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from mdo import Contour, Simulation, SteadySolution, plot_contour, plot_four, plot_wake  # noqa: E402

OUT = Path(__file__).parent / "images"
OUT.mkdir(exist_ok=True)


def save(name):
    plt.gcf().savefig(OUT / name, dpi=110, bbox_inches="tight")
    plt.close("all")


c = Contour.semicircle(80)
plot_contour(c, every=4)
save("contour.png")

sol = SteadySolution.solve(c, v_inf=(1, 0), gamma0=1.0)
plot_four(sol, xlim=(-1, 1), ylim=(-1, 1), delta=c.h / 2, title="Steady flow: M = 80, Γ0 = 1")
save("steady.png")

sim = Simulation(Contour.semicircle(24)).run(80, progress=False)
plot_wake(sim, [5, 20, 40, 80])
save("wake.png")
plot_four(sim, title=f"Unsteady flow, t = {sim.t:.0f}")
save("unsteady.png")

print("figures in", OUT)
