"""
Blending process: nonlinear and linearized dynamic simulation.

Nonlinear CSTR-style blending tank with two inlet streams and one outlet.
Includes:
- Nonlinear dynamic model
- Step change in inlet flow rate w1
- Linearization around steady state
- Comparison between nonlinear and linearized response

Based on original example by:
  Martha Grover (MATLAB, 2017)
  John Hedengren (Python, 2017)
"""

from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

RHO: float = 1000.0  # Density [kg/m^3]

# Mass fraction of species A in the inlet streams (assumed constant)
X1: float = 0.10
X2: float = 0.00


# ---------------------------------------------------------------------------
# Nonlinear model
# ---------------------------------------------------------------------------

def blending(
    t: float,
    z: Sequence[float],
    u: Sequence[float],
) -> list[float]:
    """
    Nonlinear dynamic model of the blending process.

    States
    ------
    V : float
        Volume in the tank [m^3]
    x : float
        Mass fraction of species A in the tank [-]

    Inputs
    ------
    w1 : float
        Mass flow rate, inlet stream 1 [kg/s]
    w2 : float
        Mass flow rate, inlet stream 2 [kg/s]
    w : float
        Mass flow rate, outlet [kg/s]

    Returns
    -------
    [dVdt, dxdt] : list[float]
        Time derivatives of the states.
    """
    V, x = z
    w1, w2, w = u

    dVdt = (w1 + w2 - w) / RHO
    dxdt = (w1 * (X1 - x) + w2 * (X2 - x)) / (RHO * V)

    return [dVdt, dxdt]


# ---------------------------------------------------------------------------
# Linearized model (state-space with deviation variables)
# ---------------------------------------------------------------------------

def blending_linear(
    t: float,
    x: Sequence[float],
    u: Sequence[float],
    A: np.ndarray,
    B: np.ndarray,
) -> np.ndarray:
    """
    Linearized state-space model in deviation variables.

    States (deviation variables)
    ----------------------------
    V' : float
        Deviation in volume [m^3]
    x' : float
        Deviation in mass fraction [-]

    Inputs (deviation variables)
    ----------------------------
    w1', w2', w'

    Model
    -----
    dx'/dt = A x' + B u'
    """
    x_vec = np.asarray(x, dtype=float)
    u_vec = np.asarray(u, dtype=float)
    dxdt = A @ x_vec + B @ u_vec
    return dxdt


# ---------------------------------------------------------------------------
# Helper to run a simulation with solve_ivp
# ---------------------------------------------------------------------------

def simulate_nonlinear(
    z0: Sequence[float],
    u: Sequence[float],
    tf: float,
    n_points: int = 200,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate the nonlinear blending model with constant inputs.

    Parameters
    ----------
    z0 : sequence of float
        Initial state [V0, x0]
    u : sequence of float
        Inputs [w1, w2, w]
    tf : float
        Final simulation time [s]
    n_points : int, optional
        Number of points in the time grid.

    Returns
    -------
    t : np.ndarray
        Time vector [s]
    z : np.ndarray
        State trajectories, shape (n_points, 2)
    """
    t_eval = np.linspace(0.0, tf, n_points)

    sol = solve_ivp(
        fun=lambda t, z: blending(t, z, u),
        t_span=(0.0, tf),
        y0=np.asarray(z0, dtype=float),
        t_eval=t_eval,
        vectorized=False,
    )

    if not sol.success:
        raise RuntimeError(f"Nonlinear simulation failed: {sol.message}")

    return sol.t, sol.y.T


def simulate_linear(
    z0_dev: Sequence[float],
    u_dev: Sequence[float],
    A: np.ndarray,
    B: np.ndarray,
    tf: float,
    n_points: int = 200,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate the linearized blending model with constant deviation inputs.

    Parameters
    ----------
    z0_dev : sequence of float
        Initial deviation state [V', x']
    u_dev : sequence of float
        Deviation inputs [w1', w2', w']
    A, B : np.ndarray
        State-space matrices of the linearized system
    tf : float
        Final simulation time [s]
    n_points : int, optional
        Number of points in the time grid.

    Returns
    -------
    t : np.ndarray
        Time vector [s]
    z_dev : np.ndarray
        Deviation state trajectories, shape (n_points, 2)
    """
    t_eval = np.linspace(0.0, tf, n_points)

    sol = solve_ivp(
        fun=lambda t, x: blending_linear(t, x, u_dev, A, B),
        t_span=(0.0, tf),
        y0=np.asarray(z0_dev, dtype=float),
        t_eval=t_eval,
        vectorized=False,
    )

    if not sol.success:
        raise RuntimeError(f"Linear simulation failed: {sol.message}")

    return sol.t, sol.y.T


# ---------------------------------------------------------------------------
# Main script
# ---------------------------------------------------------------------------

def main() -> None:
    # Initial conditions
    V0: float = 10.0      # Initial volume [m^3]
    x0: float = 0.025     # Initial mass fraction of A [-]
    z0 = np.array([V0, x0], dtype=float)

    # Nominal (steady-state) flows
    w1_nom: float = 1.0   # [kg/s]
    w2_nom: float = 3.0   # [kg/s]
    w_nom: float = w1_nom + w2_nom  # Outlet flow [kg/s]
    u_nom = np.array([w1_nom, w2_nom, w_nom], dtype=float)

    # Final simulation time
    tf: float = 3600.0  # [s]

    # ------------------------------------------------------------------
    # 1) Nonlinear simulation under nominal (steady) inputs
    # ------------------------------------------------------------------
    t_nom, z_nom = simulate_nonlinear(z0=z0, u=u_nom, tf=tf, n_points=200)

    # Steady-state composition (should match x0 for chosen initial conditions)
    x_bar = (w1_nom * X1 + w2_nom * X2) / (w1_nom + w2_nom)
    V_bar = V0  # We choose the operating point volume equal to initial volume

    # ------------------------------------------------------------------
    # 2) Nonlinear simulation with a step in w1 (w kept fixed)
    # ------------------------------------------------------------------
    dw1: float = 0.1
    u_step = u_nom.copy()
    u_step[0] += dw1  # step in w1 only; w remains at the nominal value

    t_step, z_step = simulate_nonlinear(z0=z0, u=u_step, tf=tf, n_points=200)

    # ------------------------------------------------------------------
    # 3) Linearization (state-space, deviation variables)
    # ------------------------------------------------------------------
    # A, B, C, D evaluated at (V_bar, x_bar, u_nom)
    A = np.array(
        [
            [0.0, 0.0],
            [0.0, (-w1_nom - w2_nom) / (RHO * V_bar)],
        ],
        dtype=float,
    )

    B = np.array(
        [
            [1.0 / RHO, 1.0 / RHO, -1.0 / RHO],
            [(X1 - x_bar) / (RHO * V_bar),
             (X2 - x_bar) / (RHO * V_bar),
             0.0],
        ],
        dtype=float,
    )

    # Measurement matrices (here just for completeness)
    # y' = C x' + D u'
    C = np.array(
        [
            [0.0, 0.0],  # y1' = w' (direct from input via D below)
            [1.0, 0.0],  # y2' = V'
            [0.0, 1.0],  # y3' = x'
        ],
        dtype=float,
    )

    D = np.array(
        [
            [1.0, 0.0, 0.0],  # w' measured directly
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ],
        dtype=float,
    )

    # Deviation variables for the step in w1
    u_dev = u_step - u_nom
    z_bar = np.array([V_bar, x_bar], dtype=float)
    z0_dev = z0 - z_bar  # should be [0, 0] for this setup

    t_lin, z_dev = simulate_linear(
        z0_dev=z0_dev,
        u_dev=u_dev,
        A=A,
        B=B,
        tf=tf,
        n_points=200,
    )

    # Reconstruct actual states from deviation variables
    z_lin = z_dev + z_bar

    # ------------------------------------------------------------------
    # 4) Optional: steady-state gain (using pseudo-inverse of -A)
    # ------------------------------------------------------------------
    # Note: A is singular (no unique steady state for V), so we use pinv.
    K = C @ np.linalg.pinv(-A) @ B + D

    # ------------------------------------------------------------------
    # 5) Compute measured outputs y for the linear model (deviation form)
    # ------------------------------------------------------------------
    # y' = C x' + D u'
    y_dev = np.zeros((C.shape[0], t_lin.size))
    for i in range(t_lin.size):
        y_dev[:, i] = C @ z_dev[i, :] + D @ u_dev

    # ------------------------------------------------------------------
    # 6) Plot results
    # ------------------------------------------------------------------
    # (a) Nonlinear – steady operation
    plt.figure(figsize=(8, 6))
    plt.subplot(2, 1, 1)
    plt.plot(t_nom, z_nom[:, 0])
    plt.xlabel("time t [s]")
    plt.ylabel(r"$V$ [m$^3$]")
    plt.title("Blending process under constant inputs (nonlinear model)")

    plt.subplot(2, 1, 2)
    plt.plot(t_nom, z_nom[:, 1])
    plt.xlabel("time t [s]")
    plt.ylabel("x [-]")
    plt.tight_layout()

    # (b) Nonlinear – step in w1
    plt.figure(figsize=(8, 6))
    plt.subplot(2, 1, 1)
    plt.plot(t_step, z_step[:, 0])
    plt.xlabel("time t [s]")
    plt.ylabel(r"$V$ [m$^3$]")
    plt.title(r"Blending process under a step input to $w_1$ (nonlinear)")

    plt.subplot(2, 1, 2)
    plt.plot(t_step, z_step[:, 1])
    plt.xlabel("time t [s]")
    plt.ylabel("x [-]")
    plt.tight_layout()

    # (c) Comparison nonlinear vs linearized – step in w1
    plt.figure(figsize=(8, 6))
    plt.subplot(2, 1, 1)
    plt.plot(t_step, z_step[:, 0], "r-", linewidth=2, label="Nonlinear")
    plt.plot(t_lin, z_lin[:, 0], "b--", linewidth=2, label="Linearized")
    plt.xlabel("time t [s]")
    plt.ylabel(r"$V$ [m$^3$]")
    plt.title(
        r"Nonlinear vs linearized model for a step change in $w_1$"
    )
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(t_step, z_step[:, 1], "k-", linewidth=2, label="Nonlinear")
    plt.plot(t_lin, z_lin[:, 1], "g:", linewidth=2, label="Linearized")
    plt.xlabel("time t [s]")
    plt.ylabel("mass fraction x [-]")
    plt.legend()
    plt.tight_layout()

    plt.show()

    # Just to avoid unused-variable warnings in some linters:
    _ = K, y_dev


if __name__ == "__main__":
    main()
