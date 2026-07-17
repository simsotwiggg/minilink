#!/usr/bin/env python3
"""Simple discrete-time linear MPC example using CVXPY + OSQP.

Install: pip install numpy cvxpy osqp matplotlib

System: double integrator (position, velocity)
x_{k+1} = A x_k + B u_k
Minimize sum (x-x_ref)'Q(x-x_ref) + u'R u with simple box constraints on u.
"""

import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np


def mpc_example():
    dt = 0.1
    A = np.array([[1.0, dt], [0.0, 1.0]])
    B = np.array([[0.5 * dt**2], [dt]])

    nx, nu = 2, 1
    N = 20  # horizon

    Q = np.diag([10.0, 1.0])  # state cost
    R = np.array([[0.1]])  # control cost

    x0 = np.array([5.0, 0.0])  # initial state: pos=5, vel=0
    x_ref = np.array([0.0, 0.0])

    u_max = 1.0
    u_min = -1.0

    # decision variables
    x = cp.Variable((nx, N + 1))
    u = cp.Variable((nu, N))

    cost = 0
    constraints = []

    constraints += [x[:, 0] == x0]

    for k in range(N):
        cost += cp.quad_form(x[:, k] - x_ref, Q) + cp.quad_form(u[:, k], R)
        constraints += [x[:, k + 1] == A @ x[:, k] + B @ u[:, k]]
        constraints += [u[:, k] <= u_max, u[:, k] >= u_min]

    # terminal cost
    cost += cp.quad_form(x[:, N] - x_ref, Q)

    prob = cp.Problem(cp.Minimize(cost), constraints)

    # solve with OSQP (fast for QPs). warm_start=True can speed repeated solves.
    prob.solve(solver=cp.OSQP, warm_start=True, verbose=False)

    if prob.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
        print("Solver did not converge: status=", prob.status)
        return

    u_opt = np.array(u.value).reshape(-1)
    x_opt = np.array(x.value)

    print("Optimal cost:", prob.value)
    print("First control action:", u_opt[0])

    t = np.arange(N + 1) * dt

    plt.figure(figsize=(6, 4))
    plt.subplot(2, 1, 1)
    plt.plot(t, x_opt[0, :], "-o", label="position")
    plt.plot(t, x_opt[1, :], "-o", label="velocity")
    plt.ylabel("state")
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.step(t[:-1], u_opt, where="post")
    plt.ylabel("control (u)")
    plt.ylim([u_min * 1.2, u_max * 1.2])
    plt.xlabel("time [s]")
    plt.grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    mpc_example()
