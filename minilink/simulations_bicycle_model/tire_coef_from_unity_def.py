import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import differential_evolution, least_squares

# ------------------------------------------------------------
# VPP data
# ------------------------------------------------------------
# x_data = longitudinal slip speed [m/s]
# mu_data = friction coefficient [-]

x_data = np.array([0.0, 1.37, 3.0, 6.0])
mu_data = np.array([0.0, 1.00, 1.25, 1.05])

# ------------------------------------------------------------
# Reference vehicle speed
# ------------------------------------------------------------

V_vehicle = 20.0  # [m/s]

# ------------------------------------------------------------
# Convert slip speed to longitudinal tire slip κ
# ------------------------------------------------------------

kappa_data = x_data / V_vehicle

print("Converted data:")
for x, kappa, mu in zip(x_data, kappa_data, mu_data):
    print(f"slip speed = {x:5.2f} m/s | kappa = {kappa:7.4f} | mu = {mu:.3f}")

# ------------------------------------------------------------
# Longitudinal Pacejka model
# ------------------------------------------------------------
# μ(κ) = D * sin(C * atan(Bκ - E(Bκ - atan(Bκ))))


def pacejka_mu_kappa(kappa, B, C, D, E):
    return D * np.sin(C * np.arctan(B * kappa - E * (B * kappa - np.arctan(B * kappa))))


# ------------------------------------------------------------
# Residuals
# ------------------------------------------------------------
# This is where the fitting happens.
# IMPORTANT: use kappa_data, not x_data.


def residuals(params):
    B, C, D, E = params
    mu_fit = pacejka_mu_kappa(kappa_data, B, C, D, E)
    return mu_fit - mu_data


# ------------------------------------------------------------
# Bounds
# ------------------------------------------------------------

lower_bounds = [0.001, 0.1, 0.1, -5.0]
upper_bounds = [500.0, 3.0, 3.0, 5.0]

bounds = list(zip(lower_bounds, upper_bounds))

# ------------------------------------------------------------
# Optimization
# ------------------------------------------------------------

global_result = differential_evolution(
    lambda p: np.sum(residuals(p) ** 2), bounds=bounds, seed=1, tol=1e-10
)

local_result = least_squares(
    residuals,
    global_result.x,
    bounds=(lower_bounds, upper_bounds),
    xtol=1e-13,
    ftol=1e-13,
    gtol=1e-13,
    max_nfev=100000,
)

B, C, D, E = local_result.x

print("\nFitted Pacejka coefficients:")
print(f"B = {B:.6f}")
print(f"C = {C:.6f}")
print(f"D = {D:.6f}")
print(f"E = {E:.6f}")

print("\nComparison:")
for x, kappa, mu_target in zip(x_data, kappa_data, mu_data):
    mu_fit = pacejka_mu_kappa(kappa, B, C, D, E)

    print(
        f"slip speed = {x:5.2f} m/s | "
        f"kappa = {kappa:7.4f} | "
        f"kappa = {100 * kappa:6.2f}% | "
        f"target mu = {mu_target:6.3f} | "
        f"fit mu = {mu_fit:6.3f}"
    )

# ------------------------------------------------------------
# Plot versus κ
# ------------------------------------------------------------

kappa_plot = np.linspace(0.0, 0.4, 300)
mu_plot = pacejka_mu_kappa(kappa_plot, B, C, D, E)

plt.figure()
plt.plot(100 * kappa_plot, mu_plot, label="Fitted Pacejka curve")
plt.scatter(100 * kappa_data, mu_data, color="red", label="VPP data converted to κ")
plt.xlabel("Longitudinal slip κ [%]")
plt.ylabel("Friction coefficient μ [-]")
plt.title("Longitudinal Pacejka Fit")
plt.grid(True)
plt.legend()
plt.show()

# ------------------------------------------------------------
# Optional: plot same model versus original slip speed
# ------------------------------------------------------------

x_plot = np.linspace(0.0, 8.0, 300)
kappa_from_speed = x_plot / V_vehicle
mu_from_speed = pacejka_mu_kappa(kappa_from_speed, B, C, D, E)

plt.figure()
plt.plot(x_plot, mu_from_speed, label="Fitted Pacejka curve")
plt.scatter(x_data, mu_data, color="red", label="Original VPP points")
plt.xlabel("Longitudinal slip speed [m/s]")
plt.ylabel("Friction coefficient μ [-]")
plt.title(f"Same Pacejka Fit Re-Plotted vs Slip Speed, V = {V_vehicle:.1f} m/s")
plt.grid(True)
plt.legend()
plt.show()
