import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import differential_evolution, least_squares

# ------------------------------------------------------------
# Input data from the VPP slip-velocity curve
# ------------------------------------------------------------
# x = slip velocity [m/s]
# mu = tire friction coefficient [-]

x_data = np.array([0.0, 1.37, 3.0, 6.0])
mu_data = np.array([0.0, 1.00, 1.25, 1.05])

# ------------------------------------------------------------
# Vehicle reference speed
# ------------------------------------------------------------
# Needed to convert slip velocity into slip angle:
#
# alpha = atan(x / V)
#
# Change this value to the speed used for your tire model / VPP conversion.

V_vehicle = 20.0  # [m/s], example: 20 m/s = 72 km/h


# ------------------------------------------------------------
# Convert slip velocity to lateral slip angle
# ------------------------------------------------------------


def slip_velocity_to_angle_rad(x, V):
    return np.arctan(x / V)


def slip_velocity_to_angle_deg(x, V):
    return np.degrees(slip_velocity_to_angle_rad(x, V))


alpha_data_rad = slip_velocity_to_angle_rad(x_data, V_vehicle)
alpha_data_deg = slip_velocity_to_angle_deg(x_data, V_vehicle)


# ------------------------------------------------------------
# Simplified Pacejka / Magic Formula model
# ------------------------------------------------------------
# mu(x) = D * sin(C * atan(B*x - E*(B*x - atan(B*x))))
#
# Here x is slip velocity [m/s].
#
# B = stiffness factor
# C = shape factor
# D = peak/friction scale
# E = curvature factor


def pacejka_mu(x, B, C, D, E):
    return D * np.sin(C * np.arctan(B * x - E * (B * x - np.arctan(B * x))))


# ------------------------------------------------------------
# Residual function for curve fitting
# ------------------------------------------------------------


def residuals(params):
    B, C, D, E = params
    mu_fit = pacejka_mu(x_data, B, C, D, E)
    return mu_fit - mu_data


# ------------------------------------------------------------
# Parameter bounds
# ------------------------------------------------------------

lower_bounds = [0.001, 0.1, 0.1, -5.0]
upper_bounds = [20.0, 3.0, 3.0, 5.0]

bounds = list(zip(lower_bounds, upper_bounds))


# ------------------------------------------------------------
# Step 1: Global optimization
# ------------------------------------------------------------

global_result = differential_evolution(
    lambda p: np.sum(residuals(p) ** 2), bounds=bounds, seed=1, tol=1e-10
)


# ------------------------------------------------------------
# Step 2: Local least-squares refinement
# ------------------------------------------------------------

local_result = least_squares(
    residuals,
    global_result.x,
    bounds=(lower_bounds, upper_bounds),
    xtol=1e-13,
    ftol=1e-13,
    gtol=1e-13,
    max_nfev=100000,
)


# ------------------------------------------------------------
# Extract fitted parameters
# ------------------------------------------------------------

B, C, D, E = local_result.x

print("Fitted Pacejka coefficients:")
print(f"B = {B:.6f}")
print(f"C = {C:.6f}")
print(f"D = {D:.6f}")
print(f"E = {E:.6f}")

print("\nComparison with target data:")
for x, alpha_deg, mu_target in zip(x_data, alpha_data_deg, mu_data):
    mu_fit = pacejka_mu(x, B, C, D, E)
    print(
        f"x = {x:5.2f} m/s | "
        f"alpha = {alpha_deg:6.2f} deg | "
        f"target mu = {mu_target:6.3f} | "
        f"fitted mu = {mu_fit:6.3f}"
    )


# ------------------------------------------------------------
# Plot versus slip velocity
# ------------------------------------------------------------

x_plot = np.linspace(0, 8, 300)
mu_plot = pacejka_mu(x_plot, B, C, D, E)

plt.figure()
plt.plot(x_plot, mu_plot, label="Fitted Pacejka curve")
plt.scatter(x_data, mu_data, color="red", label="Target VPP points")
plt.xlabel("Slip velocity x [m/s]")
plt.ylabel("Friction coefficient μ [-]")
plt.title("Pacejka Fit to Slip-Velocity Curve")
plt.grid(True)
plt.legend()
plt.show()


# ------------------------------------------------------------
# Plot versus lateral slip angle
# ------------------------------------------------------------

alpha_plot_deg = slip_velocity_to_angle_deg(x_plot, V_vehicle)

plt.figure()
plt.plot(alpha_plot_deg, mu_plot, label="Fitted Pacejka curve")
plt.scatter(alpha_data_deg, mu_data, color="red", label="Target VPP points")
plt.xlabel("Slip angle α [deg]")
plt.ylabel("Friction coefficient μ [-]")
plt.title(f"Pacejka Fit Relative to Slip Angle, V = {V_vehicle:.1f} m/s")
plt.grid(True)
plt.legend()
plt.show()
