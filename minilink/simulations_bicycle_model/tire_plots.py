import matplotlib.pyplot as plt
import numpy as np

from minilink.dynamics.catalog.vehicles.tire_models import (
    Pacejka,
)

tire = Pacejka()

Bt = 8.38
Ct = 1.61
Dt = 1.26
Et = -1.31

fitted_tire = Pacejka(
    Bx=Bt,
    Cx=Ct,
    Dx=Dt,
    Ex=Et,
)

# Vehicle parameters from your bicycle model
mass = 1500.0
gravity = 9.81
a = 1.0
b = 1.0
L = a + b

# Static normal load on front or rear tire
Fz_front = mass * gravity * b / L
Fz_rear = mass * gravity * a / L

# Slip angle range
alpha_rad = 0.0

# Pure lateral slip
kappa = np.linspace(-50.0, 50.0, 500)
kappa_rad = np.deg2rad(kappa)

# Compute forces
Fx, Fy = tire.slip2forces(alpha_rad, kappa_rad, Fz_front)

# Friction coefficient
mu_x = Fx / Fz_rear


Fx, Fy = fitted_tire.slip2forces(alpha_rad, kappa_rad, Fz_front)

# Friction coefficient
mu_x_fitted = Fx / Fz_rear

# Plot
plt.figure(figsize=(8, 5))
plt.plot(kappa, mu_x, linewidth=2, label="original tire")
plt.plot(kappa, mu_x_fitted, linewidth=2, label="fitted tire")

plt.axhline(0, color="black", linewidth=0.8)
plt.axvline(0, color="black", linewidth=0.8)
plt.grid(True)

plt.xlabel("Slip angle kappa [deg]")
plt.ylabel("Longitudinal friction coefficient mu_x = Fx / Fz_rear")
plt.title("Pacejka longitudinal friction coefficient vs slip angle")

plt.show()
