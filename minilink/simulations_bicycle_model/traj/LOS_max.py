import math

import matplotlib.pyplot as plt
import numpy as np

from minilink.core.system import System

# ============================================================
# Utilitaires géométriques LOS
# ============================================================

_EPS = 1e-12


def wrap_pi(angle):
    """
    Ramène un angle dans [-pi, pi].
    Remplace simpleSpeedBoatSim._wrap_pi si non disponible.
    """
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def _iter_segments(path_xy, closed: bool):
    n = len(path_xy)

    if n < 2:
        return

    if closed:
        for i in range(n):
            j = (i + 1) % n
            yield i, j
    else:
        for i in range(n - 1):
            yield i, i + 1


def _closest_point_on_segment(px, py, x0, y0, x1, y1):
    vx, vy = x1 - x0, y1 - y0
    wx, wy = px - x0, py - y0

    den = vx * vx + vy * vy

    if den <= _EPS:
        return x0, y0, 0.0

    t = (wx * vx + wy * vy) / den
    t = max(0.0, min(1.0, t))

    qx = x0 + t * vx
    qy = y0 + t * vy

    return qx, qy, t


def project_on_path_with_t(path_xy, x, y, closed=True):
    """
    Projette le point (x, y) sur une polyligne.

    Retourne:
        idx, t, qx, qy, e_perp, psi_path
    """

    best = None

    for i, j in _iter_segments(path_xy, closed):
        x0, y0 = path_xy[i]
        x1, y1 = path_xy[j]

        qx, qy, t = _closest_point_on_segment(x, y, x0, y0, x1, y1)

        dx = x1 - x0
        dy = y1 - y0

        seg_len = math.hypot(dx, dy)

        if seg_len <= _EPS:
            continue

        tx = dx / seg_len
        ty = dy / seg_len

        nx = -ty
        ny = tx

        e_perp = (x - qx) * nx + (y - qy) * ny
        d2 = (x - qx) ** 2 + (y - qy) ** 2

        if best is None or d2 < best[0]:
            psi_path = math.atan2(ty, tx)
            best = (d2, i, t, qx, qy, e_perp, psi_path)

    if best is None:
        raise ValueError("Path invalide pour projection.")

    _, idx, t, qx, qy, e_perp, psi_path = best

    return idx, t, qx, qy, e_perp, psi_path


def path_total_length(path_xy, closed=True):
    L = 0.0

    for i, j in _iter_segments(path_xy, closed):
        x0, y0 = path_xy[i]
        x1, y1 = path_xy[j]

        L += math.hypot(x1 - x0, y1 - y0)

    return L


def point_ahead_along_path(path_xy, idx, t, ds, closed=True):
    """
    Avance de ds mètres le long de la polyligne à partir de la projection.
    """

    n = len(path_xy)

    if n < 2:
        raise ValueError("Path trop court.")

    if closed:
        Lt = path_total_length(path_xy, closed=True)

        if Lt > _EPS:
            ds = ds % Lt

    i = idx % n
    j = (i + 1) % n

    x0, y0 = path_xy[i]
    x1, y1 = path_xy[j]

    dx = x1 - x0
    dy = y1 - y0

    seg_len = math.hypot(dx, dy)

    if seg_len <= _EPS:
        i = (i + 1) % n
        j = (i + 1) % n

        x0, y0 = path_xy[i]
        x1, y1 = path_xy[j]

        dx = x1 - x0
        dy = y1 - y0

        seg_len = math.hypot(dx, dy)

    cx = x0 + t * dx
    cy = y0 + t * dy

    rem = (1.0 - t) * seg_len

    steps = 0
    max_steps = len(path_xy) + 2

    while ds > rem and steps < max_steps:
        ds -= rem

        i = (i + 1) % n
        j = (i + 1) % n

        x0, y0 = path_xy[i]
        x1, y1 = path_xy[j]

        dx = x1 - x0
        dy = y1 - y0

        seg_len = math.hypot(dx, dy)

        if seg_len <= _EPS:
            rem = 0.0
            steps += 1
            continue

        cx = x0
        cy = y0
        rem = seg_len

        steps += 1

        if not closed and i == n - 1:
            return x0, y0, i, 0.0, math.atan2(dy, dx)

    lam = 0.0 if seg_len <= _EPS else min(1.0, ds / max(_EPS, seg_len))

    ax = cx + lam * dx
    ay = cy + lam * dy
    psi_a = math.atan2(dy, dx)

    return float(ax), float(ay), int(i), float(lam), float(psi_a)


# ============================================================
# Contrôleur LOS
# ============================================================


class LOSController:
    def __init__(
        self,
        path_xy,
        vx_nom=1.0,
        Delta=4.0,
        zeta=0.7,
        omega_n=1.2,
        ct_yr_nom=1.0,
        ki=0.0,
        i_lim=10.0,
        control_point_ahead=4.0,
        closed=False,
    ):
        self.path = np.asarray(path_xy, dtype=float)

        self.vx_nom = float(vx_nom)
        self.Delta = float(Delta)

        self.zeta = float(zeta)
        self.omega_n = float(omega_n)
        self.ct_yr_nom = float(ct_yr_nom)

        self.ki = float(ki)
        self.i_lim = float(i_lim)
        self.int_e = 0.0

        self.control_point_ahead = float(control_point_ahead)
        self.closed = bool(closed)

        self.kp = self.ct_yr_nom * self.omega_n**2
        self.kd = max(
            0.0,
            2.0 * self.zeta * self.omega_n * self.ct_yr_nom - 1.0,
        )

        self.r_max = math.radians(60.0)

    def reset(self):
        self.int_e = 0.0

    def compute(self, x, y, psi, dt, r_meas=None, vx=None, vy=None):
        if self.control_point_ahead != 0.0:
            x_ctrl = x + self.control_point_ahead * math.cos(psi)
            y_ctrl = y + self.control_point_ahead * math.sin(psi)
        else:
            x_ctrl = x
            y_ctrl = y

        idx, t, qx, qy, e_perp, psi_path = project_on_path_with_t(
            self.path,
            x_ctrl,
            y_ctrl,
            closed=self.closed,
        )

        ax, ay, _, _, _ = point_ahead_along_path(
            self.path,
            idx,
            t,
            self.Delta,
            closed=self.closed,
        )

        chi_d = math.atan2(ay - y_ctrl, ax - x_ctrl)

        chi_meas = psi

        if vx is not None and vy is not None:
            beta = math.atan2(vy, max(1e-6, vx))
            chi_meas = wrap_pi(psi + beta)

        err_chi = wrap_pi(chi_d - chi_meas)

        if self.ki > 0.0:
            self.int_e += e_perp * dt
            self.int_e = max(-self.i_lim, min(self.i_lim, self.int_e))

        r_fb = float(r_meas) if r_meas is not None else 0.0

        r_ref = self.kp * err_chi - self.kd * r_fb
        r_ref = max(-self.r_max, min(self.r_max, r_ref))

        vx_ref = self.vx_nom

        info = {
            "e_perp": e_perp,
            "chi_d": chi_d,
            "idx": idx,
            "ax": ax,
            "ay": ay,
            "x_ctrl": x_ctrl,
            "y_ctrl": y_ctrl,
            "kp": self.kp,
            "kd": self.kd,
            "err_chi": err_chi,
        }

        return vx_ref, r_ref, info


# ============================================================
# System Minilink
# ============================================================


class Los(System):
    """
    Bloc Minilink qui encapsule le contrôleur LOS.

    Sortie:
        los[0] = vx_ref
        los[1] = r_ref
        los[2] = e_perp
        los[3] = chi_d
        los[4] = ax
        los[5] = ay
        los[6] = x_ctrl
        los[7] = y_ctrl
    """

    def __init__(
        self,
        path_pts,
        vx_nom=1.0,
        Delta=4.0,
        zeta=0.7,
        omega_n=1.2,
        ct_yr_nom=1.0,
        control_point_ahead=4.0,
        closed=False,
    ):
        super().__init__(0)

        self.name = "Los"

        self.path_pts = np.asarray(path_pts, dtype=float)

        # On garde seulement x, y si les points sont en 3D.
        if self.path_pts.shape[1] >= 2:
            self.path_xy = self.path_pts[:, :2]
        else:
            raise ValueError("path_pts doit contenir au moins deux colonnes: x, y.")

        self.controller = LOSController(
            path_xy=self.path_xy,
            vx_nom=vx_nom,
            Delta=Delta,
            zeta=zeta,
            omega_n=omega_n,
            ct_yr_nom=ct_yr_nom,
            control_point_ahead=control_point_ahead,
            closed=closed,
        )

        self.last_info = None

        self.add_output_port(
            "los",
            dim=8,
            function=self.los,
            dependencies=(),
        )

    def los(self, x, u, t=0.0, params=None):
        """
        Fonction appelée par Minilink.

        Hypothèse sur l'état:
            x[0] = position x
            x[1] = position y
            x[2] = psi
            x[3] = vx, optionnel
            x[4] = vy, optionnel
            x[5] = r, optionnel
        """

        px = float(x[0])
        py = float(x[1])
        psi = float(x[2])

        vx = float(x[3]) if len(x) > 3 else None
        vy = float(x[4]) if len(x) > 4 else None
        r_meas = float(x[5]) if len(x) > 5 else None

        dt = 0.01

        if params is not None and "dt" in params:
            dt = float(params["dt"])

        vx_ref, r_ref, info = self.controller.compute(
            px,
            py,
            psi,
            dt,
            r_meas=r_meas,
            vx=vx,
            vy=vy,
        )

        self.last_info = info

        return np.array(
            [
                vx_ref,
                r_ref,
                info["e_perp"],
                info["chi_d"],
                info["ax"],
                info["ay"],
                info["x_ctrl"],
                info["y_ctrl"],
            ],
            dtype=float,
        )


# ============================================================
# Exemple de chemin
# ============================================================

_PATH_X0 = 0.0
_PATH_X1 = 0.0

pts = np.array(
    [
        [0.0, 3.0, 0.0],
        [5.0, 3.0, 0.0],
        [5.0, -_PATH_X1, 0.0],
    ]
)


def main():
    los_system = Los(
        path_pts=pts,
        vx_nom=1.0,
        Delta=4.0,
        zeta=0.7,
        omega_n=1.2,
        ct_yr_nom=1.0,
        control_point_ahead=0.5,
        closed=False,
    )

    # Exemple d'état:
    # x, y, psi, vx, vy, r
    x_state = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0])

    los_output = los_system.los(x_state, u=None, t=0.0, params={"dt": 0.01})

    print("Sortie LOS:")
    print(f"vx_ref  = {los_output[0]:.3f}")
    print(f"r_ref   = {los_output[1]:.3f}")
    print(f"e_perp  = {los_output[2]:.3f}")
    print(f"chi_d   = {los_output[3]:.3f}")
    print(f"ax, ay  = ({los_output[4]:.3f}, {los_output[5]:.3f})")
    print(f"x_ctrl  = {los_output[6]:.3f}")
    print(f"y_ctrl  = {los_output[7]:.3f}")

    # Visualisation simple
    plt.figure()
    plt.plot(pts[:, 0], pts[:, 1], "k-o", label="Path")

    plt.plot(x_state[0], x_state[1], "bo", label="Bateau")
    plt.plot(los_output[6], los_output[7], "go", label="Point de contrôle")
    plt.plot(los_output[4], los_output[5], "ro", label="Point lookahead")

    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Géométrie LOS")
    plt.show()


if __name__ == "__main__":
    main()
