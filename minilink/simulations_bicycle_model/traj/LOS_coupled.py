import math

import numpy as np
from simpleSpeedBoatSim import _wrap_pi as wrap_pi

_EPS = 1e-12


def precompute_path_geometry(path_xy, closed=True):
    """
    Precompute segment heading, heading variation, and segment length.
    Arrays are indexed by segment start index i (segment i -> i+1).
    """
    if len(path_xy) < 2:
        raise ValueError("Path too short")

    path = np.asarray(path_xy, dtype=float)
    if closed:
        nxt = np.vstack((path[1:], path[:1]))
    else:
        nxt = path[1:]
        path = path[:-1]

    dx = nxt[:, 0] - path[:, 0]
    dy = nxt[:, 1] - path[:, 1]

    dir_path = np.arctan2(dy, dx)
    dir_path = (dir_path + math.pi) % (2.0 * math.pi) - math.pi

    if closed:
        prev = np.roll(dir_path, 1)
        dir_var_path = dir_path - prev
    else:
        dir_var_path = np.zeros_like(dir_path)
        if len(dir_path) > 1:
            dir_var_path[1:] = dir_path[1:] - dir_path[:-1]
    dir_var_path = (dir_var_path + math.pi) % (2.0 * math.pi) - math.pi

    dist_path = np.hypot(dx, dy)
    return dir_path, dir_var_path, dist_path


def _interp_linear_with_zero_outside(x, x_lut, y_lut):
    """Linear interpolation with y=0 outside LUT range."""
    return float(np.interp(float(x), x_lut, y_lut, left=0.0, right=0.0))


def compute_curve_demand_norm(
    dir_var_path, dist_path, idx, t_proj, dist_lut, gain_lut, closed=True
):
    """
    Compute weighted sum of upcoming absolute heading variation.

    The accumulation starts at the remaining distance on the current segment
    and walks forward until max(dist_lut). Curvature demand is normalized by pi/2.
    """
    dir_var_path = np.asarray(dir_var_path, dtype=float)
    dist_path = np.asarray(dist_path, dtype=float)
    dist_lut = np.asarray(dist_lut, dtype=float)
    gain_lut = np.asarray(gain_lut, dtype=float)

    n = len(dist_path)
    if n == 0:
        return 0.0
    if len(dist_lut) == 0:
        return 0.0

    i = int(idx) % n
    t_proj = float(np.clip(t_proj, 0.0, 1.0))
    max_dist = float(dist_lut[-1])

    dist_ahead = (1.0 - t_proj) * float(dist_path[i])
    curve_raw = 0.0
    steps = 0
    max_steps = n if closed else max(1, n - i)

    while dist_ahead <= max_dist and steps < max_steps:
        i = (i + 1) % n
        steps += 1

        abs_dir_var = abs(float(dir_var_path[i]))
        gain = _interp_linear_with_zero_outside(dist_ahead, dist_lut, gain_lut)
        curve_raw += abs_dir_var * gain

        dist_ahead += float(dist_path[i])
        if (not closed) and (i == n - 1):
            break

    return float(curve_raw / (math.pi / 2.0))


def speed_limit_from_curve_demand(curve_demand_norm, curve_demand_lut, speed_limit_lut):
    """Map normalized curve demand to speed limit via linear interpolation."""
    x = np.asarray(curve_demand_lut, dtype=float)
    y = np.asarray(speed_limit_lut, dtype=float)
    if len(x) == 0:
        return float("inf")
    # Clamp above LUT to the last value, as in MATLAB interp1(..., 'linear', last).
    return float(
        np.interp(float(curve_demand_norm), x, y, left=float(y[0]), right=float(y[-1]))
    )


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
    t = 0.0 if t < 0.0 else (1.0 if t > 1.0 else t)
    qx, qy = x0 + t * vx, y0 + t * vy
    return qx, qy, t


def project_on_path_with_t(path_xy, x, y, closed=True):
    """
    Projette (x,y) sur la polyline (fermée par défaut).
    Retourne: idx, t∈[0,1], qx,qy, e_perp, psi_path
    """
    best = None
    for i, j in _iter_segments(path_xy, closed):
        x0, y0 = path_xy[i]
        x1, y1 = path_xy[j]
        qx, qy, t = _closest_point_on_segment(x, y, x0, y0, x1, y1)
        dx, dy = x1 - x0, y1 - y0
        seg_len = math.hypot(dx, dy)
        if seg_len <= _EPS:
            continue  # ignorer segments dégénérés
        tx, ty = dx / seg_len, dy / seg_len
        nx, ny = -ty, tx
        e = (x - qx) * nx + (y - qy) * ny
        d2 = (x - qx) ** 2 + (y - qy) ** 2
        if best is None or d2 < best[0]:
            psi_path = math.atan2(ty, tx)
            best = (d2, i, t, qx, qy, e, psi_path)
    if best is None:
        raise ValueError("Path invalide pour projection (tous segments dégénérés ?)")
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
    À partir de (idx, t) sur le segment idx, avance de ds (m) le long de la polyligne.
    Traverse les segments; boucle si closed=True. Retourne (ax, ay, idx_a, t_a, psi_a).
    """
    n = len(path_xy)
    if n < 2:
        raise ValueError("Path trop court")

    # Si fermé, ramener ds modulo la longueur totale pour éviter les tours infinis:
    if closed:
        Lt = path_total_length(path_xy, closed=True)
        if Lt > _EPS:
            ds = ds % Lt

    # point initial sur le segment (idx -> idx_next)
    i = idx % n
    j = (i + 1) % n
    x0, y0 = path_xy[i]
    x1, y1 = path_xy[j]
    dx, dy = x1 - x0, y1 - y0
    seg_len = math.hypot(dx, dy)
    if seg_len <= _EPS:
        # sauter les segments nuls
        i = (i + 1) % n
        j = (i + 1) % n
        x0, y0 = path_xy[i]
        x1, y1 = path_xy[j]
        dx, dy = x1 - x0, y1 - y0
        seg_len = math.hypot(dx, dy)

    cx, cy = x0 + t * dx, y0 + t * dy
    rem = (1.0 - t) * seg_len

    # avancer
    steps = 0
    max_steps = len(path_xy) + 2  # garde-fou
    while ds > rem and steps < max_steps:
        ds -= rem
        i = (i + 1) % n
        j = (i + 1) % n
        x0, y0 = path_xy[i]
        x1, y1 = path_xy[j]
        dx, dy = x1 - x0, y1 - y0
        seg_len = math.hypot(dx, dy)
        if seg_len <= _EPS:
            rem = 0.0
            steps += 1
            continue
        cx, cy = x0, y0
        rem = seg_len
        steps += 1

        if not closed and i == n - 1:
            # fin de chemin ouvert : on s'arrête au dernier point
            return x0, y0, i, 0.0, math.atan2(dy, dx)

    lam = 0.0 if seg_len <= _EPS else min(1.0, ds / max(_EPS, seg_len))
    ax, ay = cx + lam * dx, cy + lam * dy
    psi_a = math.atan2(dy, dx)
    return float(ax), float(ay), int(i), float(lam), float(psi_a)


# ===========================
#  Contrôleurs: LOS
# ===========================
class LOSController:
    """
    LOS avec lookahead Δ le long du PATH + régulateur PD sur la course,
    dimensionné pour un amortissement ζ et une pulsation ωn en tenant compte de ct_yr.
    Option: control_point_ahead = ℓ (m) pour projeter un point devant le bateau.
    """

    def __init__(
        self,
        path_xy,
        vx_nom=1.0,  # vitesse nominale (m/s)
        Delta=4.0,  # projection du le path
        zeta=0.7,
        omega_n=1.2,  # cibles d'amortissement et vitesse. zeta est l'amortissement désiré du système de contrôle en boucle fermée. omega_n est la pulsation naturelle du système, qui influence la rapidité de la réponse.
        ct_yr_nom=1.0,  # constante de temps de la boucle r (même ordre de grandeur que sim.ct_yr)
        ki=0.0,
        i_lim=10.0,  # gain intégral ILOS et saturation
        control_point_ahead: float = 4.0,  # projection d'un point de contrôle devant le bateau (m)
        closed: bool = True,  # chemin fermé ou ouvert
        dist_lut=None,  # LUT distance (m) pour pondération courbure
        gain_lut=None,  # LUT gain associé à dist_lut
        curve_demand_lut=None,  # LUT demande de courbure normalisée
        speed_limit_lut=None,
    ):  # LUT limitation de vitesse (m/s)
        self.path = path_xy
        self.Delta = float(Delta)
        self.zeta = float(zeta)
        self.omega_n = float(omega_n)
        self.ct_yr_nom = float(ct_yr_nom)
        self.ki = float(ki)
        self.i_lim = float(i_lim)
        self.int_e = 0.0
        self.control_point_ahead = float(control_point_ahead)
        self.closed = bool(closed)

        # gains PD issus de (ζ, ωn, ct_yr)
        self.kp = self.ct_yr_nom * (self.omega_n**2)
        self.kd = max(0.0, 2.0 * self.zeta * self.omega_n * self.ct_yr_nom - 1.0)

        # limites pratiques
        self.vx_nom = vx_nom
        self.r_max = math.radians(60.0)

        # Pré-calcul géométrie path pour limitation de vitesse par courbure.
        _, self._dir_var_path, self._dist_path = precompute_path_geometry(
            self.path, closed=self.closed
        )

        # Active la limitation seulement si les 4 LUT sont fournies.
        self.dist_lut = None if dist_lut is None else np.asarray(dist_lut, dtype=float)
        self.gain_lut = None if gain_lut is None else np.asarray(gain_lut, dtype=float)
        self.curve_demand_lut = (
            None
            if curve_demand_lut is None
            else np.asarray(curve_demand_lut, dtype=float)
        )
        self.speed_limit_lut = (
            None
            if speed_limit_lut is None
            else np.asarray(speed_limit_lut, dtype=float)
        )
        self.enable_curve_speed_limit = all(
            v is not None
            for v in (
                self.dist_lut,
                self.gain_lut,
                self.curve_demand_lut,
                self.speed_limit_lut,
            )
        )

    def reset(self):
        self.int_e = 0.0

    def compute(self, x, y, psi, dt, r_meas=None, vx=None, vy=None):
        # 1) Point de contrôle (optionnel)
        if self.control_point_ahead != 0.0:
            x_ctrl = x + self.control_point_ahead * math.cos(psi)
            y_ctrl = y + self.control_point_ahead * math.sin(psi)
        else:
            x_ctrl, y_ctrl = x, y

        # 2) Projection + lookahead Δ le long du path
        idx, t, qx, qy, e_perp, psi_path = project_on_path_with_t(
            self.path, x_ctrl, y_ctrl, closed=self.closed
        )
        ax, ay, _, _, _ = point_ahead_along_path(
            self.path, idx, t, self.Delta, closed=self.closed
        )

        # 3) Course désirée (ligne de visée vers A)
        chi_d = math.atan2(ay - y_ctrl, ax - x_ctrl)

        # 4) Mesure de course (cap + sideslip si fourni)
        chi_meas = psi
        if vx is not None and vy is not None:
            chi_meas = wrap_pi(psi + math.atan2(vy, max(1e-6, vx)))

        # 5) Erreur de course
        err_chi = wrap_pi(chi_d - chi_meas)

        # 6) ILOS optionnel (intégrale sur e_perp pour compenser un biais latéral) TODO: int_e n'est pas utilisé dans compute() pour calculer chi_d, mais pourrait l'être pour ajuster chi_d ou directement r_ref. Ici on le fait agir indirectement via chi_d en ajustant la ligne de visée.
        if self.ki > 0.0:
            self.int_e = max(-self.i_lim, min(self.i_lim, self.int_e + e_perp * dt))

        # 7) Contrôle PD sur la course -> r_ref
        #    r_ref = kp * err_chi  - kd * r_meas   (+ intégrale ILOS qui agit via chi_d, pas directement ici)
        r_fb = float(r_meas) if r_meas is not None else 0.0
        r_ref = self.kp * err_chi - self.kd * r_fb

        # saturation
        r_ref = max(-self.r_max, min(self.r_max, r_ref))

        # 8) Vitesse désirée (constante ici)
        vx_ref = self.vx_nom

        curve_demand_norm = None
        vx_limit_curve = None
        if self.enable_curve_speed_limit:
            curve_demand_norm = compute_curve_demand_norm(
                self._dir_var_path,
                self._dist_path,
                idx,
                t,
                self.dist_lut,
                self.gain_lut,
                closed=self.closed,
            )
            vx_limit_curve = speed_limit_from_curve_demand(
                curve_demand_norm, self.curve_demand_lut, self.speed_limit_lut
            )
            vx_ref = min(vx_ref, vx_limit_curve)

        return (
            vx_ref,
            r_ref,
            dict(
                e_perp=e_perp,
                chi_d=chi_d,
                idx=idx,
                t=t,
                ax=ax,
                ay=ay,
                x_ctrl=x_ctrl,
                y_ctrl=y_ctrl,
                kp=self.kp,
                kd=self.kd,
                err_chi=err_chi,
                curve_demand_norm=curve_demand_norm,
                vx_limit_curve=vx_limit_curve,
            ),
        )


# ---------- Géométrie LOS pure ----------
def los_geometry_only(path_xy, x, y, psi, Delta, control_point_ahead, closed=True):
    x_ctrl = x + control_point_ahead * math.cos(psi)
    y_ctrl = y + control_point_ahead * math.sin(psi)
    idx, t, _, _, e_perp, _ = project_on_path_with_t(
        path_xy, x_ctrl, y_ctrl, closed=closed
    )
    ax, ay, _, _, _ = point_ahead_along_path(path_xy, idx, t, Delta, closed=closed)
    chi_d = math.atan2(ay - y_ctrl, ax - x_ctrl)
    return {
        "e_perp": e_perp,
        "chi_d": chi_d,
        "idx": idx,
        "x_ctrl": x_ctrl,
        "y_ctrl": y_ctrl,
        "ax": ax,
        "ay": ay,
    }


# ---------- NOUVEAU: géométrie LOS multi-horizon ----------
def los_geometry_with_horizon(
    path_xy, x, y, psi, los: "LOSController", horizon_secs=(0.5, 1.5, 2.5), closed=True
):
    """
    Retourne la géométrie LOS au point courant + une liste de points lookahead
    à des horizons temporels (en secondes) convertis en ds = vx_nom * t.
    chi_d reste basé sur Δ (LOS standard) pour ne pas changer la commande LOS.
    """
    x_ctrl = x + los.control_point_ahead * math.cos(psi)
    y_ctrl = y + los.control_point_ahead * math.sin(psi)

    idx, t, _, _, e_perp, _ = project_on_path_with_t(
        path_xy, x_ctrl, y_ctrl, closed=closed
    )

    # LOS "standard" (Δ)
    ax_base, ay_base, _, _, _ = point_ahead_along_path(
        path_xy, idx, t, los.Delta, closed=closed
    )
    chi_d = math.atan2(ay_base - y_ctrl, ax_base - x_ctrl)

    ds_list = [max(0.0, los.vx_nom * s) for s in horizon_secs]
    ax_list, ay_list = [], []
    for ds in ds_list:
        axk, ayk, _, _, _ = point_ahead_along_path(path_xy, idx, t, ds, closed=closed)
        ax_list.append(axk)
        ay_list.append(ayk)

    return {
        "e_perp": e_perp,
        "chi_d": chi_d,
        "x_ctrl": x_ctrl,
        "y_ctrl": y_ctrl,
        "ax_list": ax_list,
        "ay_list": ay_list,
        "ds_list": ds_list,
    }


def main():
    # --- simulation ---
    dt = 0.02
    FPS = float(int(1.0 / dt))
    sim = SimpleBoatSim(
        x=0.0, y=0.0, heading=0.0, vx=0.0, yr=0.0, dt=dt, ct_x=1.0, ct_yr=1.0
    )

    # Trajectoire : rectangle arrondi (FERMÉ)
    path_xy = make_rounded_rectangle_path(Lx=40.0, Ly=20.0, R=6.0, nseg=3, narc=5)
    # print(f"Les points du chemin sont: {path_xy}")

    # Curvature anticipation LUTs.
    # distLUT/gainLUT weights how much each upcoming heading change contributes.
    distLUT = np.array([0.0, 2.0, 5.0, 10.0, 20.0, 30.0], dtype=float)
    gainLUT = np.array([1.2, 1.0, 0.6, 0.3, 0.1, 0.0], dtype=float)
    # curveDemandLUT -> speedLimitLUT maps normalized demand to speed cap.
    curveDemandLUT = np.array([0.0, 0.25, 0.5, 1.0, 1.5, 2.0], dtype=float)
    speedLimitLUT = np.array([6.0, 5.5, 3, 1.5, 1.5, 1.5], dtype=float)

    # --- renderer : bornes fixes (m) et grille (m) ---
    w, h = 1280, 720
    xmin, xmax = -60.0, 60.0
    deltaX = xmax - xmin
    whRatio = w / h
    deltaY = deltaX / whRatio
    ymin = -35.0
    ymax = ymin + deltaY
    dx_grid, dy_grid = 10.0, 10.0

    vis = render_simpleBoatSim(
        xmin, xmax, ymin, ymax, w, h, dx_grid, dy_grid, path_xy=path_xy, closed=True
    )

    # --- contrôleur LOS (closed=True pour chemin fermé)
    los = LOSController(
        path_xy,
        vx_nom=6.0,
        Delta=8.0,
        zeta=0.7,
        omega_n=2.0,
        ct_yr_nom=1.0,
        ki=0.0,
        i_lim=10.0,
        control_point_ahead=1.0,
        closed=True,
        dist_lut=distLUT,
        gain_lut=gainLUT,
        curve_demand_lut=curveDemandLUT,
        speed_limit_lut=speedLimitLUT,
    )

    REALTIME_VIS = True
    VIDEO_RECORD = False
    if VIDEO_RECORD:
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        video_out = cv2.VideoWriter("boat_simulation.avi", fourcc, FPS, (w, h))
    WINDOW = "BoatSim"
    if REALTIME_VIS:
        cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)

    t0 = time.time()
    N = int(50.0 / dt)
    print_every = max(1, int(0.5 / dt))
    for k in range(N):
        x, y, psi = sim.x, sim.y, sim.heading

        # contrôleur -> consignes
        # vx_ref, r_ref, dbg = los.compute(x, y, psi, dt)
        vx_ref, r_ref, dbg = los.compute(x, y, psi, dt, r_meas=sim.yr)
        err_chi = dbg.get("err_chi", 0.0)
        e_perp = dbg.get("e_perp", 0.0)
        idx_seg = dbg.get("idx", 0)
        ax = dbg.get("ax", None)
        ay = dbg.get("ay", None)
        x_ctrl = dbg.get("x_ctrl", None)
        y_ctrl = dbg.get("y_ctrl", None)

        if (k % print_every) == 0:
            print(f"t={k * dt:5.2f}s idx={idx_seg:3d} vx_ref={vx_ref:5.2f}")

        # avancer la simu
        x, y, psi, vx, vy, r = sim.update(vx_ref, r_ref)

        # rendu
        t_sim = (k + 1) * dt
        img = vis.update(
            x,
            y,
            psi,
            t=t_sim,
            idx_seg=idx_seg,
            e_perp=e_perp,
            vx_ref=vx_ref,
            r_ref=r_ref,
            ax=ax,
            ay=ay,
            x_ctrl=x_ctrl,
            y_ctrl=y_ctrl,
        )

        if REALTIME_VIS:
            cv2.imshow(WINDOW, img)
            if VIDEO_RECORD:
                video_out.write(img)
            sleep_left = t_sim - (time.time() - t0)
            if sleep_left > 0:
                time.sleep(sleep_left)
            if cv2.waitKey(1) & 0xFF == 27:  # ESC
                break

    if REALTIME_VIS:
        cv2.destroyAllWindows()
    if VIDEO_RECORD:
        video_out.release()
        print("Vidéo enregistrée: boat_simulation.avi")


if __name__ == "__main__":
    main()
