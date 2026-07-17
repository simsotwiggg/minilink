import math

import matplotlib.pyplot as plt
import numpy as np

from minilink.control.constant_ref import ConstantReference
from minilink.control.full_bicycle_meas import BicycleMeasurement
from minilink.control.generic_pid import PID
from minilink.control.motor_map import AccToThr
from minilink.control.steering_map import AngularSpeedToSteeringMap
from minilink.core.diagram import DiagramSystem
from minilink.dynamics.catalog.vehicles.dynamic_bicycle_SL import (
    DynamicBicycleRearWheelDriveEngine,
)
from minilink.simulations_bicycle_model.path.path_plotter import Lines
from minilink.simulations_bicycle_model.traj.LOS_modified import LosSL
from minilink.simulations_bicycle_model.traj.path_segments import (
    make_rectangle_path,
    make_rounded_rectangle_from_path,
)
from minilink.simulations_bicycle_model.vehicule_helper import (
    attach_vehicle_centered_diagram_camera,
    create_vehicle,
)


def wrap_pi(angle):
    """
    Ramène un angle dans [-pi, pi].
    Remplace simpleSpeedBoatSim._wrap_pi si non disponible.
    """
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


class PIDTheta(PID):
    def __init__(
        self,
        Kp: float = 1,
        Ki: float = 0,
        Kd: float = 0,
        tau: float = 0.1,
        meas0: float = 0,
        cmd_min: float = -np.inf,
        cmd_max: float = np.inf,
        i_min: float = -np.inf,
        i_max: float = np.inf,
        name: str = "PID",
    ):
        super().__init__(Kp, Ki, Kd, tau, meas0, cmd_min, cmd_max, i_min, i_max, name)

        self.add_input_port("u_meas", nominal_value=np.array([0.0]))
        self.add_input_port("v_meas", nominal_value=np.array([0.0]))

    def calculate_error(self, ref: float, meas: float, u) -> float:
        u_body = u[3]
        v_body = u[4]

        beta = math.atan2(v_body, max(1e-6, u_body))
        chi_meas = wrap_pi(meas + beta)
        # print(f"beta: {beta}, meas: {meas}")

        e = wrap_pi(ref - chi_meas)
        return float(e)


path_raw = make_rectangle_path(Lx=25.0, Ly=20.0)

# Rounded rectangle generated FROM the raw rectangle
path = make_rounded_rectangle_from_path(
    path_raw,
    R=7.0,
    nseg=8,
    narc=4,
    min_ds=0.1,
    closed=True,
)


def create_diagram(vehicle: DynamicBicycleRearWheelDriveEngine, vx_ref=1.0):

    path_raw_lines = Lines(pts=path_raw, name="Raw path", color="gray", linewidth=1)

    los_path = Lines(pts=path, name="Los path", color="salmon")

    los_system = LosSL(
        path_pts=path,
        vx_nom=vx_ref,
        Delta=8.0,
        omega_n=1.2,
        control_point_ahead=vehicle.a + 0.5,
        closed=True,
    )

    v_bicycle = ConstantReference(ref=vx_ref, name="Constant speed")

    r_to_steering = AngularSpeedToSteeringMap(vehicle)

    full_state_meas = BicycleMeasurement(name="Meas states", y_size=10)

    theta_pid = PIDTheta(
        Kp=5.0,
        Ki=0.0,
        Kd=0.0,
        cmd_min=-10.0,
        cmd_max=10.0,
        i_min=-1.0,
        i_max=1.0,
        name="Yaw PID",
    )

    r_pid = PID(
        Kp=0.3,
        Ki=0.0,
        Kd=0.1,
        cmd_min=-np.pi / 4.0,
        cmd_max=np.pi / 4.0,
        i_min=-np.pi / 4.0,
        i_max=np.pi / 4.0,
        name="Yaw rate PID",
    )

    acc_to_thr = AccToThr(vehicle)

    v_pid = PID(
        Kp=0.8,
        Ki=0.01,
        Kd=0.0,
        cmd_min=-10.0,
        cmd_max=10.0,
        i_min=-0.0,
        i_max=5.0,
        name="Speed PID",
    )

    diagram = DiagramSystem()
    diagram.name = "Cascade PID - DynamicBicycleRearWheelDriveEngine"

    diagram.add_subsystem(r_to_steering, "r_to_steering")
    diagram.add_subsystem(vehicle, "vehicle")
    diagram.add_subsystem(r_pid, "r_pid")
    diagram.add_subsystem(theta_pid, "theta_pid")

    diagram.add_subsystem(acc_to_thr, "acc_to_thr")
    diagram.add_subsystem(v_pid, "v_pid")

    diagram.add_subsystem(full_state_meas, "full_state_meas")
    diagram.add_subsystem(path_raw_lines, "path_raw")
    diagram.add_subsystem(los_path, "los_path")
    diagram.add_subsystem(los_system, "los_system")
    diagram.add_subsystem(v_bicycle, "v_bicycle")

    # Connect the blocks
    diagram.connect("v_bicycle", "ref", "v_pid", "ref")

    diagram.connect("los_system", "theta", "theta_pid", "ref")

    diagram.connect("theta_pid", "cmd", "r_pid", "ref")
    diagram.connect("theta_pid", "cmd", "r_to_steering", "r_targ")

    diagram.connect("vehicle", "y", "full_state_meas", "y")
    diagram.connect("full_state_meas", "u_meas", "r_to_steering", "vx_meas")
    diagram.connect("full_state_meas", "r_meas", "r_pid", "meas")
    diagram.connect("full_state_meas", "u_meas", "v_pid", "meas")
    diagram.connect("full_state_meas", "theta_meas", "theta_pid", "meas")
    diagram.connect("full_state_meas", "theta_meas", "los_system", "psi")
    diagram.connect("full_state_meas", "y_meas", "los_system", "y")
    diagram.connect("full_state_meas", "x_meas", "los_system", "x")
    diagram.connect("full_state_meas", "u_meas", "theta_pid", "u_meas")
    diagram.connect("full_state_meas", "v_meas", "theta_pid", "v_meas")

    diagram.connect("r_to_steering", "delta", "r_pid", "feedfoward")

    diagram.connect("r_pid", "cmd", "vehicle", "delta")

    diagram.connect("v_pid", "cmd", "acc_to_thr", "acc_targ")

    diagram.connect("acc_to_thr", "thr", "vehicle", "thr")
    diagram.connect("full_state_meas", "w_r_meas", "acc_to_thr", "w_motor")

    return diagram


def main():
    vx = 10.0

    vehicle = create_vehicle(Y=10.0, vx=0.0, theta=np.pi, tire_slip_mode="c")

    diagram = create_diagram(vehicle, vx_ref=vx)

    # diagram.plot_diagram()

    diagram.compute_trajectory(tf=30, dt=0.01)

    x0 = float(vehicle.x0[0])
    y0 = float(vehicle.x0[1])
    theta0 = float(vehicle.x0[2])

    los_system = LosSL(
        path_pts=path,
        Delta=8.0,
        omega_n=1.2,
        control_point_ahead=vehicle.a + 0.5,
        closed=True,
    )

    _, info = los_system.controller.compute(x0, y0, theta0)

    ax = info["ax"]
    ay = info["ay"]

    plt.figure()
    plt.plot(
        path_raw[:, 0],
        path_raw[:, 1],
        "-o",
        color="gray",
        linewidth=2,
        label="Raw path",
    )
    plt.plot(
        path[:, 0],
        path[:, 1],
        "-o",
        color="salmon",
        linewidth=1,
        label="LOS path",
    )
    plt.plot(
        ax,
        ay,
        "-o",
        color="red",
        linewidth=1,
        label="LOS first control point",
    )

    # Draw vehicle initial pose as an arrow using theta from vehicle.x0[2]
    arrow_length = vehicle.L * 2  # meters
    dx = arrow_length * math.cos(theta0)
    dy = arrow_length * math.sin(theta0)
    plt.arrow(
        x0,
        y0,
        dx,
        dy,
        head_width=0.6,
        head_length=1.0,
        fc="blue",
        ec="blue",
        linewidth=3,
        length_includes_head=True,
        label="Vehicle initial pose",
    )

    plt.xlabel("X [m]")
    plt.ylabel("Y [m]")
    plt.title("Raw path vs LOS path")
    plt.legend()
    plt.grid(True)
    plt.show()

    # --- New figure: LOS control point trajectory over time ---
    try:
        i0, i1 = diagram.state_index["vehicle"]
        xv = diagram.traj.x[i0:i1, :]
        px_t = np.array(xv[0, :]).flatten()
        py_t = np.array(xv[1, :]).flatten()
        psi_t = np.array(xv[2, :]).flatten()
    except Exception:
        # Fallback to a single initial point if trajectory format differs
        px_t = np.array([x0])
        py_t = np.array([y0])
        psi_t = np.array([theta0])

    x_ctrls = []
    y_ctrls = []
    for k in range(px_t.shape[0]):
        _px = float(px_t[k])
        _py = float(py_t[k])
        _psi = float(psi_t[k])
        _, _info = los_system.controller.compute(_px, _py, _psi)
        x_ctrls.append(_info["x_ctrl"])
        y_ctrls.append(_info["y_ctrl"])

    x_ctrls = np.array(x_ctrls)
    y_ctrls = np.array(y_ctrls)

    plt.figure()
    plt.plot(
        path[:, 0], path[:, 1], "-o", color="salmon", linewidth=1, label="LOS path"
    )
    plt.plot(
        x_ctrls,
        y_ctrls,
        "-.",
        color="orange",
        linewidth=1.2,
        label="LOS control point over time",
    )
    # Mark start and end of control point trajectory
    if x_ctrls.size > 0:
        plt.scatter(
            x_ctrls[0], y_ctrls[0], color="green", s=36, label="start ctrl point"
        )
        plt.scatter(x_ctrls[-1], y_ctrls[-1], color="red", s=36, label="end ctrl point")

    plt.xlabel("X [m]")
    plt.ylabel("Y [m]")
    plt.title("LOS V1 control point trajectory over time NO COUPLING")
    plt.legend()
    plt.grid(True)
    plt.axis("equal")
    plt.show()

    # # PID PLOTS
    traj = diagram.reconstruct_internal_signals(diagram.traj)
    pid_logs = traj.get_signal("theta_pid:pid_int_value")

    error = pid_logs[0, :]
    # meas = pid_logs[1, :]

    # ref = np.unwrap(np.array(error))

    t = traj.t

    plt.figure()
    plt.plot(t, error, label="Goal Vehicule theta")
    # plt.plot(t, meas, label="Measured Vehicule theta")
    plt.xlabel("Time [s]")
    plt.ylabel("Theta [rad]")
    plt.title("Theta PID - Reference vs Measured")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Animation

    # vehicle.camera_scale = 15.0
    attach_vehicle_centered_diagram_camera(diagram, vehicle)
    # Save animation to MP4 using the Matplotlib renderer (requires ffmpeg)
    # diagram.animate(renderer="matplotlib")

    from minilink.graphical.animation import Animator

    animator = Animator(diagram)
    animator.animate_simulation(
        diagram.traj,
        renderer="matplotlib",
        save=True,
        file_name="traj_for_path_1_combined",
        show=True,
    )


if __name__ == "__main__":
    main()
