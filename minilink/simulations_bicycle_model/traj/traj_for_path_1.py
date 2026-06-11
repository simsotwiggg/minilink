import math

import matplotlib.pyplot as plt
import numpy as np

from minilink.control.constant_ref import ConstantReference
from minilink.control.full_bicycle_meas import BicycleMeasurement
from minilink.control.generic_pid import PID, Sum
from minilink.control.motor_map import AccToRearForce, ThrMap
from minilink.control.steering_map import AngularSpeedToSteeringMap
from minilink.core.diagram import DiagramSystem
from minilink.core.system import DynamicSystem, System
from minilink.dynamics.catalog.vehicles.dynamic_bicycle import (
    DynamicBicycleRearWheelDriveEngine,
)
from minilink.simulations_bicycle_model.path.path_plotter import Lines
from minilink.simulations_bicycle_model.traj.LOS_modified import (
    Los,
    # make_rounded_path_from_points,
    # make_rounded_rectangle_path,
)
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


class PIDTheta(DynamicSystem):
    """PID Generic"""

    def __init__(
        self,
        Kp: float = 1.0,
        Ki: float = 0.0,
        Kd: float = 0.0,
        tau: float = 0.1,
        meas0: float = 0.0,
        cmd_min: float = -np.inf,
        cmd_max: float = np.inf,
        i_min: float = -np.inf,
        i_max: float = np.inf,
        name: str = "PID",
    ):
        super().__init__(2)
        self.name = name

        self.params = {
            "Kp": Kp,
            "Ki": Ki,
            "Kd": Kd,
            "tau": tau,
            "cmd_min": cmd_min,
            "cmd_max": cmd_max,
            "i_min": i_min,
            "i_max": i_max,
        }
        self.state.labels = ["int_e", "meas_filt"]
        self.x0 = np.array([0.0, meas0], dtype=float)

        self.inputs = {}
        self.add_input_port("ref", nominal_value=np.array([0.0]))
        self.add_input_port("meas", nominal_value=np.array([0.0]))

        self.outputs = {}
        self.add_output_port(
            "cmd",
            dim=1,
            function=self.h_w,
            dependencies=["ref", "meas"],
        )
        # self.add_output_port(1, "x", function=self.compute_state, dependencies=[])
        self.add_output_port(
            "logs",
            dim=2,
            function=self.data_signal,
            dependencies=["ref", "meas"],
        )

        self.add_output_port(
            "pid_int_value",
            dim=3,
            function=self.int_vars,
            dependencies=[],
        )

    def data_signal(self, x, u, t=0.0, params=None):
        ref = float(u[0])
        meas = float(u[1])
        return np.array([ref, meas], dtype=float)

    def f(self, x, u, t=0.0, params=None):
        p = self.params if params is None else params
        int_e, meas_filt = float(x[0]), float(x[1])
        ref, meas = float(u[0]), float(u[1])

        e = wrap_pi(ref - meas)
        tau = max(p["tau"], 1e-3)
        # Dirty derivative on measurement
        d_meas_filt = (meas - meas_filt) / tau

        # Integrator with simple clamp protection.
        d_int_e = e

        # e' = ref' - meas'
        # Si ref' ~= 0; il faut que ref change lentement.
        # e' = 0 - meas' = -meas'
        cmd = p["Kp"] * e + p["Ki"] * int_e - p["Kd"] * d_meas_filt
        stop_hi = (cmd >= p["cmd_max"]) and (e > 0)
        stop_lo = (cmd <= p["cmd_min"]) and (e < 0)

        d_int_e = 0.0 if (stop_hi or stop_lo) else e

        if int_e >= p["i_max"] and e > 0.0:
            d_int_e = 0.0
        elif int_e <= p["i_min"] and e < 0.0:
            d_int_e = 0.0

        return np.array([d_int_e, d_meas_filt], dtype=float)

    def h_w(self, x, u, t=0.0, params=None):
        p = self.params if params is None else params
        int_e, meas_filt = float(x[0]), float(x[1])
        ref, meas = float(u[0]), float(u[1])

        e = wrap_pi(ref - meas)

        tau = max(p["tau"], 1e-3)
        d_filt = (meas - meas_filt) / tau

        cmd = p["Kp"] * e + p["Ki"] * int_e - p["Kd"] * d_filt

        cmd = np.clip(cmd, p["cmd_min"], p["cmd_max"])

        return np.array([cmd], dtype=float)

    def int_vars(self, x, u, t=0.0, params=None):
        p = self.params if params is None else params
        int_e, meas_filt = float(x[0]), float(x[1])
        ref, meas = float(u[0]), float(u[1])

        e = ref - meas

        tau = max(p["tau"], 1e-3)
        d_filt = (meas - meas_filt) / tau

        cmd = p["Kp"] * e + p["Ki"] * int_e - p["Kd"] * d_filt

        cmd = np.clip(cmd, p["cmd_min"], p["cmd_max"])

        return np.array([e, d_filt, int_e], dtype=float)

    def get_kinematic_geometry(self):
        return []

    def get_kinematic_transforms(self, _x, _u, _t):
        return []


def create_diagram(vehicle: DynamicBicycleRearWheelDriveEngine, vx_ref=1.0):

    path_raw = make_rectangle_path(Lx=40.0, Ly=20.0)
    path_raw_lines = Lines(pts=path_raw, name="Raw path", color="green", linewidth=1)

    # Rounded rectangle generated FROM the raw rectangle
    path = make_rounded_rectangle_from_path(
        path_raw,
        R=7.0,
        nseg=8,
        narc=4,
        min_ds=0.1,
        closed=True,
    )

    los_path = Lines(pts=path, name="Los path", color="blue")

    los_system = Los(
        path_pts=path,
        vx_nom=vx_ref,
        Delta=8.0,
        omega_n=1.2,
        control_point_ahead=vehicle.a + 0.5,
        closed=True,
    )

    v_bicycle = ConstantReference(ref=vx_ref, name="Constant angular speed")

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
        name="Yaw rate PID",
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

    thr_map = ThrMap(vehicle)
    acc_to_force = AccToRearForce(vehicle)

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

    sum_bloc = Sum(max=np.pi / 2.0, min=-np.pi / 2.0)

    diagram = DiagramSystem()
    diagram.name = "Cascade PID - DynamicBicycleRearWheelDriveEngine"

    diagram.add_subsystem(r_to_steering, "r_to_steering")
    diagram.add_subsystem(vehicle, "vehicle")
    diagram.add_subsystem(r_pid, "r_pid")
    diagram.add_subsystem(theta_pid, "theta_pid")

    diagram.add_subsystem(acc_to_force, "acc_to_force")
    diagram.add_subsystem(thr_map, "thr_map")
    diagram.add_subsystem(v_pid, "v_pid")

    diagram.add_subsystem(full_state_meas, "full_state_meas")
    diagram.add_subsystem(sum_bloc, "sum_bloc")

    diagram.add_subsystem(path_raw_lines, "path_raw")
    diagram.add_subsystem(los_path, "los_path")
    diagram.add_subsystem(los_system, "los_system")
    diagram.add_subsystem(v_bicycle, "v_bicycle")

    diagram.connect("v_bicycle", "ref", "v_pid", "ref")

    diagram.connect("v_pid", "cmd", "acc_to_force", "acc_targ")

    diagram.connect("los_system", "theta", "theta_pid", "ref")

    diagram.connect("theta_pid", "cmd", "r_pid", "ref")
    diagram.connect("theta_pid", "cmd", "r_to_steering", "r_targ")

    diagram.connect("vehicle", "y", "full_state_meas", "y")
    diagram.connect("full_state_meas", "vx_meas", "r_to_steering", "vx_meas")
    diagram.connect("full_state_meas", "r_meas", "r_pid", "meas")
    diagram.connect("full_state_meas", "vx_meas", "v_pid", "meas")
    diagram.connect("full_state_meas", "theta_meas", "theta_pid", "meas")
    diagram.connect("full_state_meas", "theta_meas", "los_system", "psi")
    diagram.connect("full_state_meas", "y_meas", "los_system", "y")
    diagram.connect("full_state_meas", "x_meas", "los_system", "x")

    diagram.connect("r_pid", "cmd", "sum_bloc", "1")
    diagram.connect("r_to_steering", "delta", "sum_bloc", "2")

    diagram.connect("sum_bloc", "result", "vehicle", "delta")
    diagram.connect("acc_to_force", "F_rear", "thr_map", "F_rear")
    diagram.connect("thr_map", "thr", "vehicle", "thr")
    diagram.connect("full_state_meas", "w_r_meas", "thr_map", "w_rear")

    return diagram


def main():
    vx = 10.0

    vehicle = create_vehicle(Y=15.0, vx=vx, theta=np.pi, tire_slip_mode=None)

    diagram = create_diagram(vehicle, vx_ref=vx)

    # diagram.plot_diagram()

    diagram.compute_trajectory(tf=20, dt=0.005)

    # traj = diagram.reconstruct_internal_signals(diagram.traj)
    # pid_logs = traj.get_signal("los_system:logs")

    # meas = pid_logs[0, :]

    # t = traj.t

    # plt.figure()
    # plt.plot(t, meas, label="Measured Error LOS")
    # plt.xlabel("Time [s]")
    # plt.ylabel("Theta [rad]")
    # plt.title("Error LOS")
    # plt.legend()
    # plt.grid(True)
    # plt.show()

    traj = diagram.reconstruct_internal_signals(diagram.traj)
    pid_logs = traj.get_signal("v_pid:logs")

    ref = pid_logs[0, :]
    meas = pid_logs[1, :]

    ref = np.unwrap(np.array(ref))

    t = traj.t

    plt.figure()
    plt.plot(t, ref, label="Goal Vehicule theta")
    plt.plot(t, meas, label="Measured Vehicule theta")
    plt.xlabel("Time [s]")
    plt.ylabel("Theta [rad]")
    plt.title("Theta PID - Reference vs Measured")
    plt.legend()
    plt.grid(True)
    plt.show()

    attach_vehicle_centered_diagram_camera(diagram, vehicle)

    diagram.animate(renderer="matplotlib")


if __name__ == "__main__":
    main()
