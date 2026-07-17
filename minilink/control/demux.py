import numpy as np

from minilink.core.system import System


class Demux(System):
    """A simple measurement block that outputs a single element of the input vector."""

    def __init__(self, name: str, y_size: int = 10):
        super().__init__(0)

        self.name = name

        self.y_size = int(y_size)

        self.inputs = {}
        self.add_input_port(
            "cmd",
            nominal_value=np.zeros(self.y_size),
        )

        self.outputs = {}
        self.add_output_port(
            "theta_ref",
            dim=1,
            function=self.theta_ref,
            dependencies=["cmd"],
        )

        self.add_output_port(
            "u_ref",
            dim=1,
            function=self.u_ref,
            dependencies=["cmd"],
        )

    def theta_ref(self, x, u, t=0.0, params=None):
        # print(f"Demux: theta: {u[0]}")

        return np.array([u[0]], dtype=float)

    def u_ref(self, x, u, t=0.0, params=None):
        # print(f"Demux:u: {u[1]}")

        return np.array([u[1]], dtype=float)

    def get_kinematic_geometry(self):
        return []

    def get_kinematic_transforms(self, x, u, t):
        return []
