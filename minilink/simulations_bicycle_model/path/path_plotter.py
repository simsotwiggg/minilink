import matplotlib.pyplot as plt
import numpy as np

from minilink.core.system import System
from minilink.graphical.animation.primitives import (
    CustomLine,
)


class Lines(System):
    def __init__(
        self,
        pts,
        name: str,
        color: str = "seagreen",
        linewidth: int = 2,
        style: str = "--",
    ):
        super().__init__(0)
        self.name = name
        self.pts = pts
        self.color = color
        self.linewidth = linewidth
        self.style = style

    def get_kinematic_geometry(self):
        return [
            CustomLine(
                self.pts, color=self.color, linewidth=self.linewidth, style=self.style
            )
        ]

    def get_kinematic_transforms(self, x, u, t):
        return [np.eye(4)]
