####### 3D Pencil reconstruction ##########
'''
Assumptions:
2 event cameras have identified the slope and intercept of the pencil from their pov
they send those two values to this program

Ouputs:
Using those 4 values, a 3d estimate of the pencil is made
Specifically, the position (X, Y) of the pencil at the height of the cameras, as well as the slope of the pencil (alpha_x, alpha_y).

'''

import numpy as np
from dataclasses import dataclass

@dataclass
class PencilState:
	X: float
	Y: float
	alpha_x: float
	alpha_y: float


def desired_table_position(PencilState):
	hi=1