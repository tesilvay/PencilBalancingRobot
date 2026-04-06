import numpy as np
from src.shared import WorkspaceParams





class BasePlant:
    def __init__(self, w:WorkspaceParams, max_acc:float):
        self.x_ref = w.x_ref
        self.y_ref = w.y_ref
        self.safe_radius = w.safe_radius
        self.max_acc = max_acc
        
    
    def apply_workspace_limits(self, x, vx, y, vy):
        if self.safe_radius is None:
            return x, vx, y, vy

        dx = x - self.x_ref
        dy = y - self.y_ref
        dist = np.sqrt(dx * dx + dy * dy)

        if dist <= self.safe_radius:
            return x, vx, y, vy

        scale = self.safe_radius / dist
        dx *= scale
        dy *= scale
        x = self.x_ref + dx
        y = self.y_ref + dy

        normal = np.array([dx, dy]) / self.safe_radius
        vel = np.array([vx, vy])
        v_out = np.dot(vel, normal)

        if v_out > 0:
            vel = vel - v_out * normal

        return x, vel[0], y, vel[1]

    def clamp_acceleration(self, vx_dot, vy_dot):
        if self.max_acc is None:
            return vx_dot, vy_dot

        acc_vec = np.array([vx_dot, vy_dot])
        norm    = np.linalg.norm(acc_vec)

        if norm > self.max_acc and norm > 0:
            acc_vec = acc_vec * (self.max_acc / norm)

        return acc_vec[0], acc_vec[1]