from sim_types import SystemState, TableCommand, TableAccel, PhysicalParams

class BalancerPlant:

    def __init__(self, param: PhysicalParams):
        self.g = param.g
        self.l = param.com_length
        self.tau = param.tau
        self.zeta = param.zeta

    def step(self, state_x: SystemState, command_u: TableCommand, dt):

        # ---- unpack state ----
        x = state_x.x
        x_dot = state_x.x_dot
        alpha_x = state_x.alpha_x
        alpha_x_dot = state_x.alpha_x_dot

        y = state_x.y
        y_dot = state_x.y_dot
        alpha_y = state_x.alpha_y
        alpha_y_dot = state_x.alpha_y_dot
        
        # ---- unpack command ----
        x_des = command_u.x_des
        y_des = command_u.y_des

        # ---- table dynamics ----
        x_ddot = (1/self.tau**2)*(x_des - x) - (2*self.zeta/self.tau)*x_dot
        y_ddot = (1/self.tau**2)*(y_des - y) - (2*self.zeta/self.tau)*y_dot

        # ---- pencil dynamics ----
        alpha_x_ddot = (self.g/self.l)*alpha_x - (1/self.l)*x_ddot
        alpha_y_ddot = (self.g/self.l)*alpha_y - (1/self.l)*y_ddot

        # ---- integrate ----
        x_dot += x_ddot * dt
        x += x_dot * dt

        y_dot += y_ddot * dt
        y += y_dot * dt

        alpha_x_dot += alpha_x_ddot * dt
        alpha_x += alpha_x_dot * dt

        alpha_y_dot += alpha_y_ddot * dt
        alpha_y += alpha_y_dot * dt

        return SystemState(
            x=x,
            x_dot=x_dot,
            alpha_x=alpha_x,
            alpha_x_dot=alpha_x_dot,
            y=y,
            y_dot=y_dot,
            alpha_y=alpha_y,
            alpha_y_dot=alpha_y_dot
        ), TableAccel(
            x_ddot=x_ddot,
            y_ddot=y_ddot
        )
 