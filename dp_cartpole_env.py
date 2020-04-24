"""
This file may not be shared/redistributed freely. Please read copyright notice in the git repo.
"""
from collections import OrderedDict
import sympy as sym
import numpy as np
from dp_symbolic_env import symv
from gym.envs.classic_control import rendering
from dp_symbolic_env import DPSymbolicEnvironment



def dim1(x):
    return np.reshape(x, (x.size,) ) if x is not None else x


class CartpoleSinCosEnvironment(DPSymbolicEnvironment):
    """ Symbolic version of the discrete Cartpole environment. """
    action_size = 1
    state_size = 5

    def __init__(self,
                 dt,
                 cost=None,
                 constrain=True,
                 min_bounds=-1.0,
                 max_bounds=1.0,
                 mc=1.0,
                 mp=0.1,
                 l=1.0,
                 g=9.80665,
                 **kwargs):
        """Cartpole dynamics.

        Args:
            dt: Time step [s].
            constrain: Whether to constrain the action space or not.
            min_bounds: Minimum bounds for action [N].
            max_bounds: Maximum bounds for action [N].
            mc: Cart mass [kg].
            mp: Pendulum mass [kg].
            l: Pendulum length [m].
            g: Gravity acceleration [m/s^2].
            **kwargs: Additional key-word arguments to pass to the
                AutoDiffDynamics constructor.

        Note:
            state: [x, x', sin(theta), cos(theta), theta']
            action: [F]
            theta: 0 is pointing up and increasing clockwise.
        """

        self.constrained = constrain
        self.min_bounds = min_bounds
        self.max_bounds = max_bounds

        self.l = l
        self.mp_val = mp
        self.g_val = g
        self.mc_val = mc

        super(CartpoleSinCosEnvironment, self).__init__(dt=dt, cost=cost)

    def render(self, x=None):
        # the render function we use assumes parameterization in terms of these.
        sin_theta = np.float(x[2])
        cos_theta = np.float(x[3])
        theta = np.arctan2(sin_theta, cos_theta)
        x_theta = [x[0], x[1], theta, x[4]]
        render_cartpole(self, x=x_theta, mode="human")

    def x_cont2x_discrete(self, xs, method = None):
        """
        converts state space with theta to state with sin cos
        :param xs: x with theta in space
        :return:  x with sin and cos
        """
        if method is None: return [xs[0], xs[1], sym.sin(xs[2]), sym.cos(xs[2]), xs[3]]
        elif method == "numpy": return [xs[0], xs[1], np.sin(xs[2]), np.cos(xs[2]), xs[3]]

    def x_discrete2x_cont(self, xs, method = None):
        if method is None: return [xs[0], xs[1], sym.atan2(xs[2], xs[3]), xs[4]]
        elif method == "numpy": return [xs[0], xs[1], np.arctan2(xs[2], xs[3]), xs[4]]

    def system_derivatives(self, xs, us):
        """
        Compute the system derivatives.
        :params
            xs: current state
            us: force

        :return: System derivatives
        """
        if len(xs) == 5:
            xs = self.x_discrete2x_cont(xs)
        min_bounds = self.min_bounds
        max_bounds = self.max_bounds

        # Define dynamics model as per Razvan V. Florian's
        # "Correct equations for the dynamics of the cart-pole system".
        # Friction is neglected.

        mp = self.mp_val
        l = self.l
        mc = self.mc_val
        g = self.g_val

        x_dot = xs[1]
        theta = xs[2]
        sin_theta = sym.sin(theta)
        cos_theta = sym.cos(theta)
        theta_dot = xs[3]
        F = sym.tanh(us[0]) * (max_bounds - min_bounds) / 2.0
        # Define dynamics model as per Razvan V. Florian's
        # "Correct equations for the dynamics of the cart-pole system".
        # Friction is neglected.

        # Eq. (23)
        temp = (F + mp * l * theta_dot ** 2 * sin_theta) / (mc + mp)
        numerator = g * sin_theta - cos_theta * temp
        denominator = l * (4.0 / 3.0 - mp * cos_theta ** 2 / (mc + mp))
        theta_dot_dot = numerator / denominator

        # Eq. (24)
        x_dot_dot = temp - mp * l * theta_dot_dot * cos_theta / (mc + mp)
        xp = [x_dot,
              x_dot_dot,
              theta_dot,
              theta_dot_dot]

        return xp

    def sym_f_discrete(self, xs, u, dt):
        """
        Compute the discrete time system dynamics with Euler Integration

        :params
            xs: Current State
            u:  Force
            dt: Discrete Time Step
        :return: Next state [x + dt * xdot, xdot + dt * xdotdot, sin(theta + dt * thetadot), cos(theta + dt * thetadot), thetadot + dt * thetadotdot]
        """

        derives = self.system_derivatives(xs, u)
        xs = self.x_discrete2x_cont(xs)
        euler = xs * np.array([1, 1, 1, .99]) + dt * np.asarray(derives)
        euler_expand = [euler[0], euler[1], sym.sin(euler[2]), sym.cos(euler[2]), euler[3]]

        return euler_expand

    def step(self, x0, u_fun, N_steps, method = None):
        """
        Computes the next state of system using RK4 integration
        :param x: Current State
        :param u: Force
        :param dt: Time step
        :return: Next State
        """

        xs = self.Runge_Kutta4(x0, u_fun, 0, N_steps+1, method = method)
        xs = np.array([self.x_cont2x_discrete(x, "numpy") for x in xs])
        return xs[1:]


    def Runge_Kutta4(self, x0, u_fun, t0, tF, method=None):
        if len(x0) == 5:
            x0 = self.x_discrete2x_cont(x0, 'numpy')
        N_steps = int(tF - t0)
        time = np.linspace(t0, tF, N_steps)
        """ Making System Dynamics time dependent given an interpolation function of the trajectory of discrete actions """
        u = symv("u", self.action_size)
        x = symv('x', self.state_size-1)
        derivatives = self.system_derivatives(x, u)
        f_sym = sym.lambdify((tuple(x), tuple(u)), derivatives, 'numpy')
        f = lambda t, x: f_sym(x, u_fun(t).reshape(-1))

        xs = [np.asarray(x0)]
        us = [u_fun(t0).reshape(-1)]

        for n in range(N_steps-1):

            h = self.dt
            t_current = time[n]
            x_current = xs[n]

            k1 = h * np.asarray(f(t_current, x_current))
            k2 = h * np.asarray(f(t_current + h / 2, x_current + k1 / 2))
            k3 = h * np.asarray(f(t_current + h / 2, x_current + k2 / 2))
            k4 = h * np.asarray(f(t_current + h, x_current + k3))

            x_next = x_current * np.array([1, 1, 1, .99]) + 1 / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            if method == "Euler":
                x_next = x_current * np.array([1, 1, 1, .99]) + k1


            u_next = u_fun(t_current + h).reshape(-1)

            xs.append(x_next)
            us.append(u_next)

        xs = np.stack(xs)
        return xs


    def transform_actions(self, us):
        return np.tanh(us) * (self.max_bounds - self.min_bounds) / 2

    def reduce_state(self, state):
        """Reduces a non-angular state into an angular state by replacing
        sin(theta) and cos(theta) with theta.

        In this case, it converts:

            [x, x', sin(theta), cos(theta), theta'] -> [x, x', theta, theta']

        Args:
            state: Augmented state vector [state_size].

        Returns:
            Reduced state size [reducted_state_size].
        """
        if state.ndim == 1:
            x, x_dot, sin_theta, cos_theta, theta_dot = state
        else:
            x = state[..., 0].reshape(-1, 1)
            x_dot = state[..., 1].reshape(-1, 1)
            sin_theta = state[..., 2].reshape(-1, 1)
            cos_theta = state[..., 3].reshape(-1, 1)
            theta_dot = state[..., 4].reshape(-1, 1)

        theta = np.arctan2(sin_theta, cos_theta)
        return np.hstack([x, x_dot, theta, theta_dot])


def render_cartpole(env, x=None, mode='human', close=False):
    if not hasattr(env, 'viewer'):
        env.viewer = None

    if x is None:
        x = env.state
    if close:
        if env.viewer is not None:
            env.viewer.close()
            env.viewer = None
        return None

    screen_width = 600
    screen_height = 400

    world_width = 8  # max visible position of cart
    scale = screen_width / world_width
    carty = 200  # TOP OF CART
    polewidth = 8.0
    # return
    polelen = scale * env.l # 0.6 or self.l

    cartwidth = 40.0
    cartheight = 20.0

    if env.viewer is None:
        env.viewer = rendering.Viewer(screen_width, screen_height)

        l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2

        cart = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
        env.carttrans = rendering.Transform()
        cart.add_attr(env.carttrans)
        cart.set_color(1, 0, 0)
        env.viewer.add_geom(cart)

        l, r, t, b = -polewidth / 2, polewidth / 2, polelen - polewidth / 2, -polewidth / 2
        pole = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
        pole.set_color(0, 0, 1)
        env.poletrans = rendering.Transform(translation=(0, 0))
        pole.add_attr(env.poletrans)
        pole.add_attr(env.carttrans)
        env.viewer.add_geom(pole)

        env.axle = rendering.make_circle(polewidth / 2)
        env.axle.add_attr(env.poletrans)
        env.axle.add_attr(env.carttrans)
        env.axle.set_color(0.1, 1, 1)
        env.viewer.add_geom(env.axle)

        # Make another circle on the top of the pole
        env.pole_bob = rendering.make_circle(polewidth / 2)
        env.pole_bob_trans = rendering.Transform()
        env.pole_bob.add_attr(env.pole_bob_trans)
        env.pole_bob.add_attr(env.poletrans)
        env.pole_bob.add_attr(env.carttrans)
        env.pole_bob.set_color(0, 0, 0)
        env.viewer.add_geom(env.pole_bob)

        env.wheel_l = rendering.make_circle(cartheight / 4)
        env.wheel_r = rendering.make_circle(cartheight / 4)
        env.wheeltrans_l = rendering.Transform(translation=(-cartwidth / 2, -cartheight / 2))
        env.wheeltrans_r = rendering.Transform(translation=(cartwidth / 2, -cartheight / 2))
        env.wheel_l.add_attr(env.wheeltrans_l)
        env.wheel_l.add_attr(env.carttrans)
        env.wheel_r.add_attr(env.wheeltrans_r)
        env.wheel_r.add_attr(env.carttrans)
        env.wheel_l.set_color(0, 0, 0)  # Black, (B, G, R)
        env.wheel_r.set_color(0, 0, 0)  # Black, (B, G, R)
        env.viewer.add_geom(env.wheel_l)
        env.viewer.add_geom(env.wheel_r)

        env.track = rendering.Line((0, carty - cartheight / 2 - cartheight / 4),
                                   (screen_width, carty - cartheight / 2 - cartheight / 4))
        env.track.set_color(0, 0, 0)
        env.viewer.add_geom(env.track)

    if x is None:
        return None

    cartx = x[0] * scale + screen_width / 2.0  # MIDDLE OF CART
    env.carttrans.set_translation(cartx, carty)
    env.poletrans.set_rotation(-x[2] + 0 * np.pi)
    env.pole_bob_trans.set_translation(-env.l * np.sin(x[2]), env.l * np.cos(x[2]))

    return env.viewer.render(return_rgb_array=mode == 'rgb_array')




