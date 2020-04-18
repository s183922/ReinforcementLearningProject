"""
This file may not be shared/redistributed freely. Please read copyright notice in the git repo.
"""
from scipy.optimize import Bounds
import sympy as sym
import numpy as np
from ct_env import ContiniousSymbolicEnvironment
from ct_cost import SymbolicQRCost
from dp_cartpole_env import render_cartpole

class ContiniousCartpole(ContiniousSymbolicEnvironment):
    state_size = 4
    action_size = 1
    state_labels = ["x", "d_x", "theta", "d_theta"]
    render = render_cartpole
    def __init__(self, mc=2,
                 mp=0.5,
                 l=0.5,
                 g=9.807, maxForce=50, dist=1.0, simple_bounds=None, cost=None):

        self.mc = mc
        self.mp = mp
        self.l = l
        self.g = g
        '''
        Default to kellys swingup task. (From matlab repo)
        '''
        c0, b0, _ = kelly_swingup(maxForce=maxForce, dist=dist)
        if simple_bounds is None:
            simple_bounds = b0
        if cost is None:
            cost = c0

        super(ContiniousCartpole, self).__init__(cost=cost, simple_bounds=simple_bounds)


    def sym_f(self, x, u, t=None):
        mp = self.mp
        l = self.l
        mc = self.mc
        g = self.g

        x_dot = x[1]
        theta = x[2]
        sin_theta = sym.sin(theta)
        cos_theta = sym.cos(theta)
        theta_dot = x[3]
        F = u[0]
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

    def guess(self):
        guess = {'t0': 0,
                 'tF': 2,
                 'x': [np.asarray( self.simple_bounds_['x0'].lb ), np.asarray( self.simple_bounds_['xF'].ub )  ],
                 'u': [ np.asarray( [0] ), np.asarray(  [0] ) ] }
        return guess


def upright_swingup():
    return kelly_swingup()

def kelly_swingup(maxForce=50, dist=1.0):
    """
    Return problem roughly comparable to the Kelly swingup task
    note we have to flip coordinate system because we are using corrected dynamics.
    https://github.com/MatthewPeterKelly/OptimTraj/blob/master/demo/cartPole/MAIN_minTime.m

    Use the SymbolicQRCost to get the cost function.
    """
    simple_bounds = {'t0': Bounds([0], [0]),
                     'tF': Bounds([0.01], [np.inf])}
    # add missing constraints to simple_bounds here;
    # you should add constraints for x, u x0 and xF.
    # Remember the x-position can be within [-2*dist, 2*dist]
    # TODO: 4 lines missing.
    raise NotImplementedError("")
    cost = SymbolicQRCost(c=1)  # just minimum time
    args = {}
    return cost, simple_bounds, args
