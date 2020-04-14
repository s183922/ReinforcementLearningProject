import os, sys
path = sys.path[0]
os.chdir(path)
import gym
from collections import OrderedDict
from AcrobotEnv import AcrobotEnv
import numpy as np
import sympy as sym
from cost import QRCost
from ilqr import ilqr_basic, ilqr_linesearch

def ilqr(env, N, x0, n_iter, use_linesearch, verbose=True):
    if not use_linesearch:
        xs, us, J_hist = ilqr_basic(env, N, x0, n_iterations=n_iter,verbose=verbose) 
    else:
        xs, us, J_hist = ilqr_linesearch(env, N, x0, n_iterations=n_iter, tol=1e-6,verbose=verbose)
    xs, us = np.stack(xs), np.stack(us)
    return xs, us, J_hist


def symv(s, n):
    """
    Returns a vector of symbolic functions. For instance if s='x' and n=3 then it will return
    [x0,x1,x2]
    where x0,..,x2 are symbolic variables.
    """
    return sym.symbols(" ".join(["%s%i," % (s, i) for i in range(n)]))

class SymbolicAcrobot(AcrobotEnv):
    def __init__(self, cost = None, g = 9.82, dt = 0.005):
        super(SymbolicAcrobot, self).__init__()
        self.cost = cost
        self.gravity = g
        self.dt = dt

        self.l_1, self.l_2, self.m_1, self.m_2, self.lc1, self.lc2, self.i1, self.i2, self.sym_g = sym.symbols('l_1, l_2, m_1, m_2, lc1, lc2, i1, i2, g')
        
        par_map = OrderedDict()
        par_map[self.l_1] = self.LINK_LENGTH_1
        par_map[self.l_2] = self.LINK_LENGTH_2
        par_map[self.m_1] = self.LINK_MASS_1
        par_map[self.m_2] = self.LINK_MASS_2
        par_map[self.lc1] = self.LINK_COM_POS_1
        par_map[self.lc2] = self.LINK_COM_POS_2
        par_map[self.i1] = self.LINK_MOI
        par_map[self.i2] = self.LINK_MOI
        par_map[self.sym_g] = self.gravity
        self.par_map = par_map


        u = symv("u", self.action_size)
        x = symv('x', self.state_size)
        """ y is a symbolic variable representing y = f(xs, us, dt) """
        y = self.sym_f_discrete(x, u, self.dt)
        
        # """ compute the symbolic derivate of y wrt. z = (x,u): dy/dz """
        dy_dz = sym.Matrix([[sym.diff(f, zi) for zi in list(x)+list(u)] for f in y])
        # # """ Define (numpy) functions giving next state and the derivatives """
        self.f_z = sym.lambdify((tuple(x), tuple(u)), dy_dz, 'numpy')
        self.f_discrete = sym.lambdify((tuple(x), tuple(u)), y, 'numpy') 


    def sym_f_discrete(self, xs, us, dt):
        """
        Custom Made by Peter
        """
        m1 = self.m_1
        m2 = self.m_2
        l1 = self.l_1
        lc1 = self.lc1
        lc2 = self.lc2
        I1 = self.i1
        I2 = self.i2
        g = self.sym_g
        a = us[0]
        s = xs
        theta1 = s[0]
        theta2 = s[1]
        dtheta1 = s[2]
        dtheta2 = s[3]

        d1 = m1 * lc1 ** 2 + m2 * \
            (l1 ** 2 + lc2 ** 2 + 2 * l1 * lc2 * sym.cos(theta2)) + I1 + I2
        d2 = m2 * (lc2 ** 2 + l1 * lc2 * sym.cos(theta2)) + I2
        phi2 = m2 * lc2 * g * sym.cos(theta1 + theta2 - sym.pi / 2.)
        phi1 = - m2 * l1 * lc2 * dtheta2 ** 2 * sym.sin(theta2) \
               - 2 * m2 * l1 * lc2 * dtheta2 * dtheta1 * sym.sin(theta2)  \
            + (m1 * lc1 + m2 * l1) * g * sym.cos(theta1 - sym.pi / 2) + phi2
        if self.book_or_nips == "nips":
            # the following line is consistent with the description in the
            # paper
            ddtheta2 = (a + d2 / d1 * phi1 - phi2) / \
                (m2 * lc2 ** 2 + I2 - d2 ** 2 / d1)
        else:
            # the following line is consistent with the java implementation and the
            # book
            ddtheta2 = (a + d2 / d1 * phi1 - m2 * l1 * lc2 * dtheta1 ** 2 * sym.sin(theta2) - phi2) \
                / (m2 * lc2 ** 2 + I2 - d2 ** 2 / d1)
        ddtheta1 = -(d2 * ddtheta2 + phi1) / d1

        xp = [theta1 + dt * dtheta1, theta2 + dt * dtheta2, dtheta1 + dt * ddtheta1, dtheta2 + dt * ddtheta2]
        f = [ff.subs(self.par_map) for ff in xp]
        return f

    def f(self, x, u, i, compute_jacobian=False): 
        """
        return f(x,u), f_x, f_u, and Hessians (not implemented)
        where f_x is the derivative df/dx, etc.
        """
        fx = np.asarray( self.f_discrete(x, u) )
        if compute_jacobian:
            J = self.f_z(x, u)
            f_xx, f_ux, f_uu = None,None,None  # Not implemented.
            return fx, J[:, :self.state_size], J[:, self.state_size:], f_xx, f_ux, f_uu
        else:
            return fx 

    def g(self, x, u, i=None, terminal=False, compute_gradients=False): 
        v = self.cost.g(x, u, i, terminal=terminal) # Terminal is deprecated, use gN
        return v[0] if not compute_gradients else v 

    def gN(self, x, i=None, compute_gradients=False):  
        v = self.cost.gN(x) # Not gonna lie this is a bit goofy.
        return v[0] if not compute_gradients else v  



x_goal = np.array([np.pi, 0, 0, 0])

# Instantaneous state cost.
state_size = 4
Q = np.diag([100, 1.0, 0, 0.0])
R = np.array([[0.1]])
# Terminal state cost.
Q_terminal = 100 * np.eye(state_size)

# Instantaneous control cost.
cost = QRCost(Q, R, QN=Q_terminal, x_goal=x_goal)


env = SymbolicAcrobot(cost = cost)
env.reset()

# N = 300
# x0 = env.state
# xs, us, J_hist = ilqr(env, N, x0, n_iter=300, use_linesearch=True)


env.reset()
for _ in range(1000):
    env.render()
    env.step(env.action_space.sample())
env.close
print(env.action_space.sample())