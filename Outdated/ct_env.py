"""
This file may not be shared/redistributed freely. Please read copyright notice in the git repo.

References:
  [Kel17] Matthew Kelly. An introduction to trajectory optimization: how to do your own direct collocation. SIAM Review, 59(4):849–904, 2017. (See kelly2017.pdf). 
"""
import sympy as sym
import numpy as np
import time
from dp_symbolic_env import symv
import sys
import matplotlib.pyplot as plt
from irlc import savepdf

'''
Continious time environments.
'''
class ContiniousSymbolicEnvironment: 
    state_size, action_size = -1, -1   # Dimension of state/action vectors x, u
    def __init__(self, cost=None, simple_bounds=None):
        self.cost = cost
        self.simple_bounds_ = simple_bounds
        t = sym.symbols("t")
        x = symv("x", self.state_size)
        u = symv("u", self.action_size)
        self.f = sym.lambdify( (x, u, t), self.sym_f(x, u, t)) 

    def simulate(self, x0, u_fun, t0, tF, N_steps = 1000, method='rk4'):
        """
        Defaults to RK4 simulation of the trajectory from x0, u0, t0 to tf

        Method can be either 'rk4' or 'euler'

        u_fun has to be a function which returns a list/tuple with same dimension as action_space
        x0 is initial position; it too must be a list of size state_space
        """
        tt = np.linspace(t0, tF, N_steps)
        y = [ np.asarray(x0) ]
        u = [ u_fun( t0 )]
        for n in range(N_steps-1):
            h = tt[n+1] - tt[n]
            tn = tt[n]
            yn = y[n]
            un = u[n]
            unp = u_fun(tn + h)
            if method == 'rk4':
                k1 = h * np.asarray(self.f(yn, un, tn))
                k2 = h * np.asarray(self.f(yn + k1 / 2, u_fun(tn + h / 2), tn + h / 2))
                k3 = h * np.asarray(self.f(yn + k2 / 2, u_fun(tn + h / 2), tn + h / 2))
                k4 = h * np.asarray(self.f(yn + k3, unp, tn + h))
                ynp = yn + 1 / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


            #raise NotImplementedError("")
            elif method == 'euler':
                ynp = yn + h * np.asarray(self.f(yn, un, tn))
            else:
                raise Exception("Bad integration method", method)
            y.append(ynp)
            u.append(unp)

        y = np.stack(y, axis=0)
        u = np.stack(u, axis=0)
        return y, u, tt

    def render(self, x=None):
        raise NotImplementedError()

    def animate_rollout(self, x0, u_fun, t0, tF, N_steps = 1000, fps=30):
        if sys.gettrace() is not None:
            print("Not animating stuff in debugger as it crashes.")
            return

        y, _, tt = self.simulate(x0, u_fun, t0, tF, N_steps=1000)
        secs = tF-t0
        frames = int( np.ceil( secs * fps ) )
        I = np.round( np.linspace(0, N_steps-1, frames)).astype(int)
        y = y[I,:]

        for i in range(frames):
            self.render(x=y[i] )
            time.sleep(0.1)

    def sym_f(self, x, u, t=None):
        raise NotImplementedError()

    def sym_J(self, t0, tF, x0, xF):
        """ Compute Mayer term in cost function """
        return self.cost.sym_J(t0, tF, x0, xF)

    def sym_w(self, x, u, t):
        ''' Compute Lagrange term in cost function '''
        return self.cost.sym_w(x=x, u=u, t=t)

    def sym_h(self, x, u, t):
        '''
        Dynamical path constraint. See (Kel17, Eq.(1.3))
        h(x, u, t) <= 0
        '''
        return []

    def simple_bounds(self):
        ''' Simple inequality constraints (i.e. z_lb <= z <= z_ub)
        Returned as a dict with keys representing the variables they constrain. For instance:

        >>> sb = env.simple_bounds()
        >>> b = sb['x0']

        b will now be a scipy Bounds object and constraints can be found as:

        >>> b.lb <= x0 <= b.ub

        returned as a dict; see implementations. '''
        return self.simple_bounds_

    def sym_g(self, t0,tF,x0,xF):
        '''
        Boundary constraints

        g(t0,tF,x0,xF) <= 0.

        Note: We will not use this function for the course.
        '''
        return []



def plot_trajectory(x_res, tt, lt='k-', ax=None, labels=None, legend=None):
    M = x_res.shape[1]
    if labels is None:
        labels = [f"x_{i}" for i in range(M)]

    ax = plt.subplots(2, (M + 1) // 2)[1] if ax is None else ax
    for i in range(M):
        a = ax[i]
        a.plot(tt, x_res[:, i], lt, label=legend)
        a.set_title(labels[i])

    return ax

def make_space_above(axes, topmargin=1.0):
    """ increase figure size to make topmargin (in inches) space for
        titles, without changing the axes sizes"""
    fig = axes.flatten()[0].figure
    s = fig.subplotpars
    w, h = fig.get_size_inches()

    figh = h - (1-s.top)*h  + topmargin
    fig.subplots_adjust(bottom=s.bottom*h/figh, top=1-topmargin/figh)
    fig.set_figheight(figh)


def plot_solutions(env, solutions, animate=True, pdf=None):
    for sol in solutions:
        grd = sol['grid']
        x_res = sol['grid']['x']
        u_res = sol['grid']['u']
        ts = sol['grid']['ts']
        u_fun = sol['fun']['u']
        x_sim, u_sim, t_sim = env.simulate(x0=grd['x'][0, :], u_fun=u_fun, t0=grd['ts'][0], tF=grd['ts'][-1],
                                           N_steps=1000)
        if animate:
            env.animate_rollout(x0=grd['x'][0, :], u_fun=u_fun, t0=grd['ts'][0], tF=grd['ts'][-1], N_steps=1000, fps=30)

        eqC_val = sol['eqC_val']
        labels = env.state_labels
        print("Initial State: " + ",".join(labels))
        print(x_res[0])
        print("Final State:")
        print(x_res[-1])

        plt.figure()
        ax = plot_trajectory(x_res, ts, lt='k-', labels=labels, legend="Direct prediction")
        plot_trajectory(x_sim, t_sim, lt='r-', ax=ax, labels=labels, legend="RK4 exact simulation")
        plt.suptitle("Coordinates", fontsize=14, y=0.98)
        make_space_above(ax, topmargin=0.5)

        if pdf:
            savepdf(pdf +"_x")
        plt.show()

        plt.plot(u_res[:, 0])
        plt.title("action")
        plt.show()

        plt.figure()
        plot_trajectory(eqC_val, ts[:-1], lt='b-', labels=labels)
        plt.suptitle("defects (equality constraint violations)")
        plt.show()

