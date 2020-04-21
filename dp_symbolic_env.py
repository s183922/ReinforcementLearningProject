"""
This file may not be shared/redistributed freely. Please read copyright notice in the git repo.
"""
import numpy as np
import sympy as sym

def symv(s, n):
    """
    Returns a vector of symbolic functions. For instance if s='x' and n=3 then it will return
    [x0,x1,x2]
    where x0,..,x2 are symbolic variables.
    """
    return sym.symbols(" ".join(["%s%i," % (s, i) for i in range(n)]))

class DPSymbolicEnvironment: 
    state_size = -1  # set these in implementing class.
    action_size = -1
    def __init__(self, dt, cost):
        self.dt = dt
        self.cost = cost
        """ Initialize symbolic variables representing inputs and actions. """
        u = symv("u", self.action_size)
        x = symv('x', self.state_size-1)
        t = sym.symbols("t")
        """ y is a symbolic variable representing y = f(xs, us, dt) """
        euler = self.sym_f_discrete(x, u, dt)

    
        """ compute the symbolic derivate of y wrt. z = (x,u): dy/dz """
        dy_dz = sym.Matrix([[sym.diff(f, zi) for zi in list(x)+list(u)] for f in euler])
        """ Define (numpy) functions giving next state and the derivatives """
        self.f_z = sym.lambdify((tuple(x), tuple(u)), dy_dz, 'numpy')
        self.f_discrete = sym.lambdify((tuple(x), tuple(u)), euler, 'numpy')



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

    def sym_f_discrete(self, xs, us, dt): 
        raise NotImplementedError("")

    def system_derivatives(self, xs, us):
        raise NotImplementedError("")

    def g(self, x, u, i=None, terminal=False, compute_gradients=False):
        v = self.cost.g(x, u, i, terminal=terminal) # Terminal is deprecated, use gN
        return v[0] if not compute_gradients else v 

    def gN(self, x, i=None, compute_gradients=False):  
        v = self.cost.gN(x) # Not gonna lie this is a bit goofy.
        return v[0] if not compute_gradients else v


    def render(self, x=None):
        raise NotImplementedError("No render function")
