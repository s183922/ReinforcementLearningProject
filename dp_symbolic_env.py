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
    def __init__(self, dt, cost, ddp = False):
        self.dt = dt
        self.cost = cost
        """ Initialize symbolic variables representing inputs and actions. """
        u = symv("u", self.action_size)
        x = symv('x', self.state_size)


        """ 
            y is a symbolic variable representing the system dynamics y = f(xs, us, dt)
            It's equal to x_{k+1} = [ x_{k+1}, x'_{k+1}, sin(theta_{k+1}), cos(theta_{k+1}, theta'_{k+1} ]
            Each element in x_{k+1} is a function of the state and action f(x1, x2, x3, x4, x5, u)
        """
        y = self.sym_f_discrete(x, u, dt)

        """ compute the symbolic derivate of y wrt. z = (x,u): dy/dz 
            
            Each element in y is differentiated with respect to each of the six parameters x1, x2, x3, x4, x5, u.
            Thus yielding in a 5 x 6 matrix.
        """
        self.dy_dz = sym.Matrix([[sym.diff(f, zi) for zi in list(x)+list(u)] for f in y])

        """ Define (numpy) functions giving next state and the derivatives """
        self.f_z = sym.lambdify((tuple(x), tuple(u)), self.dy_dz, 'numpy')

        self.f_discrete = sym.lambdify((tuple(x), tuple(u)), y, 'numpy')

        if ddp:
            """ Compute the Hessian - the double derivatives 
                Each element in dy_dz is differentiated with respect to each parameter x1, x2, x3, x4, x5, u.
                Resulting in a 30 x 6 matrix. Each row represent the derivatives of en element in dy_dz 
                The last column is f_uu
                Every 6th row is f_ux
                The rest is f_xx            
            """
            self.dy_dzdz = sym.Matrix([[sym.diff(f, zi) for zi in list(x) + list(u)] for f in self.dy_dz])
            self.f_zz = sym.lambdify((tuple(x), tuple(u)), self.dy_dzdz, 'numpy')



    def f(self, x, u, i, compute_jacobian=False, compute_Hessian = False):
        """
        return f(x,u), f_x, f_u, and Hessians (not implemented)
        where f_x is the derivative df/dx, etc.
        """
        fx = np.asarray( self.f_discrete(x, u) )
        if compute_jacobian:
            J = self.f_z(x, u)
            f_xx, f_ux, f_uu = None, None, None
            if compute_Hessian:
                H = self.f_zz(x, u)
                """ H:
                 0  [f1_x1x1, f1_x1x2, f1_x1x3, f1_x1x4, f1_x1x5, f1_x1u]
                 1  [f1_x2x1, f1_x2x2, f1_x2x3, f1_x2x4, f1_x2x5, f1_x2u]
                 2  [f1_x3x1, f1_x3x2, f1_x3x3, f1_x3x4, f1_x3x5, f1_x3u]
                    .               .           .           .           .
                    .               .           .           .           .
                 5  [f1_ux1,  f1_ux2,  f1_ux3,  f1_ux4,  f1_ux5,   f1_uu]
                 6  [f2_x1x1, f2_x1x2, f2_x1x3, f2_x1x4, f2_x1x5, f2_x1u]
                  .               .           .           .           .
                  .               .           .           .           .
                 30 [f5_ux1,  f5_ux2,  f5_ux3,  f5_ux4,  f5_ux5,   f5_uu]
                
                    f_uu : Every sixth element in the last column 
                    f_ux: Every sixth row minus the last column
                    f_xx: everything else                
                """
                f_uu = H[self.state_size:H.shape[0]:self.state_size+1, self.state_size]
                f_ux = H[5:H.shape[0]:6, :self.state_size]

                f_xx_idx = np.ones(H.shape[0], dtype=np.bool)
                f_xx_idx[self.state_size:H.shape[0]:self.state_size+1] = 0
                f_xx = H[f_xx_idx, :self.state_size].reshape(self.state_size, self.state_size, self.state_size)
                """ 
                    f_uu should take the form (1x5): [f1_uu, f2_uu, f3_uu, f4_uu, f5_uu]
                    f_ux should take the form (5x5): [[f1_x1u, f1_x2u, f1_x3u, f1_x4u, f1_x5_u], [f2_x1u, f2_x2u, f2_x3u, f2_x4u, f2_x5_u], ...,
                                                     [f5_x1u, f5_x2u, f5_x3u, f5_x4u, f5_x5_u]]
                    f_xx should take the form (5x5x5): [[[f1_x1x1, f1_x2x1, f1_x3x1, f1_x4x1, f1_x5x1], ... [f1_x1x5, f1_x2x5, f1_x3x5, f1_x4x5, f1_x5x5]] ,
                                                         [f2_x1x1, f2_x2x1, f2_x3x1, f2_x4x1, f2_x5x1], ... [f2_x1x5, f2_x2x5, f2_x3x5, f2_x4x5, f2_x5x5]] 
                """

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
