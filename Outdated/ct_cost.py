"""
This file may not be shared/redistributed freely. Please read copyright notice in the git repo.
"""
import sympy as sym

class SymbolicMayerLagrangeCost:
    """
    Symbolic MayerLagrange cost function
    """
    def sym_J(self, t0, tF, x0, xF):
        # compute Mayer term
        raise NotImplementedError()

    def sym_w(self, x, u, t):
        # compute Lagrange term
        raise NotImplementedError()
        # return sum( [du ** 2 for du in u] )

def mat(x):
    return sym.Matrix(x) if x is not None else x

def sym2np(x):
    if x is None:
        return x
    f = sym.lambdify([], x)
    return f()

class SymbolicQRCost(SymbolicMayerLagrangeCost):
    def __init__(self, Q=None, R=None, x_target=None, c=0, xF_linear=None, x_linear=None):
        self.Q = sym.Matrix(Q) if Q is not None else Q
        self.R = sym.Matrix(R) if R is not None else R
        self.x_target = mat(x_target)
        self.x_linear = mat(x_linear)
        self.xF_linear = sym.Matrix(xF_linear) if xF_linear is not None else xF_linear
        self.c = c
        # as numpy arrays for fast access
        self.Q_np = sym2np(self.Q)
        self.R_np = sym2np(self.R)
        self.c_np = sym2np(self.c)
        self.x_target_np = sym2np(self.x_target)
        self.x_linear_np = sym2np(self.x_linear)

    def sym_J(self, t0, tF, x0, xF):
        sxF = sym.Matrix(xF)
        J = sym.Matrix( [[0.0]] )

        if self.xF_linear is not None:
            J = J + sxF.transpose() @ self.xF_linear
        return J[0,0]


    def sym_w(self, x, u, t):
        '''
        Implements:

        w = 0.5 * ((x-xt)' * Q * (x-xt) + u' * R * u) + c

        '''
        um = sym.Matrix(u)
        xm = sym.Matrix(x)

        w = sym.Matrix( [[0.0]] ) + sym.Matrix([[self.c]])
        if self.x_target is not None:
            xm = xm - self.x_target

        if self.R is not None:
            w += 0.5 * um.transpose() @ self.R @ um
        if self.Q is not None:
            w += 0.5 * xm.transpose() @ self.Q @ xm
        w = w[0,0]
        return w
