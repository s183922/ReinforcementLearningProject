
import os, sys
path = sys.path[0]
os.chdir(path)
import gym
from collections import OrderedDict
from AcrobotEnv import AcrobotEnv
from env import SymbolicAcrobot
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