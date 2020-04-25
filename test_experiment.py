import pickle
import numpy as np
import matplotlib.pyplot as plt
import time
from dp_cartpole_env import CartpoleSinCosEnvironment as d_env
dt = 0.05
max_force = 5
env = d_env(dt, min_bounds = - max_force, max_bounds = max_force)

true_mpc = pickle.load(open("Trajectories/xs_mpc_H30_S2_N200.pkl", "rb"))
model_mpc = pickle.load(open("Trajectories/xs_predicted_mpc_H30_S2_N200.pkl", "rb"))
mpc_actions = pickle.load(open("Trajectories/us_mpc_H30_S2_N200.pkl", "rb"))

true_ilqr = pickle.load(open("Trajectories/xs_ilqr_H200_S2_N200.pkl", "rb"))
model_ilqr = pickle.load(open("Trajectories/xs_predicted_ilqr_H200_S2_N200.pkl", "rb"))
ilqr_actions = pickle.load(open("Trajectories/us_ilqr_H200_S2_N200.pkl", "rb"))


for x in true_mpc:
    env.render(x)
    time.sleep(0.06)
env.viewer.close()