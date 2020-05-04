import pickle
from Cartpole.iLQR.cost import QRCost
from Cartpole.iLQR.ilqr import *
from dp_cartpole_env import CartpoleSinCosEnvironment as d_env
dt = 0.05
max_force = 5
pole_length = 1.0
action_size = 1
goal_angle = 0
init_angle =  np.pi
x_goal = np.array([0.0, 0.0, np.sin(goal_angle), np.cos(goal_angle), 0.0])
x0 = np.array([0, 0, np.sin(init_angle), np.cos(init_angle), 0])
Q = np.diag([1, 0.01, 30, 30, 0.01])  # diag([1.0, 1.0, 1.0, 1.0, 1.0])
R = 0.01 * np.eye(action_size)
cost = QRCost(Q, R, QN=None, x_goal=x_goal)
env = d_env(dt, cost = cost,l=pole_length, min_bounds = - max_force, max_bounds = max_force)

true_ilqr = pickle.load(open("Trajectories/xs_ilqr_no_mpc300.pkl", "rb"))
model_ilqr = pickle.load(open("Trajectories/xs_predicted_ilqr_no_mpc300.pkl", "rb"))
ilqr_actions = pickle.load(open("Trajectories/us_ilqr_no_mpc300.pkl", "rb"))

true_mpc = pickle.load(open("Trajectories/xs_ilqr_mpc300.pkl", "rb"))
model_mpc = pickle.load(open("Trajectories/xs_predicted_ilqr_mpc300.pkl", "rb"))
mpc_actions = pickle.load(open("Trajectories/us_ilqr_mpc300.pkl", "rb"))

true_ddp_ilqr = pickle.load(open("Trajectories/xs_ddp_no_mpc300.pkl", "rb"))
model_ddp_ilqr = pickle.load(open("Trajectories/xs_predicted_ddp_no_mpc300.pkl", "rb"))
ddp_ilqr_actions = pickle.load(open("Trajectories/us_ddp_no_mpc300.pkl", "rb"))

true_ddp_mpc = pickle.load(open("Trajectories/xs_ddp_mpc300.pkl", "rb"))
model_ddp_mpc = pickle.load(open("Trajectories/xs_predicted_ddp_mpc300.pkl", "rb"))
ddp_mpc_actions = pickle.load(open("Trajectories/us_ddp_mpc300.pkl", "rb"))