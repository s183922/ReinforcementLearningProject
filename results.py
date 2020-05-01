import pickle
from Cartpole.iLQR.cost import QRCost
from Cartpole.iLQR.ilqr import *
from Cartpole.Visualization import get_angle, get_diff, plot_angle, plot_tan2, get_x
from dp_cartpole_env import CartpoleSinCosEnvironment as d_env

#dt = 0.05
#max_force = 5
#pole_length = 1.0
#action_size = 1
#goal_angle = 0
#init_angle =  np.pi
#x_goal = np.array([0.0, 0.0, np.sin(goal_angle), np.cos(goal_angle), 0.0])
#x0 = np.array([0, 0, np.sin(init_angle), np.cos(init_angle), 0])
#Q = np.diag([1, 0.01, 30, 30, 0.01])  # diag([1.0, 1.0, 1.0, 1.0, 1.0])
#R = 0.01 * np.eye(action_size)
#cost = QRCost(Q, R, QN=None, x_goal=x_goal)
#env = d_env(dt, cost = cost,l=pole_length, min_bounds = - max_force, max_bounds = max_force)

true_mpc = pickle.load(open("Trajectories/xs_mpc_H30_S2_N200.pkl", "rb"))
model_mpc = pickle.load(open("Trajectories/xx_mpc_H30_S2_N200.pkl", "rb"))
mpc_actions = pickle.load(open("Trajectories/us_mpc_H30_S2_N200.pkl", "rb"))
mpc_cost = pickle.load(open("Trajectories/JJ_mpc_H30_S2_N200.pkl", "rb"))

true_ilqr = pickle.load(open("Trajectories/xs_ilqr_H200_S2_N200.pkl", "rb"))
model_ilqr = pickle.load(open("Trajectories/xs_predicted_ilqr_H200_S2_N200.pkl", "rb"))
ilqr_actions = pickle.load(open("Trajectories/us_ilqr_H200_S2_N200.pkl", "rb"))
ilqr_cost = pickle.load(open("Trajectories/JJ_ilqr_H200_S2_N200.pkl", "rb"))

c, c_x, c_u, c_xx, c_ux, c_uu = QRCost.g(self=cost, x=xx[i + 1], u=u[i + 1], i=None, terminal=False)

sin_ilqr_p, cos_ilqr_p = get_angle(model_ilqr)
sin_ilqr_t, cos_ilqr_t = get_angle(true_ilqr)
sin_mpc_p, cos_mpc_p = get_angle(model_mpc)
sin_mpc_t, cos_mpc_t = get_angle(true_mpc)
sin_ilqr_d, cos_ilqr_d = get_diff(model_ilqr,true_ilqr)
sin_mpc_d, cos_mpc_d = get_diff(model_mpc,true_mpc)

x_ilqr_t = get_x(true_ilqr)
x_mpc_t = get_x(true_mpc)

J = np.zeros((201,1))
J1 = np.zeros((201,1))
for i in range(201):
    J[i] = compute_J(env, x_ilqr_t[i], ilqr_actions[i])
    J1[i] = compute_J(env, x_mpc_t[i], mpc_actions[i])
N = 200
L1, L_x1, L_u1, L_xx1, L_ux1, L_uu1 = [None] * (N + 1), [None] * (N + 1), [None] * (N), [None] * (N + 1), [None] * (N), [None] * (N)
L2, L_x2, L_u2, L_xx2, L_ux2, L_uu2 = [None] * (N + 1), [None] * (N + 1), [None] * (N), [None] * (N + 1), [None] * (N), [None] * (N)

for i in range(200):
    L1[i], L_x1[i], L_u1[i], L_xx1[i], L_ux1[i], L_uu1[i] = env.g(x_ilqr_t[i], ilqr_actions[i], i, terminal=False,compute_gradients=True)
    L2[i], L_x2[i], L_u2[i], L_xx2[i], L_ux2[i], L_uu2[i] = env.g(x_mpc_t[i], mpc_actions[i], i, terminal=False,compute_gradients=True)

plt.plot(L1)
plt.xlabel("Time steps")
plt.ylabel("J")
plt.title("Cost pr iteration")
plt.show()

plt.plot(ilqr_cost)
plt.xlabel("Time steps")
plt.ylabel("J")
plt.title("Cost pr iteration")
plt.show()

plt.plot(L2)
plt.xlabel("Time steps")
plt.ylabel("J")
plt.title("Cost pr iteration")
plt.show()

plot_tan2(sin_ilqr_p,cos_ilqr_p,sin_ilqr_t,cos_ilqr_t)
plt.title("iLQR trajectory")
plt.show()

plot_tan2(sin_mpc_p,cos_mpc_p,sin_mpc_t,cos_mpc_t)
plt.title("MPC trajectory")
plt.show()

plt.plot(ilqr_cost)
plt.xlabel("Time steps")
plt.ylabel("J")
plt.title("Cost pr iteration")
plt.show()

plt.plot(mpc_cost)
plt.xlabel("Time steps")
plt.ylabel("J")
plt.title("Cost pr iteration")
plt.show()

plot_angle(sin_ilqr_p,sin_ilqr_t)
plt.title("iLQR trajectory sinus")
plt.ylabel("Sin to angle")
plt.show()
plot_angle(cos_ilqr_p,cos_ilqr_t)
plt.title("iLQR trajectory cosinus")
plt.ylabel("Cos to angle")
plt.show()
plot_angle(sin_mpc_p,sin_mpc_t)
plt.title("MPC trajectory sinus")
plt.ylabel("Sin to angle")
plt.show()
plot_angle(cos_mpc_p,cos_mpc_t)
plt.title("MPC trajectory cosinus")
plt.ylabel("Cos to angle")
plt.show()

plt.plot(sin_ilqr_d)
plt.title("Difference in predicted and actual sin to angle iLQR")
plt.xlabel("Time steps")
plt.ylabel("Difference")
plt.show()

plt.plot(cos_ilqr_d)
plt.title("Difference in predicted and actual cos to angle iLQR")
plt.xlabel("Time steps")
plt.ylabel("Difference")
plt.show()

plt.plot(sin_mpc_d)
plt.title("Difference in predicted and actual sin to angle MPC")
plt.xlabel("Time steps")
plt.ylabel("Difference")
plt.show()

plt.plot(cos_mpc_d)
plt.title("Difference in predicted and actual cos to angle MPC")
plt.xlabel("Time steps")
plt.ylabel("Difference")
plt.show()

#v_ilqr_t = true_ilqr[:][2:4]

#for x in true_mpc:
#    env.render(x)
#    time.sleep(0.06)
#env.viewer.close()