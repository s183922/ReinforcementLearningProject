"""
This file may not be shared/redistributed freely. Please read copyright notice in the git repo.
"""
import numpy as np
from iLQR.cost import QRCost
from ilqr_cartpole_example import ilqr
from dp_cartpole_env import CartpoleSinCosEnvironment
import matplotlib.pyplot as plt
import time
import pickle
from scipy.interpolate import interp1d
def render_(xs, env):
    for i in range(len(xs)):
        x = xs[i]
        time.sleep(0.08)
        env.render(x=x)

def cartpole_balance(RecedingHorizon=True, sigma=0.0, time_horizon_length=30, ddp = False):
    N = 200 ## Simulation Time steps
    np.random.seed(1)
    dt = 0.05
    pole_length = 1.0
    action_size = 1
    max_force = 5.0
    
    goal_angle = 0
    init_angle =  np.pi
    x_goal = np.array([0.0, 0.0, np.sin(goal_angle), np.cos(goal_angle), 0.0])
    x0 = np.array([0, 0, np.sin(init_angle), np.cos(init_angle), 0])
    Q = np.diag([1, 0.01, 30, 30, 0.01]) # diag([1.0, 1.0, 1.0, 1.0, 1.0])
    R = 0.01 * np.eye(action_size)

    cost = QRCost(Q, R, QN=None, x_goal=x_goal)
    env = CartpoleSinCosEnvironment(dt=dt, cost=cost, l=pole_length, min_bounds=-max_force, max_bounds=max_force, ddp = ddp)

    x = [x0, ]
    x_model = [x0, ]
    x_current, x_ = x0, x0
    u = np.zeros((N,1))
    for i in range(N):
        if RecedingHorizon:
            xs, us, J_hist = ilqr(env, time_horizon_length, x_current, n_iter = 300, use_linesearch = True, verbose = False, ddp = ddp)
            u[i] = np.copy(us[0])

            #raise NotImplementedError("Call ilqr and obtain u[i] as the first part of the optimised ac
            # tion-sequence")
        elif i == 0:
            xs, us, J_hist = ilqr(env, N, x_current, n_iter=300, use_linesearch=True)
            u[i] = np.copy(us[i])
        else:
            u[i] = np.copy(us[i])



        #us += np.random.normal(0.0, scale=sigma, size=time_horizon_length).reshape(time_horizon_length,1)

        """ Euler dynamics = bad """
        if not RecedingHorizon:
            x_ = env.f(x_, u[i], i)
            x_model.append(x_)
        else:
            x_ = env.f(x_current, u[i], i)
            x_model.append(x_)

        u[i] += np.random.normal(0.0, scale=sigma, size=None)
        """ Truer dynamics RK4 = good """
        x_current = env.step(x_current, us, 1, u[i])[0]

        x.append(x_current)

        #env.render(x_current)
        print(f"Iteration {i}, x={x_current[0]}")

    
    #render_(x,env)
    #env.viewer.close()
    import os, sys; os.chdir(sys.path[0])

    ss = "ddp"
    N1 = time_horizon_length if RecedingHorizon else N
    pickle.dump(x, open(f"Trajectories/xs_{ss}_H{N1}_S{sigma}_N{N}.pkl", "wb"))
    pickle.dump(u, open(f"Trajectories/us_{ss}_H{N1}_S{sigma}_N{N}.pkl", "wb"))
    pickle.dump(x_model, open(f"Trajectories/xs_predicted_{ss}_H{N1}_S{sigma}_N{N}.pkl", "wb"))
    plt.plot(np.squeeze(x)[:, 4])
    plt.plot(np.squeeze(u))
    plt.legend(["Angle", "Action"],)
    # out = f""
    ss = "mpc" if RecedingHorizon else "no_mpc"
    plt.title(ss)
    from irlc import savepdf
    savepdf(f"cartpole_{ss}")
    plt.show()


if __name__ == "__main__":
    sigma = 2
    time_horizon_length = 30  # Control Horizon
    ddp = True
    # Test without receding horizon. This should fail for positive sigma.
    #cartpole_balance(RecedingHorizon=False, sigma=sigma, time_horizon_length=time_horizon_length)

    # Test with receding horizon. This should succeed even for positive sigma.
    cartpole_balance(RecedingHorizon=True, sigma=sigma, time_horizon_length=time_horizon_length, ddp = ddp)
