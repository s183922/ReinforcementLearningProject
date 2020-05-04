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
    N = 300 ## Simulation Time steps
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

        elif i == 0:
            xs, us, J_hist = ilqr(env, N+1, x_current, n_iter=300, use_linesearch=True, verbose = True, ddp = ddp)
            u = np.copy(us)
            x = np.vstack((np.asarray(x0), env.step(x0, us + np.random.normal(0, sigma, N+1).reshape(-1,1) , N_steps = N)))
            x_model = xs[:-1]





        if  RecedingHorizon:
            """ Euler dynamics = bad """
            x_ = env.f(x_current, u[i], i)
            x_model.append(x_)
            """ Truer dynamics RK4 = good """
            us += np.random.normal(0, sigma, time_horizon_length).reshape(-1,1)
            x_current = env.step(x_current, us, 1)[0]
            x.append(x_current)

        #env.render(x_current)
        print(f"Iteration {i}, x={x[i][0]}")

    
#    render_(x,env)
 #   env.viewer.close()

  #  render_(x_model, env)
  #  env.viewer.close()
    import os, sys; os.chdir(sys.path[0])

    ss = "ddp" if ddp else "ilqr"
    ss1 = "mpc" if RecedingHorizon else "no_mpc"

    pickle.dump(x, open(f"Trajectories/xs_{ss}_{ss1}300.pkl", "wb"))
    pickle.dump(u, open(f"Trajectories/us_{ss}_{ss1}300.pkl", "wb"))
    pickle.dump(x_model, open(f"Trajectories/xs_predicted_{ss}_{ss1}300.pkl", "wb"))
    plt.plot(np.squeeze(x)[:, 4])
    plt.plot(np.squeeze(u))
    plt.legend(["Angle", "Action"],)

    plt.title(ss + " " + ss1)
    from irlc import savepdf
    savepdf(f"cartpole_{ss}")
    plt.show()


if __name__ == "__main__":
    sigma = 2
    time_horizon_length = 30  # Control Horizon

    # Test without receding horizon. This should fail for positive sigma.
    #cartpole_balance(RecedingHorizon=False, sigma=sigma, time_horizon_length=time_horizon_length, ddp = False)
    #cartpole_balance(RecedingHorizon=False, sigma=sigma, time_horizon_length=time_horizon_length, ddp = True)
    # Test with receding horizon. This should succeed even for positive sigma.
    #cartpole_balance(RecedingHorizon=True, sigma=sigma, time_horizon_length=time_horizon_length, ddp = False)
    cartpole_balance(RecedingHorizon=True, sigma=sigma, time_horizon_length=time_horizon_length, ddp = True)
