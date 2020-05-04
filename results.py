from load_trajectories import *
from get_values import get_angle, get_diff, plot_angle, plot_tan2, get_x, plot_cost, get_speed, plot_speed

c1,c2,c3,c4 = np.zeros((200,1)),np.zeros((200,1)),np.zeros((200,1)),np.zeros((200,1))
c11,c22,c33,c44 = np.zeros((200,1)),np.zeros((200,1)),np.zeros((200,1)),np.zeros((200,1))
c_1,c_2,c_3,c_4 = np.zeros((200,1)),np.zeros((200,1)),np.zeros((200,1)),np.zeros((200,1))
c_11,c_22,c_33,c_44 = np.zeros((200,1)),np.zeros((200,1)),np.zeros((200,1)),np.zeros((200,1))

m1, m2, m3, m4 = get_x(true_ilqr),get_x(true_mpc),get_x(true_ddp_ilqr),get_x(true_ddp_mpc)
m11, m22, m33, m44 = get_x(model_ilqr),get_x(model_mpc),get_x(model_ddp_ilqr),get_x(model_ddp_mpc)

for i in range(200):
    c1[i], _, _, _, _, _ = QRCost.g(self=cost, x=m1[i], u=ilqr_actions[i], i=None, terminal=False)
    c2[i], _, _, _, _, _ = QRCost.g(self=cost, x= m2[i], u=mpc_actions[i], i=None, terminal=False)
    c3[i], _, _, _, _, _ = QRCost.g(self=cost, x=m3[i], u=ddp_ilqr_actions[i], i=None, terminal=False)
    c4[i], _, _, _, _, _ = QRCost.g(self=cost, x=m4[i], u=ddp_mpc_actions[i], i=None, terminal=False)
    c11[i], c22[i],c33[i],c44[i] = np.sum(c1),np.sum(c2),np.sum(c3),np.sum(c4)
    c_1[i], _, _, _, _, _ = QRCost.g(self=cost, x=m11[i], u=ilqr_actions[i], i=None, terminal=False)
    c_2[i], _, _, _, _, _ = QRCost.g(self=cost, x=m22[i], u=mpc_actions[i], i=None, terminal=False)
    c_3[i], _, _, _, _, _ = QRCost.g(self=cost, x=m33[i], u=ddp_ilqr_actions[i], i=None, terminal=False)
    c_4[i], _, _, _, _, _ = QRCost.g(self=cost, x=m44[i], u=ddp_mpc_actions[i], i=None, terminal=False)
    c_11[i], c_22[i], c_33[i], c_44[i] = np.sum(c_1), np.sum(c_2), np.sum(c_3), np.sum(c_4)

sin_ilqr_p, cos_ilqr_p = get_angle(model_ilqr)
sin_ilqr_t, cos_ilqr_t = get_angle(true_ilqr)
sin_mpc_p, cos_mpc_p = get_angle(model_mpc)
sin_mpc_t, cos_mpc_t = get_angle(true_mpc)

sin_ddp_p, cos_ddp_p = get_angle(model_ddp_ilqr)
sin_ddp_t, cos_ddp_t = get_angle(true_ddp_ilqr)
sin_ddp1_p, cos_ddp1_p = get_angle(model_ddp_mpc)
sin_ddp1_t, cos_ddp1_t = get_angle(true_ddp_mpc)

ilqr_speed = get_speed(true_ilqr)
mpc_speed = get_speed(true_mpc)
ddp_speed = get_speed(true_ddp_ilqr)
ddp1_speed = get_speed(true_ddp_mpc)

plot_speed(ilqr_speed,mpc_speed,"iLQR","iLQR MPC",fig_nr=9,ylabel="X'",fig="speed_ilqr_mpc")
plot_speed(ddp_speed,ddp1_speed,"DDP","DDP MPC",fig_nr=10,ylabel="X'",fig="speed_ddp_ddpmpc")

plot_cost(c1,c2,c3,c4,title="Actual cost pr iteration",fig="cost_pr_iteration",fig_nr=1)
plot_cost(c11,c22,c33,c44,title="Actual accumulated cost",fig="accumulated_cost",fig_nr=2)
plot_cost(c_1,c_2,c_3,c_4,title="Predicted cost pr iteration",fig="cost_pr_iteration_p",fig_nr=3)
plot_cost(c_11,c_22,c_33,c_44,title="Predicted accumulated cost",fig="accumulated_cost_p",fig_nr=4)

plot_tan2(sin_ilqr_p,cos_ilqr_p,sin_ilqr_t,cos_ilqr_t,title="iLQR trajectory theta",fig_nr=5,fig="ilqr_theta")
plot_tan2(sin_mpc_p,cos_mpc_p,sin_mpc_t,cos_mpc_t,title="iLQR MPC trajectory theta",fig_nr=6,fig="mpc_theta")
plot_tan2(sin_ddp_p,cos_ddp_p,sin_ddp_t,cos_ddp_t,title="DDP trajectory theta",fig_nr=7,fig="ddp_theta")
plot_tan2(sin_ddp1_p,cos_ddp1_p,sin_ddp1_t,cos_ddp1_t,title="DDP MPC trajectory theta",fig_nr=8,fig="ddp1_theta")