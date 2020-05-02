import numpy as np
import matplotlib.pyplot as plt

def get_x(model):
    k = 0
    x = np.zeros((301, 1))
    for i in model:
        x[k] = i[0]
        k += 1
    return x

def get_angle(model):
    k = 0
    sin_p = np.zeros((301, 1))
    cos_p = np.zeros((301, 1))
    for i in model:
        sin_p[k] = i[2]
        cos_p[k] = i[3]
        k += 1
    return sin_p, cos_p

def get_diff(model1,model2):
    k = 0
    sin_p1 = np.zeros((301, 1))
    cos_p1 = np.zeros((301, 1))
    for i in model1:
        sin_p1[k] = i[2]
        cos_p1[k] = i[3]
        k += 1
    k = 0
    sin_p2 = np.zeros((301, 1))
    cos_p2 = np.zeros((301, 1))
    for i in model2:
        sin_p2[k] = i[2]
        cos_p2[k] = i[3]
        k += 1
    return np.abs(sin_p1)-np.abs(sin_p2),np.abs(cos_p1)-np.abs(cos_p2)

def plot_angle(angle_p,angle_t):
    plt.plot(angle_p)
    plt.plot(angle_t)
    plt.xlabel("Time steps")
    plt.legend(["Predicted", "Actual"])

def plot_tan2(sin1,cos1,sin2,cos2):
    plt.plot(np.arctan2(sin1, cos1))
    plt.plot(np.arctan2(sin2, cos2))
    plt.legend(["Predicted", "Actual"])
    plt.xlabel("Time steps")
    plt.ylabel("Angle")
