import csv
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from multi_layer_perceptrons import BackPropagation
from multi_layer_perceptrons import CentralDifferences

################################################################################

def plot(X, Y, E, title):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(title, fontsize=17)
    ax1.grid(True)
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.plot(X.flatten(), T.flatten(), 'o')
    curve, = ax1.plot([], [], 'r')

    ax2.set_xlim(0, len(E))
    ax2.set_ylim(E[-1], E[0])
    error, = ax2.plot([], [], 'c')
    ax2.set_xlabel("Iterations")
    ax2.set_ylabel("Error")
    ax2.grid(True)

    lines = [curve, error]
    xdata = []
    ydata = []

    anim = animation.FuncAnimation(
        fig,
        animate,
        frames=len(Y),
        interval=100,
        fargs=(X, Y, E, lines, xdata, ydata),
        blit=False,
        repeat=False
    )

    return anim

def animate(frame, X, Y, E, lines, xdata, ydata):
    xdata.append(frame)
    ydata.append(E[frame])

    lines[0].set_data(X.flatten(), Y[frame].flatten())
    lines[1].set_data(xdata, ydata)
    return lines

################################################################################

#   Load data from the file
X = []
T = []

with open('data.csv') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    for row in reader:
        if row[0] == "x" and row[1] == "t":
            continue
        else:
            X.append([float(row[0])])
            T.append([float(row[1])])

X = np.array(X)
T = np.array(T)

################################################################################

nn1 = BackPropagation(numIns=1, numHiddens=30, numOuts=1)
Y1, E1 = nn1.train(X, T, maxIters=90, eta_wji=0.05, eta_wkj=0.08)

################################################################################

nn2 = CentralDifferences(numIns=1, numHiddens=15, numOuts=1)
Y2, E2 = nn2.train(X, T, maxIters=90, eps=0.03, eta_wji=0.05, eta_wkj=0.08)

################################################################################

anim1 = plot(X, Y1, E1, "Demonstration of Training Neural Network Using Backpropagation Algorithm")
anim2 = plot(X, Y2, E2, "Demonstration of Training Neural Network Using Central Differences")
plt.show([anim1, anim2])

################################################################################

# timeCD = []
# timeBP = []
# numWeights_cd = []
# numWeights_bp = []
# for i in range(2, 20):
#     nn = BackPropagation(numIns=1, numHiddens=i, numOuts=1)
#     numWeights_bp.append(nn.numIns * (nn.numHiddens-1) + nn.numHiddens * nn.numOuts)
#     startTime = time.time()
#     nn.train(X, T, maxIters=90, eta_wji=0.05, eta_wkj=0.08)
#     endTime = time.time()
#     timeBP.append(endTime - startTime)
#
# for i in range(2, 20):
#     nn = CentralDifferences(numIns=1, numHiddens=i, numOuts=1)
#     numWeights_cd.append(nn.numIns * (nn.numHiddens-1) + nn.numHiddens * nn.numOuts)
#     startTime = time.time()
#     nn.train(X, T, maxIters=90, eps=0.03, eta_wji=0.05, eta_wkj=0.08)
#     endTime = time.time()
#     timeCD.append(endTime - startTime)
#
# plt.figure()
# plt.plot(numWeights_cd, timeCD, 'r')
# plt.xlabel("Weights")
# plt.ylabel("Time (s)")
# plt.title("Time vs Weights (Central Differences)")
# plt.grid(True)
#
# plt.figure()
# plt.plot(numWeights_bp, timeBP, 'r')
# plt.xlabel("Weights")
# plt.ylabel("Time (s)")
# plt.title("Time vs Weights (Backpropagation)")
# plt.grid(True)
#
# plt.show()
