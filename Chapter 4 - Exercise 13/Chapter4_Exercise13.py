import csv
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from multi_layer_perceptrons import MultiLayerPerceptrons

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

nn1 = MultiLayerPerceptrons(numIns=1, numHiddens=2, numOuts=1)
Y1, E1 = nn1.train(X, T, maxIters=90, eta_wji=0.05, eta_wkj=0.08)

################################################################################

# anim = plot(
#     X,
#     Y1,
#     E1,
#     "Demonstration of Training Neural Network Using Backpropagation Algorithm"
# )
# plt.show(anim)
