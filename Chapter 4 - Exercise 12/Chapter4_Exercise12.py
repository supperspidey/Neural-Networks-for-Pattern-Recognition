import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from multi_layer_perceptrons import MultiLayerPerceptrons

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

nn = MultiLayerPerceptrons(numIns=1, numHiddens=15, numOuts=1)

#   Train the neural network
Y, E = nn.train(X, T, maxIters=2000, eta_wji=0.07, eta_wkj=0.07)

################################################################################

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Animate the linear separators
ax1.grid(True)
ax1.set_xlabel("x")
ax1.set_ylabel("y")
ax1.plot(X.flatten(), T.flatten(), 'o')
curve, = ax1.plot([], [], 'r')

###############################################################################

# Plot the error over time
ax2.set_xlim(0, len(E))
ax2.set_ylim(E[-1], E[0])
error, = ax2.plot([], [], 'c')
ax2.set_xlabel("Iterations")
ax2.set_ylabel("Error")
ax2.grid(True)

###############################################################################

lines = [curve, error]
xdata = []
ydata = []

def animate(i):
    xdata.append(i)
    ydata.append(E[i])

    curve.set_data(X.flatten(), Y[i].flatten())
    error.set_data(xdata, ydata)
    return lines

anim = animation.FuncAnimation(
    fig,
    animate,
    frames=len(Y),
    interval=100,
    blit=False,
    repeat=False
)

###############################################################################

plt.show()
