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
Y, E = nn.train(X, T, maxIters=200, eta_wji=0.02, eta_wkj=0.04)
print "Error: " + str(E)

################################################################################

#   Show the result
fig, ax = plt.subplots()
ax.grid(True)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.plot(X.flatten(), T.flatten(), 'o')
curves, = ax.plot([], [], 'r')

def animate(frame):
    curves.set_data(X.flatten(), Y[frame].flatten())
    return curves

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
