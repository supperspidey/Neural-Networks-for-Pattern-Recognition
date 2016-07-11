import numpy as np
import csv
from perceptron import perceptron
from utilities import hyperplane
import matplotlib.pyplot as plt
from matplotlib import animation

###############################################################################

# Load data from file
X = []
t = []
with open('Chapter 3 - Exercise 9/data.csv', 'rb') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    for row in reader:
        if row[0] == "x1" and row[1] == "x2" and row[2] == "label":
            continue
        else:
            X.append([float(row[0]), float(row[1])])
            t.append(float(row[2]))

X = np.array(X)
t = np.array(t)

###############################################################################

W = perceptron(X, t)

###############################################################################

x_line, Y = hyperplane(np.min(X[:, 0]), np.max(X[:, 0]), W)

###############################################################################

fig, ax = plt.subplots()
ax.set_xlim(np.min(X[:, 0]), np.max(X[:, 0]))
ax.set_ylim(np.min(X[:, 1]), np.max(X[:, 1]))
ax.grid(True)
ax.set_xlabel("x1")
ax.set_ylabel("x2")
ax.set_title("Demonstration of Perceptron Algorithm")
ax.plot(X[t == 1, 0], X[t == 1, 1], 'x')
ax.plot(X[t == -1, 0], X[t == -1, 1], 'o')
line, = ax.plot([], [], 'r')

def animate(frame):
    line.set_data(x_line, Y[frame])
    return line

anim = animation.FuncAnimation(
    fig,
    animate,
    frames=len(W),
    interval=2000,
    blit=False,
    repeat=False
)

###############################################################################

plt.show()
