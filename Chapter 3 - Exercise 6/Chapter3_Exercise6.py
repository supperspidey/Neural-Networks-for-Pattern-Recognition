import numpy as np
import csv
import matplotlib.pyplot as plt
from matplotlib import animation
from gradient_descent import gradient_descent_batch
from gradient_descent import gradient_descent_stochastic
from utilities import hyperplane
import time

###############################################################################

# Load data from file
X = []
t = []
with open('Chapter 3 - Exercise 6/data.csv', 'rb') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    for row in reader:
        if row[0] == "x1" and row[1] == "x2" and row[2] == "label":
            continue
        else:
            X.append([float(row[0]), float(row[1])])
            t.append(float(row[2]))

X = np.array(X)
t = np.array(t)

# Pass the data and class labels to the gradient descent function
start = time.time()
W_s, E_s = gradient_descent_stochastic(X, t, 0.1, maxIters=10)
end = time.time()
print "Time elapsed for stochastic: " + str(end - start) + " secs"
start = time.time()
W_b, E_b = gradient_descent_batch(X, t, 0.1, maxIters=2000)
end = time.time()
print "Time elapsed for batch: " + str(end - start) + " secs"

# Generate a hyperplane from the weight vector
x_line, Y_s = hyperplane(np.min(X[:, 0]), np.max(X[:, 0]), W_s)
x_line, Y_b = hyperplane(np.min(X[:, 0]), np.max(X[:, 0]), W_b)

###############################################################################

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Animate the linear separators
ax1.set_xlim(np.min(X[:, 0]), np.max(X[:, 0]))
ax1.set_ylim(np.min(X[:, 1]), np.max(X[:, 1]))
ax1.grid(True)
ax1.set_xlabel("x1")
ax1.set_ylabel("x2")
ax1.set_title("Demonstration of Gradient Descent")
classA, = ax1.plot(X[t == 1, 0], X[t == 1, 1], 'x')
classB, = ax1.plot(X[t == 0, 0], X[t == 0, 1], 'o')
line_s, = ax1.plot([], [], 'r')
line_b, = ax1.plot([], [], 'c')
ax1.legend(
    [line_s, line_b],
    ['Stochastic', 'Batch']
).draggable()

###############################################################################

# Plot the error over time
ax2.set_xlim(0, len(E_s))
ax2.set_ylim(E_b[-1], E_b[0])
error_b, = ax2.plot([], [], 'c')
error_s, = ax2.plot([], [], 'r')
ax2.set_xlabel("Iterations")
ax2.set_ylabel("Error")
ax2.set_title("Sum-of-Square Error Over All Iterations")
ax2.legend([error_s, error_b], ['Stochastic', 'Batch'])
ax2.grid(True)

###############################################################################

lines = [line_s, line_b, error_s, error_b]
xdata = []
ydata_s = []
ydata_b = []

def animate(i):
    xdata.append(i)
    ydata_s.append(E_s[i])
    ydata_b.append(E_b[i])

    line_s.set_data(x_line, Y_s[i])
    line_b.set_data(x_line, Y_b[i])
    error_b.set_data(xdata, ydata_b)
    error_s.set_data(xdata, ydata_s)
    return lines

anim = animation.FuncAnimation(
    fig,
    animate,
    frames=len(Y_b),
    interval=1,
    blit=False,
    repeat=False
)

###############################################################################

plt.show()
