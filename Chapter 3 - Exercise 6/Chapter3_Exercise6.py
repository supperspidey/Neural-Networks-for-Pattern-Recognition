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
with open('data.csv', 'rb') as csvfile:
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

# Animate the linear separators
fig = plt.figure()
ax = plt.axes(
    xlim=(np.min(X[:, 0]), np.max(X[:, 0])),
    ylim=(np.min(X[:, 1]), np.max(X[:, 1]))
)
ax.grid(True)
ax.set_xlabel("x1")
ax.set_ylabel("x2")
ax.set_title("Demonstration of Gradient Descent")
classA, = ax.plot(X[t == 1, 0], X[t == 1, 1], 'x')
classB, = ax.plot(X[t == 0, 0], X[t == 0, 1], 'o')
line_s, = ax.plot(x_line, Y_s[-1])
line_b, = ax.plot(x_line, Y_b[-1])
ax.legend(
    [classA, classB, line_s, line_b],
    ['Class A', 'Class B', 'Stochastic', 'Batch']
).draggable()

def animate(i):
    line_s.set_data(x_line, Y_s[i])
    line_b.set_data(x_line, Y_b[i])
    return line_s,

anim = animation.FuncAnimation(
    fig,
    animate,
    frames=len(Y_b),
    interval=1,
    blit=False,
    repeat=False
)

###############################################################################

# Plot the error over time
plt.figure()
plt.axes(
    xlim=(0, len(E_s)),
    ylim=(E_b[-1], E_b[0])
)
error_b, = plt.plot(range(1, len(E_b) + 1), E_b)
error_s, = plt.plot(range(1, len(E_s) + 1), E_s)
plt.xlabel("Iterations")
plt.ylabel("Error")
plt.title("Sum-of-Square Error Over All Iterations")
plt.legend([error_b, error_s], ['Batch', 'Stochastic'])
plt.grid(True)

###############################################################################

plt.show()
