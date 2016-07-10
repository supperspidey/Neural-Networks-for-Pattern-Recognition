import numpy as np
import csv
from perceptron import perceptron
from utilities import hyperplane
import matplotlib.pyplot as plt

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

w = perceptron(X, t)

###############################################################################

x_line, Y = hyperplane(np.min(X[:, 0]), np.max(X[:, 0]), w)

###############################################################################

plt.figure()
plt.plot(X[t == 1, 0], X[t == 1, 1], 'x')
plt.plot(X[t == -1, 0], X[t == -1, 1], 'o')
plt.plot(x_line, Y[0])

plt.show()
