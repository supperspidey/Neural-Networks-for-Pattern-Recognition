import numpy as np
from scipy.special import expit

###############################################################################

def activate(x, w):
    if np.inner(x, w) >= 0:
        return 1
    else:
        return -1

def hyperplane(xmin, xmax, W):
    x_line = np.linspace(xmin, xmax, 10)
    Y = [-(W[0][0]/W[0][1] * x_line + W[0][2]/W[0][1])]
    for n in range(1, len(W)):
        Y = np.concatenate(
            (Y, [-(W[n][0]/W[n][1] * x_line + W[n][2]/W[n][1])]),
            axis=0
        )

    return x_line, Y

def perceptron(X, t):
    X_1 = np.append(X, np.ones([len(X), 1]), axis=1)
    w = 0.1 * np.ones(X_1.shape[1])
    converged = False
    aPointMisclassified = False

    while not converged:
        for n in range(0, len(X_1)):
            if activate(X_1[n], w) != t[n]:
                aPointMisclassified = True
                w = w + X_1[n] * t[n]
            else:
                aPointMisclassified = False
                continue

        if not aPointMisclassified:
            converged = True

    return [w]
