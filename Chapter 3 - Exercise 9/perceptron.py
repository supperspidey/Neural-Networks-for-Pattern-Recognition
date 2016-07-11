import numpy as np
from scipy.special import expit

###############################################################################

def activate(x, w):
    if np.inner(x, w) >= 0:
        return 1
    else:
        return -1

###############################################################################

def perceptron(X, t):
    X_1 = np.append(X, np.ones([len(X), 1]), axis=1)
    w = 0.1 * np.ones(X_1.shape[1])
    W = [w]
    converged = False
    aPointMisclassified = False

    while not converged:
        for n in range(0, len(X_1)):
            if activate(X_1[n], w) != t[n]:
                aPointMisclassified = True
                w = w + X_1[n] * t[n]
                W = np.concatenate((W, [w]), axis=0)
            else:
                aPointMisclassified = False
                continue

        if not aPointMisclassified:
            converged = True

    return W
