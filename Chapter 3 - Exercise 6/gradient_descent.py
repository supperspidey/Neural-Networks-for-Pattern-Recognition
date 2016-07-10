import numpy as np
from scipy.special import expit

###############################################################################

def sum_of_square_error(X, t, w):
    error = 0
    for n in range(0, len(X)):
        error += np.power((expit(np.inner(w, X[n])) - t[n]), 2)
    return 0.5 * error

###############################################################################

def error_derivative(x, t, w):
    a = np.inner(w, x)
    delta = expit(a) - t
    g_prime = expit(a) * (1 - expit(a))

    return delta * g_prime * x

###############################################################################

def gradient_descent_batch(X, t, eta=0.01, maxIters=1000):
    X_1 = np.append(X, np.ones([len(X), 1]), axis=1)
    w = 0.1 * np.ones(X_1.shape[1])
    W = [w]
    E = [sum_of_square_error(X_1, t, w)]
    error=0.001

    for i in range(0, maxIters):
        dEn_dw = np.zeros(len(w))
        for n in range(0, len(X_1)):
            derivative = error_derivative(X_1[n], t[n], w)
            dEn_dw = np.add(dEn_dw, derivative)
        w = w - eta * dEn_dw
        W = np.concatenate((W, [w]), axis=0)
        E.append(sum_of_square_error(X_1, t, w))

    return W, E

###############################################################################

def gradient_descent_stochastic(X, t, eta=0.01, maxIters=1000):
    X_1 = np.append(X, np.ones([len(X), 1]), axis=1)
    w = 0.1 * np.ones(X_1.shape[1])
    X_1 = np.append(X_1, t.reshape((len(t), 1)), axis=1)
    error=0.001
    W = [w]
    E = [sum_of_square_error(X_1[:, 0:3], X_1[:, 3], w)]

    for i in range(0, maxIters):
        # np.random.shuffle(X_1)
        for n in range(0, len(X_1)):
            w = w - eta * error_derivative(X_1[n, 0:3], X_1[n, 3], w)
            W = np.concatenate((W, [w]), axis=0)
            E.append(sum_of_square_error(X_1[:, 0:3], X_1[:, 3], w))

    return W, E
