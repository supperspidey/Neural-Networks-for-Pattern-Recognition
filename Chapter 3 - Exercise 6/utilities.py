from scipy.special import expit
import numpy as np

###############################################################################

def sum_of_square_error(X, t, w):
    error = 0
    for n in range(0, len(X)):
        error += np.power((expit(np.inner(w, X[n])) - t[n]), 2)
    return 0.5 * error

def error_derivative(x, t, w):
    a = np.inner(w, x)
    delta = expit(a) - t
    g_prime = expit(a) * (1 - expit(a))

    return delta * g_prime * x

def hyperplane(xmin, xmax, W):
    x_line = np.linspace(xmin, xmax, 10)
    Y = [-(W[0][0]/W[0][1] * x_line + W[0][2]/W[0][1])]
    for n in range(1, len(W)):
        Y = np.concatenate((Y, [-(W[n][0]/W[n][1] * x_line + W[n][2]/W[n][1])]), axis=0)

    return x_line, Y
