import numpy as np

###############################################################################

def hyperplane(xmin, xmax, W):
    x_line = np.linspace(xmin, xmax, 10)
    Y = [-(W[0][0]/W[0][1] * x_line + W[0][2]/W[0][1])]
    for n in range(1, len(W)):
        Y = np.concatenate(
            (Y, [-(W[n][0]/W[n][1] * x_line + W[n][2]/W[n][1])]),
            axis=0
        )

    return x_line, Y
