# This is a commit

import numpy as np

###############################################################################

class ActivationTypes:
    hiddenLayer = 1
    outputLayer = 2

class MultiLayerPerceptrons:

    def __init__(self, numIns, numHiddens, numOuts):
        # The extra units are for biases and perminently set to 1
        self.numOuts = numOuts
        self.numIns = numIns + 1
        self.numHiddens = numHiddens + 1

        self.w_ji = np.random.standard_normal(
            self.numIns * numHiddens
        )
        self.dE_dwji = np.zeros(self.numIns * numHiddens)

        self.w_kj = np.random.standard_normal(
            self.numHiddens * numOuts
        )
        self.dE_dwkj = np.zeros(self.numHiddens * numOuts)

        self.hiddenUnits = np.append(1, np.zeros(numHiddens))
        self.outputs = np.zeros(numOuts)

    def forwardPropagate(self, x):
        x_b = np.append(1, x)

        ji = 0
        for j in range(1, self.numHiddens):
            self.hiddenUnits[j] += np.sum(
                np.multiply(self.w_ji[ji:ji+self.numIns], x_b)
            )
            ji += self.numIns
        self.hiddenUnits[1:self.numHiddens] = self.__activate(
            self.hiddenUnits[1:self.numHiddens],
            ActivationTypes.hiddenLayer
        )

        kj = 0
        for k in range(0, self.numOuts):
            self.outputs[k] += np.sum(
                np.multiply(self.w_kj[kj:kj+self.numHiddens], self.hiddenUnits)
            )
            kj += self.numHiddens
        self.outputs = self.__activate(
            self.outputs,
            ActivationTypes.outputLayer
        )

    def backPropagate(self, x, t):
        x_b = np.append(1, x)
        del_k = np.subtract(self.outputs, t)
        kj = 0
        for k in range(0, self.numOuts):
            self.dE_dwkj[kj:kj+self.numHiddens] = del_k[k] * self.hiddenUnits
            kj += self.numHiddens

        del_j = np.zeros(self.numHiddens)
        kj = 0
        for k in range(0, self.numOuts):
            del_j = np.add(
                del_j,
                del_k[k] * self.w_kj[kj:kj+self.numHiddens]
            )
            kj += self.numHiddens

        ji = 0
        g_prime = self.__computeHiddenUnitsTanhDerivative()
        for j in range(1, self.numHiddens):
            self.dE_dwji[ji:ji+self.numIns] = g_prime[j] * del_j[j] * x_b
            ji += self.numIns

        print self.dE_dwji
        print self.dE_dwkj

    def __activate(self, a, type):
        if type == ActivationTypes.hiddenLayer:
            return np.tanh(a)
        else:
            return a

    def __computeHiddenUnitsTanhDerivative(self):
        return 1 - np.square(self.hiddenUnits)

x = np.array([2, 3])
t = np.array([-2, 1])
nn = MultiLayerPerceptrons(2, 5, 2)
nn.forwardPropagate(x)
nn.backPropagate(x, t)
