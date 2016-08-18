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

        self.w_kj = np.random.standard_normal(
            self.numHiddens * numOuts
        )

        self.hiddenUnits = np.append(1, np.zeros(numHiddens))

    def __forwardPropagate(self, x):
        x_b = np.append(1, x)
        hiddenUnits = np.append(1, np.zeros(self.numHiddens-1))
        outputs = np.zeros(self.numOuts)

        ji = 0
        for j in range(1, self.numHiddens):
            hiddenUnits[j] += np.sum(
                np.multiply(self.w_ji[ji:ji+self.numIns], x_b)
            )
            ji += self.numIns
        hiddenUnits[1:self.numHiddens] = self.__activate(
            hiddenUnits[1:self.numHiddens],
            ActivationTypes.hiddenLayer
        )

        kj = 0
        for k in range(0, self.numOuts):
            outputs[k] += np.sum(
                np.multiply(self.w_kj[kj:kj+self.numHiddens], hiddenUnits)
            )
            kj += self.numHiddens

        outputs = self.__activate(
            outputs,
            ActivationTypes.outputLayer
        )

        return outputs, hiddenUnits

    def __backPropagate(self, x, t, outputs, hiddenUnits):
        x_b = np.append(1, x)
        del_k = np.subtract(outputs, t)
        kj = 0
        dE_dwji = np.zeros(self.numIns * (self.numHiddens-1))
        dE_dwkj = np.zeros(self.numHiddens * self.numOuts)
        for k in range(0, self.numOuts):
            dE_dwkj[kj:kj+self.numHiddens] = del_k[k] * hiddenUnits
            kj += self.numHiddens

        del_j = np.zeros(self.numHiddens)
        kj = 0
        for k in range(0, self.numOuts):
            del_j = np.add(
                del_j,
                del_k[k] * np.array(self.w_kj[kj:kj+self.numHiddens])
            )
            kj += self.numHiddens

        ji = 0
        g_prime = self.__computeHiddenUnitsDerivative(hiddenUnits)
        for j in range(1, self.numHiddens):
            dE_dwji[ji:ji+self.numIns] = g_prime[j] * del_j[j] * x_b
            ji += self.numIns

        return dE_dwji, dE_dwkj

    def train(self, X, T, maxIters=100, eta_wji=0.05, eta_wkj=0.05):
        Y = []
        E = []
        for epoch in range(0, maxIters):
            y = []
            error = 0

            for n in range(0, len(X)):
                outputs, hiddenUnits = self.__forwardPropagate(X[n])
                y.append(outputs)
                error += self.__error(outputs, T[n])
                dEn_dwji, dEn_dwkj = self.__backPropagate(
                    X[n], T[n], outputs, hiddenUnits
                )
                self.w_ji = np.add(self.w_ji, -eta_wji * dEn_dwji)
                self.w_kj = np.add(self.w_kj, -eta_wkj * dEn_dwkj)

            Y.append(y)
            E.append(error)

        return np.array(Y), np.array(E)

    def predict(self, X):
        T = []
        for n in range(0, len(X)):
            outputs, _ = self.__forwardPropagate(X[n])
            T.append(outputs)
        T = np.array(T)
        return T

    def __error(self, y, t):
        error = 0
        for k in range(0, len(y)):
            error += np.square(y[k] - t[k])
        return 0.5 * error

    def __activate(self, a, type):
        if type == ActivationTypes.hiddenLayer:
            return np.tanh(a)
        else:
            return a

    def __computeHiddenUnitsDerivative(self, hiddenUnits):
        return 1 - np.square(hiddenUnits)
