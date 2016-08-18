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
        g_prime = self.__firstDerivatives(hiddenUnits)
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

        y, _ = self.__forwardPropagate(X[n])
        print self.__hessianMatrix(X[0], T[0], y, self.w_ji, self.w_kj)

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

    def __firstDerivatives(self, hiddenUnits):
        return 1 - np.square(hiddenUnits)

    def __secondDerivatives(self, hiddenUnits):
        g_prime = self.__firstDerivatives(hiddenUnits)
        return -2 * np.multiply(hiddenUnits, g_prime)

    def __hessianMatrix(self, x, t, y, w_ji, w_kj):
        #   Note: Each row is treated as a column
        identityMatrix = np.identity(len(w_ji) + len(w_kj))
        hessian = []
        for col in range(0, len(identityMatrix)):
            v = identityMatrix[col]
            hessian.append(self.__R_operate(x, t, y, v, w_ji, w_kj))

        return np.array(hessian).T

    def __R_operate(self, x, t, y, v, w_ji, w_kj):
        v_ji = np.array(v[0:len(w_ji)])
        v_kj = np.array(v[len(w_ji):len(v)])
        x_b = np.append(1, x)

        R_aj = np.zeros(self.numHiddens)
        aj = np.zeros(self.numHiddens)
        ji = 0
        for j in range(1, len(R_aj)):
            aj[j] = np.sum(np.multiply(x_b, w_ji[ji:ji+self.numIns]))
            R_aj[j] = np.sum(np.multiply(x_b, v_ji[ji:ji+self.numIns]))
            ji += self.numIns

        zj = self.__activate(aj, type=ActivationTypes.hiddenLayer)
        zj[0] = 1
        zj_p = self.__firstDerivatives(zj)
        zj_p[0] = 0
        R_zj = np.multiply(zj_p, R_aj)

        R_yk = np.zeros(self.numOuts)
        kj = 0
        for k in range(0, len(R_yk)):
            R_yk[k] = np.add(
                np.sum(np.multiply(w_kj[kj:kj+self.numHiddens], R_zj)),
                np.sum(np.multiply(v_ji[kj:kj+self.numHiddens], zj))
            )
            kj += self.numHiddens

        del_k = np.subtract(y, t)
        R_del_k = np.array(R_yk)
        zj_pp = self.__secondDerivatives(zj)
        R_del_j = np.zeros(self.numHiddens)
        for j in range(1, len(R_del_j)):
            expr1 = zj_pp[j] * R_aj[j] * np.sum(
                np.multiply(del_k, self.w_kj[j:len(self.w_kj):self.numHiddens])
            )
            expr2 = zj_p[j] * np.sum(
                np.multiply(del_k, v_kj[j:len(v_kj):self.numHiddens])
            )
            expr3 = zj_p[j] * np.sum(
                np.multiply(R_del_k, self.w_kj[j:len(self.w_kj):self.numHiddens])
            )
            R_del_j[j] = expr1 + expr2 + expr3

        R_dEdwkj = np.zeros(len(self.w_kj))
        k = 0
        j = 0
        for kj in range(0, len(self.w_kj)):
            R_dEdwkj[kj] = R_del_k[k] * zj[j] + del_k[k] * R_zj[j]
            j += 1
            if j == self.numHiddens:
                j = 0
                k += 1

        R_dEdwji = np.zeros(len(self.w_ji))
        j = 0
        i = 0
        for ji in range(0, len(self.w_ji)):
            R_dEdwji[ji] = x_b[i] * R_del_j[j]
            i += 1
            if i == self.numIns:
                i = 0
                j += 1

        return np.append(R_dEdwji, R_dEdwkj)
