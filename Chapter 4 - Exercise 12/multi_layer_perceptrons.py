import numpy as np

###############################################################################

class MultiLayerPerceptrons:

    def __init__(self, numIns, numHiddens, numOuts):
        # The extra units are for biases and perminently set to 1
        self.numOuts = numOuts
        self.numIns = numIns + 1
        self.numHiddens = numHiddens + 1
        self.firstLayerWeights = np.random.standard_normal(
            self.numIns * numHiddens
        )
        self.secondLayerWeights = np.random.standard_normal(
            self.numHiddens
        )
        self.hiddenUnits = np.zeros(numHiddens)
        self.hiddenUnits = np.append(1, self.hiddenUnits)
        self.outputs = np.zeros(numOuts)

    def forwardPropagate(self, x):
        x_b = np.append(1, x)
        j = 1
        k = 0
        stride = 0

        for i in range(0, self.numIns):
            while stride != self.firstLayerWeights.size:
                self.hiddenUnits[j] += (self.firstLayerWeights[i+stride] * x_b[i])
                j += 1
                stride += self.numIns
            j = 1
            stride = 0
        self.hiddenUnits[1:self.numHiddens] = self.__activateHiddenUnits()

        for j in range(0, self.numHiddens):
            while stride != self.secondLayerWeights.size:
                self.outputs[k] += (self.secondLayerWeights[i+stride] * self.hiddenUnits[j])
                k += 1
                stride += self.numHiddens
            k = 0
            stride = 0
        self.outputs = self.__activateOutputUnits()

    def __activateHiddenUnits(self):
        return 1 - np.square(
            np.tanh(self.hiddenUnits[1:self.numHiddens])
        )

    def __activateOutputUnits(self):
        return self.outputs

    def computeError(self, t):
        return np.sum(np.power(np.subtract(self.outputs, t), 2))


x = [2, 3]
t = [-2, 1]
nn = MultiLayerPerceptrons(2, 5, 2)
nn.forwardPropagate(np.array([2, 3]))
print nn.outputs
print nn.computeError(t)
