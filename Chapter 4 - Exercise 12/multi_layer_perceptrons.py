import numpy as np

###############################################################################

class MultiLayerPerceptrons:

    def __init__(self, numIns, numHiddens):
        # The extra units are for biases and perminently set to 1
        self.inputDimension = numIns + 1
        self.numberOfHiddenUnits = numHiddens + 1
        self.firstLayerWeights = np.ones(
            self.inputDimension * numHiddens
        )
        self.secondLayerWeights = np.ones(self.numberOfHiddenUnits)
