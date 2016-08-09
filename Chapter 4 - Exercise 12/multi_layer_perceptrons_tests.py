import unittest
import numpy as np
from multi_layer_perceptrons import MultiLayerPerceptrons

class MultiLayerPerceptronsTests(unittest.TestCase):
    def setUp(self):
        self.testInstance = MultiLayerPerceptrons(numIns=1, numHiddens=5)

    def testParameters(self):
        self.assertEqual(
            self.testInstance.inputDimension,
            2,
            "The input dimension should have been numIns(1) + 1 = 2."
        )
        self.assertEqual(
            self.testInstance.numberOfHiddenUnits,
            6,
            "The number of hidden units should have been numHiddens(5) + 1 = 6."
        )
        self.assertEqual(
            self.testInstance.firstLayerWeights.shape,
            ((1 + 1) * 5, ),
            "The dimension of firstLayerWeights should have been (10, )."
        )
        self.assertEqual(
            self.testInstance.secondLayerWeights.shape,
            (5 + 1, ),
            "The dimension of firstLayerWeights should have been (6, )."
        )

###############################################################################

if __name__ == '__main__':
    unittest.main()
