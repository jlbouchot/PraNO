import numpy as np

from Algos.NumericalAlgos import NumericalAlgos

__author__ = ["Jean-Luc Bouchot"]
__copyright__ = "Jean-Luc Bouchot"
__credits__ = ["Jean-Luc Bouchot"]
__license__ = "GPLv3"
__version__ = "0.1.0-dev"
__maintainer__ = "Jean-Luc Bouchot"
__email__ = "jlbouchot@gmail.com"
__status__ = "Development"
__lastmodified__ = "2020/03/22"
__created__ = "2020/03/22"

class GradientDescentAlgos(NumericalAlgos):
    """Generic abstract class for optimising with gradient descent algorithms."""

    algoType = "Gradient descent algorithm"

    def __init__(self, anOperator, x0): 
        super().__init__(anOperator)
        self.x0 = x0



