import numpy as np

from Algos.NumericalAlgos import NumericalAlgos

__author__ = ["Jean-Luc Bouchot"]
__copyright__ = "Jean-Luc Bouchot"
__credits__ = ["Jean-Luc Bouchot"]
__license__ = "GPLv3"
__version__ = "0.1.0-dev"
__maintainer__ = "Jean-Luc Bouchot"
__email__ = "jlbmathit@gmail.com"
__status__ = "Development"
__lastmodified__ = "2021/03/18"
__created__ = "2021/03/18"

class QNewtonType(NumericalAlgos):
    """Generic abstract class for Quasi Newton based optimization algorithms."""

    algoType = "Optimization algorithms"

    def __init__(self, anOperator, x0, nbIter): 
        super().__init__(anOperator)
        self.x0 = x0
        self.d = 1 if type(x0) == float else len(x0)
        self.estimates = np.zeros([self.d,nbIter+1])
        self.estimates[:,0] = x0
        self.nbIter = nbIter

