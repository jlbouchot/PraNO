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
__lastmodified__ = "2020/03/30"
__created__ = "2020/03/29"

class RootFindingAlgos(NumericalAlgos):
    """Generic abstract class for root finding algorithms."""

    algoType = "Root finding algorithm"

    def __init__(self, anOperator, x0, nbIter): 
        super().__init__(anOperator)
        self.x0 = x0
        self.d = 1 if type(x0) == float else len(x0)
        self.estimates = np.zeros([self.d,nbIter+1])
        self.nbIter = nbIter
        self.estimates[:,0] = x0

