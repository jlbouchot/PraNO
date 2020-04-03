import numpy as np

from Algos.NumericalAlgos.LinearSystems import LinearSystems

__author__ = ["Jean-Luc Bouchot"]
__copyright__ = "Jean-Luc Bouchot"
__credits__ = ["Jean-Luc Bouchot"]
__license__ = "GPLv3"
__version__ = "0.1.0-dev"
__maintainer__ = "Jean-Luc Bouchot"
__email__ = "jlbmathit@gmail.com"
__status__ = "Development"
__lastmodified__ = "2020/04/03"
__created__ = "2020/04/03"

class SVDBased(LinearSystems):
    """Generic class for solving linear systems by iterative methods.""" 

    __init__(self, A, x0, rhs, nbIter):  
        super().__init__(A, x0, rhs)
        
        self.nbIter = nbIter

        self.estimates = np.zeros([self.d, nbIter + 1])
        self.estimates[:,0] = x0
        self.errors = 10000*np.ones([nbIter + 1])
        self.errors[:,0] = np.linalg.norm(self.A*x0 - self.b)

        
