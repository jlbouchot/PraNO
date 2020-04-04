import numpy as np

from Algos.LinearSystems.LinearSystems import LinearSystems

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
    """Class to solve linear system in a single iterations, via an SVD.""" 
    
    def __init__(self, anOperator, rhs): 
        super().__init__(anOperator, rhs)

        self.algoName = "SVD Direct linear solver"


    def solve(self): 
        # Compute the SVD 

        # Ax = y <=> x = VS^{-1}U^Ty (with the appropriate care for 0 sing. values.)
        self.xstar = xNew

        return xNew    

