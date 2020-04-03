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
__lastmodified__ = "2020/04/03"
__created__ = "2020/04/03"

class LinearSystems(NumericalAlgos):
    """Generic abstract class for finding solutions to linear systems."""

    algoType = "Linear system solver"

    def __init__(self, A, rhs, nbIter = 1, x0 = None): # nbIter is set to 1 in case of direct methods.
        super().__init__(lambda x: A*x - rhs)  # Doing this brings all the theory from root finding!
        if x0 is not None: # This is to distinguish between iterative and direct methods
            self.x0 = x0
        self.nbIter = nbIter 
        self.d = 1 if type(x0) == float else len(x0)

        self.A = A
        self.b = rhs # We'll pretty much be solving Ax = b

