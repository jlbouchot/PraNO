import numpy as np

__author__ = ["Jean-Luc Bouchot"]
__copyright__ = "Jean-Luc Bouchot"
__credits__ = ["Jean-Luc Bouchot"]
__license__ = "GPLv3"
__version__ = "0.1.0-dev"
__maintainer__ = "Jean-Luc Bouchot"
__email__ = "jlbouchot@gmail.com"
__status__ = "Development"
__lastmodified__ = "2020/03/19"
__created__ = "2020/03/19"

class FPAlgo(FixedPointProblems):
    """Implementation of the classical fixed point algorithm."""


    # It is the duty of the user to make sure that the FP iterations is indeed converging. 
    # The reason for this is that, should we have checks here, we wouldn't be able to demonstrate 
    # counter examples. 
    def __init__(self, anOperator, x0, nbIter): 
        super().__init__(anOperator)
        self.nbIter = nbITer
        self.x0 = x0 
        self.d = length(x0)
        self.iterates.append()


    def solve(): 
        xOld = self.x0
        for oneiter in xrange(0,self.nbIter) : 
            xNew = self.lhs(xOld)
            self.iterates.append(xNew)
            xOld = xNew

        return xNew

