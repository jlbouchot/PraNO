import numpy as np

from Algos.RootFindingAlgos.RootFindingAlgos import RootFindingAlgos

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

class NewtonsIterations1D(RootFindingAlgos): 
    """Implementation of the classical/vanilla Newton iterations in 1 D."""


    # For an efficient vanilla implementation, we need the jacobian to be specified 
    # Ideally, one would return an error or, at the very least, a warning in case the derivative gets closer to 0
    def __init__(self, anOperator, x0, aJacobian, nbIter): 
        # self.nbIter = nbIter
        #self.d = 1
        super().__init__(anOperator, x0, nbIter)

        self.J = aJacobian

        #self.estimates.append(x0)
        self.errors = np.zeros([self.d, nbIter+1])
        self.errors[:,0] = anOperator(x0)

        self.algoName = "Newton method 1D"


    def solve(self): 
        xOld = self.x0
        for oneiter in range(0,self.nbIter) : 
            curGradient = self.J(xOld)
            xNew = xOld - self.errors[:,oneiter]/curGradient
            self.errors[:,oneiter+1] = self.lhs(xNew)
            self.estimates[:,oneiter+1] = xNew
            xOld = xNew

        self.xstar = xNew

        return xNew

class NewtonsIterationsOverdetermined(RootFindingAlgos): 
    """Implementation of the classical/vanilla Newton iterations in 1 D."""


    # For an efficient vanilla implementation, we need the jacobian to be specified 
    # Ideally, one would return an error or, at the very least, a warning in case the derivative gets closer to 0
    def __init__(self, anOperator, x0, aJacobian, nbIter): 
        super().__init__(anOperator, x0, nbIter)

        self.J = aJacobian

        self.errors = np.zeros(nbIter+1)
        curValue = anOperator(x0)
        self.errors = 100000*np.ones([len(curValue),nbIter+1])
        self.errors[:,0] = curValue

        self.algoName = "Newton method overdetermined"


    def solve(self): 
        xOld = self.x0

        for oneiter in range(0,self.nbIter) : 
            curGradient = self.J(xOld)
            xNew = xOld - np.dot(np.linalg.pinv(curGradient),self.errors[:,oneiter])
            self.errors[:,oneiter+1] = self.lhs(xNew)
            self.estimates[:,oneiter+1] = xNew
            xOld = xNew

        self.xstar = xNew

        return xNew
