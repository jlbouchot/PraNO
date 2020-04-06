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
__lastmodified__ = "2020/04/06"
__created__ = "2020/04/06"

class VanillaLM(NumericalAlgos): 
    '''Basic Levenberg-Marquardt algorithm without any extra feature''' 

    def __init__(self, anOperator, x0, aJacobian, aHessian, nbIter, omega0):  
        super().__init__(anOperator, x0)
        self.nbIter = nbIter 

        self.J = aJacobian
        self.H = aHessian

        self.d = 1 if type(x0) == float else len(x0)

        self.algoName = "Vanilla LM optimisation"

        curValue = self.J(x0)

        self.errors = 100000*np.ones([len(curValue),nbIter+1])
        self.errors[:,0] = curValue

        self.omegas = np.zeros([nbIter+1])
        self.omegas[0] = omega0

    def solve(self): 
        xOld = self.x0
        for oneiter in range(0,self.nbIter) : 
            curGradient = self.J(xOld)
            xNew = xOld - np.dot(np.linalg.pinv(self.H(xOld) + self.omegas[nbIter]*np.identity(self.d)),curGradient)

            self.errors[:,oneiter+1] = self.lhs(xNew)
            if self.errors[:,oneiter+1] < self.errors[:,oneiter]:
                self.omegas[nbIter+1] = self.omegas[nbIter+1]/5
                self.estimates[:,oneiter+1] = xOld
            else :
                self.omegas[nbIter+1] = self.omegas[nbIter+1]*5
                self.estimates[:,oneiter+1] = xNew
                xOld = xNew

        self.xstar = xNew

        return xNew

