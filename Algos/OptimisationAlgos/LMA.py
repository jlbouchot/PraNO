import numpy as np 

from Algos.NumericalAlgos import NumericalAlgos
from Algos.LinearSystems import GaussSeidelSolver as GSSolver 

__author__ = ["Jean-Luc Bouchot"]
__copyright__ = "Jean-Luc Bouchot"
__credits__ = ["Jean-Luc Bouchot"]
__license__ = "GPLv3"
__version__ = "0.1.0-dev"
__maintainer__ = "Jean-Luc Bouchot"
__email__ = "jlbmathit@gmail.com"
__status__ = "Development"
__created__ = "2020/04/06"
__lastmodified__ = "2020/04/08"

class VanillaLM(NumericalAlgos): 
    '''Basic Levenberg-Marquardt algorithm without any extra feature''' 

    def __init__(self, anOperator, x0, aJacobian, aHessian, nbIter, omega0):  
        super().__init__(anOperator)

        self.x0 = x0
        self.nbIter = nbIter 

        self.J = aJacobian
        self.H = aHessian

        self.d = 1 if type(x0) == float else len(x0)

        self.algoName = "Vanilla LM optimisation"

        curValue = self.lhs(x0)

        self.estimates = np.zeros([self.d,nbIter+1])
        self.estimates[:,0] = x0

        self.errors = 100000*np.ones([nbIter+1])
        self.errors[0] = curValue

        self.omegas = np.zeros([nbIter+1])
        self.omegas[0] = omega0

    def solve(self): 
        xOld = self.x0
        for oneiter in range(0,self.nbIter) : 
            curGradient = self.J(xOld)
            xNew = xOld - np.dot(np.linalg.pinv(self.H(xOld) + self.omegas[oneiter]*np.identity(self.d)),curGradient)

            self.errors[oneiter+1] = self.lhs(xNew)
            if self.errors[oneiter+1] < self.errors[oneiter]:
                self.omegas[oneiter+1] = self.omegas[oneiter+1]/5
                self.estimates[:,oneiter+1] = xOld
            else :
                self.omegas[oneiter+1] = self.omegas[oneiter+1]*5
                self.estimates[:,oneiter+1] = xNew
                xOld = xNew

        self.xstar = xNew

        return xNew


class ClassicalLM(NumericalAlgos): 
    '''Basic Levenberg-Marquardt algorithm without any extra feature, simpler handling of parameters''' 

    def __init__(self, anOperator, x0, aJacobian, nbIter, omega0):  
        super().__init__(anOperator)

        self.x0 = x0
        self.nbIter = nbIter 

        self.J = aJacobian

        self.d = 1 if type(x0) == float else len(x0)

        self.algoName = "Classical LM optimisation"

        curValue = self.lhs(x0)

        self.estimates = np.zeros([self.d,nbIter+1])
        self.estimates[:,0] = x0

        self.residuals = np.zeros([len(curValue), nbIter+1])
        self.residuals[:,0] = curValue

        self.errors = 100000*np.ones([nbIter+1])
        self.errors[0] = np.linalg.norm(curValue)

        self.omegas = np.zeros([nbIter+1])
        self.omegas[0] = omega0

    def solve(self): 
        xOld = self.x0
        for oneiter in range(0,self.nbIter) : 
            curGradient = self.J(xOld)
            # Solving the linear system A x step = b for the step 
            b_vector = np.matmul(curGradient.transpose(),self.residuals[:,oneiter])
            A_matrix = np.matmul(curGradient.transpose(),curGradient) + self.omegas[oneiter]*np.identity(self.d)
            # Create a newSolver with the appropriate values 
            # linearSolver = GSSolver(A_matrix, np.zeros([self.d]), b_vector, 500)
            # thisSolver.solve()
            # linearSolver.solve()
            # step = thisSolver.xstar
            # step = linearSolver.xStar
            step = np.dot(np.linalg.pinv(A_matrix),b_vector)
            xNew = xOld - step
            # xNew = xOld - np.dot(np.linalg.pinv(np.matmul(curGradient.transpose(),curGradient) + self.omegas[oneiter]*np.identity(self.d)),np.matmul(curGradient.transpose(),self.residuals[:,oneiter]))

            self.errors[oneiter+1] = np.linalg.norm(self.lhs(xNew))
            if self.errors[oneiter+1] > self.errors[oneiter]:
                self.omegas[oneiter+1] = self.omegas[oneiter+1]*5
                self.estimates[:,oneiter+1] = xOld
                self.residuals[:,oneiter+1] = self.lhs(xOld)
            else :
                self.omegas[oneiter+1] = self.omegas[oneiter+1]/5
                self.estimates[:,oneiter+1] = xNew
                xOld = xNew
                self.residuals[:,oneiter+1] = self.lhs(xNew)

        self.xstar = xNew

        return xNew


class SndOrderLM(NumericalAlgos): 
    '''Levenberg-Marquardt algorithm with second order damping''' 

    def __init__(self, anOperator, x0, aJacobian, nbIter, omega0):  
        super().__init__(anOperator)

        self.x0 = x0
        self.nbIter = nbIter 

        self.J = aJacobian

        self.d = 1 if type(x0) == float else len(x0)

        self.algoName = "Second order LM optimisation"

        curValue = self.lhs(x0)

        self.estimates = np.zeros([self.d,nbIter+1])
        self.estimates[:,0] = x0

        self.residuals = np.zeros([len(curValue), nbIter+1])
        self.residuals[:,0] = curValue

        self.errors = 100000*np.ones([nbIter+1])
        self.errors[0] = np.linalg.norm(curValue)

        self.omegas = np.zeros([nbIter+1])
        self.omegas[0] = omega0

    def solve(self): 
        xOld = self.x0
        for oneiter in range(0,self.nbIter) : 
            curGradient = self.J(xOld)
            curHessian = np.matmul(curGradient.transpose(),curGradient)
            xNew = xOld - np.dot(np.linalg.pinv(curHessian + self.omegas[oneiter]*np.diag(np.diag(curHessian))),np.matmul(curGradient.transpose(),self.residuals[:,oneiter]))

            self.errors[oneiter+1] = np.linalg.norm(self.lhs(xNew))
            if self.errors[oneiter+1] > self.errors[oneiter]:
                self.omegas[oneiter+1] = self.omegas[oneiter+1]*5
                self.estimates[:,oneiter+1] = xOld
                self.residuals[:,oneiter+1] = self.lhs(xOld)
            else :
                self.omegas[oneiter+1] = self.omegas[oneiter+1]/5
                self.estimates[:,oneiter+1] = xNew
                xOld = xNew
                self.residuals[:,oneiter+1] = self.lhs(xNew)

        self.xstar = xNew

        return xNew

