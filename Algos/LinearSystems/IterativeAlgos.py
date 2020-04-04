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

class IterativeSolver(LinearSystems):
    """Generic class for solving linear systems by iterative methods.""" 

    def __init__(self, A, x0, rhs, nbIter):
        super().__init__(A, rhs, nbIter, x0)

        self.estimates = np.zeros([self.d, nbIter + 1])
        self.estimates[:,0] = x0
        self.errors = 10000*np.ones([nbIter + 1])
        self.errors[0] = np.linalg.norm(np.matmul(self.A,x0) - self.b)

        

    def solve_triangle(self, b): 
        xOut = np.zeros([self.d])
        for i in range(0,self.d): 
            minusPart = 0
            for j in range(0,i): 
                minusPart = minusPart + self.matLHS[i,j]*xOut[j]
            xOut[i] = 1.0/self.matLHS[i,i]*(b[i] - minusPart)

        return xOut


    def solve_diagonal(self, b):
        xOut = np.zeros([self.d])
        for i in range(0,self.d): 
            xOut[i] = b[i]/self.matLHS[i,i]

        return xOut

class JacobiSolver(IterativeSolver): 

    def __init__(self, A, x0, rhs, nbIter):
        super().__init__(A, x0, rhs, nbIter)
        self.matLHS = np.diag(np.diag(self.A)) # Contains the matrix for the LHS in matLHS x = b - matRHS x
        self.matRHS = self.A - self.matLHS
        
        
    def solve(self): 
        
        for oneiter in range(0,self.nbIter): 
            curRHS = self.b - np.matmul(self.matRHS,self.estimates[:,oneiter])
            xNew = self.solve_diagonal(curRHS)
            self.estimates[:,oneiter+1] = xNew
            self.errors[oneiter+1] = np.linalg.norm(np.matmul(self.A,xNew) - self.b)

        self.xStar = xNew

class GaussSeidelSolver(IterativeSolver): 

    def __init__(self, A, x0, rhs, nbIter):
        super().__init__(A, x0, rhs, nbIter)
        self.matLHS = np.zeros([self.d,self.d])
        for i in range(0,self.d): 
            for j in range(0,i+1): 
                self.matLHS[i,j] = self.A[i,j]
        self.matRHS = self.A - self.matLHS
        
        
    def solve(self): 
        
        for oneiter in range(0,self.nbIter): 
            curRHS = self.b - np.matmul(self.matRHS,self.estimates[:,oneiter])
            xNew = self.solve_triangle(curRHS)
            self.estimates[:,oneiter+1] = xNew
            self.errors[oneiter+1] = np.linalg.norm(np.matmul(self.A,xNew) - self.b)

        self.xStar = xNew
