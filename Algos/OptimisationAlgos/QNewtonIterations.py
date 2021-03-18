import numpy as np

from Algos.OptimisationAlgos.QNewtonFamily import QNewtonType

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

class QNewtonOptim(QNewtonType): 
    """Implementation of the Quasi Newton Newton iterations with line search."""


    # For an efficient vanilla implementation, we need the jacobian to be specified 
    # Ideally, one would return an error or, at the very least, a warning in case the derivative gets closer to 0
    def __init__(self, anOperator, x0, aJacobian, Hinv_at0 = None, update_type = 'BFGS', nbIter = 500, tol = 1e-5): 
        '''
        Initialise a Quasi Newton solver. 
        * anOperatorn   : the function we are trying to minimize 
        * x0            : a starting point -- will be defaulted to the null vector 
        * aJacobian     : a handler to the computation of the Jacobian
        * Hinv_at0      : the approximate Hessian at 0. (defaulted to the identity matrix)
        * update_type   : the type of (inverse) approximate Hessian update. Can be BFGS (default) or SR1
        '''

        super().__init__(anOperator, x0, nbIter)


        self.n = len(x0)
        self.J = aJacobian
        if not Hinv_at0: 
            self.B = np.eye(self.n) 
        else: 
            self.B = Hinv_at0

        self.algoName = update_type + " Quasi Newton optimisation with line search"
        if update_type is "BFGS": 
            self.update = lambda x0,x1,g0,g1: self.udpateHinv_BFGS(x0,x1,g0,g1)
        elif update_type is "SR1": 
            self.update = lambda x0,x1,g0,g1: self.udpateHinv_SR1(x0,x1,g0,g1)
            
        self.tol = tol



    def solve(self): 
    
        def backtrack(alpha, rho, func, xk, pk): # This should be in a separate class to make sure it is usable to other algorithms 
            x1 = xk+alpha*pk
            f_at_0 = func(xk)
            while(func(x1) >= f_at_0 and alpha > 0.01): 
                alpha *= rho 
                x1 = xk+alpha*pk
            return alpha
    
        x0 = self.x0
        grad0 = self.J(x0)
        
        p0 = -np.matmul(self.B,grad0)
        
        # Proceed with the first iteration to get started 
        # alpha =  backtrack(1, 0.95, self.lhs, x0, p0)
        alpha = 0.001
        x1 = x0 + alpha*p0
        grad1 = self.J(x1)
        self.update(x0,x1,grad0,grad1)
        
        nbIter = 1
        
        # Until gradient is small enough
        while( (nbIter <= self.nbIter) and (np.linalg.norm(grad1) > self.tol) ): 
            x0 = x1 
            grad0 = grad1 
            # Compute next direction 
            p0 = -np.matmul(self.B,grad0)
            
            # Ensure Wolfe conditions 
            alpha = backtrack(0.5 + nbIter/2/self.nbIter, 0.95, self.lhs, x0, p0)
            x1 = x0 + alpha*p0 
            grad1 = self.J(x1)
            print("x1 = {} and x0 = {}".format(x1,x0))
            
            # Update the inverse Hessian 
            self.update(x0,x1,grad0,grad1)
            
            nbIter += 1
            
        print(x1)
        return x1


    def udpateHinv_BFGS(self,xk, xkp1, gradk, gradkp1): # We probably could manage to keep x's and grad's inside somewhere? 
        yk = gradkp1 - gradk
        sk = xkp1 - xk
        rhok = 1/np.dot(yk,sk)
        outersy = np.outer(sk,yk)
        self.B = np.matmul(np.eye(self.n) - rhok*outersy, np.matmul(self.B, np.eye(self.n) - rhok*outersy.transpose()) ) + rhok*np.outer(sk,sk)
    
    def udpateHinv_SR1(self,xk, xkp1, gradk, gradkp1): # We probably could manage to keep x's and grad's inside somewhere? 
        yk = gradkp1 - gradk
        sk = xp1 - xk
        rhok = 1/np.dot(yk,sk)
        outersy = np.outer(sk,yk)
        self.B = np.matmul(np.eye(self.n) - rhok*outersy, np.matmul(self.B, np.eye(self.n) - rhok*outersy.transpose()) ) + rhok*np.outer(sk,sk)


