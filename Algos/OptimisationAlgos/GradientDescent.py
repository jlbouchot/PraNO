import numpy as np

from Algos.OptimisationAlgos.GradientDescentAlgos import GradientDescentAlgos

__author__ = ["Jean-Luc Bouchot"]
__copyright__ = "Jean-Luc Bouchot"
__credits__ = ["Jean-Luc Bouchot"]
__license__ = "GPLv3"
__version__ = "0.1.0-dev"
__maintainer__ = "Jean-Luc Bouchot"
__email__ = "jlbouchot@gmail.com"
__status__ = "Development"
__lastmodified__ = "2020/03/22"
__created__ = "2020/03/22"

class VanillaGD(GradientDescentAlgos):
    """Implementation of the classical/vanilla gradient descent."""


    # For an efficient vanilla implementation, we need the jacobian to be specified 
    def __init__(self, anOperator, x0, aJacobian, learningRate, nbIter): 
        self.nbIter = nbIter
        self.stepSize =  learningRate
        self.d = 1 if type(x0) == float else len(x0)
        super().__init__(anOperator, learningRate)

        self.J = aJacobian

        self.estimates.append(x0)
        self.errors = np.zeros(nbIter+1)
        self.errors[0] = anOperator(x0)


    def solve(self): 
        xOld = self.x0
        for oneiter in range(0,self.nbIter) : 
            curGradient = self.J(xOld)
            xNew = xOld - self.stepSize*curGradient
            self.errors[oneiter+1] = self.lhs(xNew)
            self.estimates.append(xNew)
            xOld = xNew

        return xNew

class ArmijoGoldsteinGD(GradientDescentAlgos): 
    """Implementation of gradient descent with armijo-goldstein rule for stepsize."""


    # For an efficient vanilla implementation, we need the jacobian to be specified 
    def __init__(self, anOperator, x0, aJacobian, learningRate, nbIter, armijoConstant): 
        self.nbIter = nbIter
        self.stepSize = np.zeros(nbIter+1)
        self.stepSize[0] = learningRate
        self.d = 1 if type(x0) == float else len(x0)
        super().__init__(anOperator, learningRate)

        self.J = aJacobian
        self.c = armijoConstant

        self.estimates.append(x0)
        self.errors = np.zeros(nbIter+1)
        self.errors[0] = anOperator(x0)

        self.innerIterMax = 20


    def solve(self): 
        xOld = self.x0
        for oneiter in range(0,self.nbIter) : 
            curGradient = self.J(xOld)

            curInnerIter = 0
            curStepSize = self.c*self.stepSize[oneiter] # This will change with time
            xNew = xOld - curStepSize*curGradient
            while (self.lhs(xNew) > self.errors[oneiter] and curInnerIter < self.innerIterMax): 
                # You may uncomment the following line to check if indeed you pass in this loop!
                # print("I'm in outer loop number " + str(oneiter) + " and inside the inner loop number " + str(curInnerIter))
                curStepSize = curStepSize/2
                xNew = xOld - curStepSize*curGradient
                curInnerIter = curInnerIter +1 

            if (self.lhs(xNew) > self.errors[oneiter]): 
                break
            self.stepSize[oneiter+1] = curStepSize
            self.errors[oneiter+1] = self.lhs(xNew)
            self.estimates.append(xNew)
            xOld = xNew

        return xNew

