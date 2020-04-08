from Algos.FixedPointAlgos.BanachFixedPoint import * 
from Algos.OptimisationAlgos.GradientDescent import VanillaGD as VGD
from Algos.OptimisationAlgos.GradientDescent import ArmijoGoldsteinGD as AGGD
from Algos.RootFindingAlgos.NewtonsMethod import NewtonsIterations1D as Newton1D
from Algos.RootFindingAlgos.NewtonsMethod import NewtonsIterationsOverdetermined as NewtonND
from Algos.OptimisationAlgos.NewtonsIterations import NewtonOptim as NewtonOptim
from Algos.LinearSystems.IterativeAlgos import JacobiSolver as LinearJacobi
from Algos.LinearSystems.IterativeAlgos import GaussSeidelSolver as GSSolver
from Algos.OptimisationAlgos.LMA import VanillaLM as VanillaLM
from Algos.OptimisationAlgos.LMA import ClassicalLM as ClassicalLM


import numpy as np 
import matplotlib.pyplot as plt

##########################################
## Test the fixed point iterations
##########################################

functionToSolve = lambda x: np.sqrt(x)
x0 = 1.5
nbIter = 10

myFPIterations = FPAlgo(functionToSolve, x0, nbIter)

myFPIterations.solve()
myFPIterations.print()

##########################################
## Test the vanilla gradient descent 
##########################################

functionForVGD = lambda x: (x-15)**2
jacobianForVGD = lambda x: 2*(x-15)

x0 = -0.314
nbIter = 500
learningRate = 0.05

myVGD = VGD(functionForVGD, x0, jacobianForVGD, learningRate, nbIter)
myVGD.solve()
myVGD.print()


##########################################
## Test the Gradient descent with Armijo-Goldstein line search test
##########################################

functionForAGGD = lambda x: (x-5)**2
jacobianForAGGD = lambda x: 2*(x-5)

x0 = 3.314
nbIter = 5000


armijoLearning = 3
armijoTest = 0.9
myAGGD = AGGD(functionForAGGD, x0, jacobianForAGGD, armijoLearning, nbIter, armijoTest)
myAGGD.solve()
myAGGD.print()

##########################################
## Test the 1D Newton method for root finding
##########################################
functionForNewtonRoot = lambda x: (x-5)**2
jacobianForNewtonRoot = lambda x: 2*(x-5)

x0 = 3.314
nbIter = 50

myNewton = Newton1D(functionForNewtonRoot, x0, jacobianForNewtonRoot, nbIter)
myNewton.solve()
myNewton.print()


##########################################
## Test the nD Newton method for root finding
##########################################
functionForNewtonRoot = lambda x: np.array([(x[0]-5)**2, x[0]**2 + x[1]**2-25-16, x[1]-4])
jacobianForNewtonRoot = lambda x: np.array([[2*(x[0]-5), 0], [2*x[0], 2*x[1]], [0, 1] ])

x0 = np.array([3,3])
nbIter = 50

myNewtonND = NewtonND(functionForNewtonRoot, x0, jacobianForNewtonRoot, nbIter)
myNewtonND.solve()
myNewtonND.print()

##########################################
## Test the nD Newton method for optimization
##########################################
# Definition of Powell's function 
functionForNewtonOptim = lambda x: np.array([(x[0]+10*x[1])**2 + 5*(x[2]-x[3])**2 + (x[1]-2*x[2])**4 + 10*(x[0]-x[3])**4])
jacobianForNewtonOptim = lambda x: np.array([2*(x[0]+10*x[1]) + 40*(x[0]-x[3])**3, 20*(x[0]+10*x[1]) + 4*(x[1]-2*x[2])**3, 10*(x[2]-x[3]) - 8*(x[1]-2*x[2])**3, -10*(x[2]-x[3]) - 40*(x[0]-x[3])**3])
hessianForNewtonOptim = lambda x: np.array([[2+120*(x[0]-x[3])**2, 20, 0, -120*(x[0]-x[3])**2], [20, 200+12*(x[1]-2*x[2])**2, -24*(x[1]-2*x[2])**2, 0], [0, -24*(x[1]-2*x[2])**2, 10+48*(x[1]-x[2])**2, -10], [-120*(x[0]-x[3])**2, 0, -10, 10+120*(x[0]-x[3])**2] ])

x0 = np.array([3,-1,0,1])
nbIter = 50

myNewtonOptim = NewtonOptim(functionForNewtonOptim, x0, jacobianForNewtonOptim, hessianForNewtonOptim, nbIter)
myNewtonOptim.solve()
myNewtonOptim.print()

##########################################
## Test Jacobi iterations for solving square linear systems
##########################################

# define a square matrix with non 0 diagonal entries 
matSize = 500
aMatrixForJacobi = 0.5*np.random.randn(matSize,matSize)
# MAke it likely diagonal dominant
aMatrixForJacobi = aMatrixForJacobi + np.diag(0.5*np.random.randn(matSize)+matSize/2)
# Define a right hand side
trueSolution = np.ones(matSize)
b_rhs = np.matmul(aMatrixForJacobi,trueSolution)

# Create the solver 
x0Jacobi = np.random.rand(matSize)
nbIterJacobi = 50
myJacobi = LinearJacobi(aMatrixForJacobi, x0Jacobi, b_rhs, nbIterJacobi)
# And solve!
myJacobi.solve()
print("Norm of the error for Jacobi solver is {}".format(np.linalg.norm(myJacobi.xStar - trueSolution)))

##########################################
## Test Gauss Seidel iterations for solving square linear systems
##########################################

## define a square matrix with non 0 diagonal entries 
#matSize = 50
#aMatrixForJacobi = 0.5*np.random.randn(matSize,matSize)
## MAke it likely diagonal dominant
#aMatrixForJacobi = aMatrixForJacobi + np.diag(0.5*np.random.randn(matSize)+matSize/2)
## Define a right hand side
#trueSolution = np.ones(matSize)
#b_rhs = np.matmul(aMatrixForJacobi,trueSolution)

## Create the solver 
#x0Jacobi = np.random.rand(matSize)
#nbIterJacobi = 50

myGS = GSSolver(aMatrixForJacobi, x0Jacobi, b_rhs, nbIterJacobi)
# And solve!
myGS.solve()
print("Norm of the error for GS solver is {}".format(np.linalg.norm(myGS.xStar - trueSolution)))


##########################################
## Tests for Vanilla Levenberg Marquard -- see p.248 of Nocedal
##########################################
# Target function is defined f(t) = x1 +tx2 + t^2x3 + x4e^{-tx5}
def targetBloodDecay(params, samplingPts): 
    return np.array([params[0] + t*params[1] + t**2*params[2] + params[3]*np.exp(-t*params[4]) for t in samplingPts])

def residualFunction(params,samplingPts, targetValues): 
    return targetBloodDecay(params, samplingPts) - targetValues

# Define the gradient with respect to the parameters for a set of ts
def jacobianOfTarget(params, samplingPts): 
    return np.array([[1,t,t**2,np.exp(-t*params[4]), -t*params[3]*np.exp(-t*params[4])] for t in samplingPts])

# Now let's go on to the (approximate) Hessian
def approxHessian(params, samplingPts): 
    curJacobian = jacobianOfTarget(params, samplingPts)
    return np.matmul(curJacobian.transpose(), curJacobian)

theta = np.array([2,0.5, 1, 2.5, 0.5]) # Set of parameters 
ts = range(0,10)
ys = targetBloodDecay(theta,ts)

#print(ys)
#print(jacobianOfTarget(theta,ts))
#print(approxHessianForLM(theta,ts))

x0 = np.array([1,1,1,1,1])
nbIter = 50
omega0 = 10

funToOptLM = lambda x: 1/2*np.linalg.norm(residualFunction(x,ts, ys))**2

myVanillaLevenberg = VanillaLM(funToOptLM, x0, lambda x: np.matmul(jacobianOfTarget(x,ts).transpose(),residualFunction(x,ts,ys)), lambda x: approxHessian(x,ts), nbIter, omega0)
myVanillaLevenberg.solve()
myVanillaLevenberg.print()

##########################################
## Tests for Vanilla Levenberg Marquard
##########################################

# Function is defined as f(t) = a*cos(b*t) -> Find a and b
def funForLM(param, samplingPts): 
    return np.array([param[0]*t + param[1] for t in samplingPts])

def gradientForLM(param, samplingPts, targetYs): 
    return np.matmul(np.array([[t, 1 ] for t in samplingPts]).transpose(),funForLM(param,samplingPts)-targetYs)

def approxHessianForLM(param, samplingPts): 
    jacobian = np.array([[t, 1 ] for t in samplingPts])
    return np.matmul(jacobian.transpose(),jacobian)

theta = np.array([2,0.5]) # Set of parameters 
ts = range(1,50)
ys = funForLM(theta,ts)

x0 = np.array([4,1])
nbIter = 50
omega0 = 10

functionToOptLM = lambda x: 1/2*np.linalg.norm(funForLM(x,ts) - ys)**2
myVanillaLevenberg = VanillaLM(functionToOptLM, x0, lambda x: gradientForLM(x,ts, ys), lambda x: approxHessianForLM(x,ts), nbIter, omega0)
myVanillaLevenberg.solve()
myVanillaLevenberg.print()


##########################################
## Tests for parameter estimations in LM with sum_i |r(t_i; theta) - y_i|^2
## i.e. only need to specify the jacobian and the parametrized residual
##########################################
# Target function is defined f(t) = x1 +tx2 + t^2x3 + x4e^{-tx5}
def targetBloodDecay(params, samplingPts): 
    return np.array([params[0] + t*params[1] + t**2*params[2] + params[3]*np.exp(-t*params[4]) for t in samplingPts])

def residualFunction(params,samplingPts, targetValues): 
    return targetBloodDecay(params, samplingPts) - targetValues

# Define the gradient with respect to the parameters for a set of ts
def jacobianOfTarget(params, samplingPts): 
    return np.array([[1,t,t**2,np.exp(-t*params[4]), -t*params[3]*np.exp(-t*params[4])] for t in samplingPts])


theta = np.array([2,0.5, 1, 2.5, 0.5]) # Set of parameters 
ts = range(0,10)
ys = targetBloodDecay(theta,ts)

#print(ys)
#print(jacobianOfTarget(theta,ts))
#print(approxHessianForLM(theta,ts))

x0 = np.array([1,1,1,1,1])
nbIter = 50
omega0 = 10


myClassicalLM = ClassicalLM(lambda x: residualFunction(x, ts, ys), x0, lambda x: jacobianOfTarget(x,ts), nbIter, omega0)
myClassicalLM.solve()
myClassicalLM.print()
