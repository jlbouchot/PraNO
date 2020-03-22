Numerical optimization -- A practical overview 
==============================================

Introduction
------------

This repository serves serves both as a collection of numerical methods and as a teaching tool. 

All remarks should be directed to me: jlbmathit@gmail.com 

As a collection of methods with the basic maths needed, I try to keep all files independent from one another, such that the interested reader may skip directly to the algorithm of their choice. 
Note however that there is inherent complexification in algorithms, and it may be difficult to go for a proximal method / ADMM without a more than basic understanding of gradient descent. 

This repository is intended as a review of various optimization methods for practitioners and optimization-enthusiasts alike. 
The goal is to give the reader a practical understanding of when to use and not to use certain algorithms.
Even though I try to keep the mathematical/technical content to a minimum, I will not sacrifice the mathematical rigour. 

What can I see here? 
--------------------

Non-ordered list of algorithms which are (will be?) covered here, with their current development status. 
The novice reader is strongly advised to look at the Introductory page to get some familiarity with the topics covered. 
Once enough algorithms are described, this list will be updated to reflect what I believe is an increasing sequence in complexity - This being an absolutely subjective measure, some people's opinions may diverge. 
Note that this is not an non-overlapping set of the world of numerical optimization (for instance, LM algo is **A** nonlinear least squares, Dogleg is **A** trust-region approach, etc ... )
* Gradient descent (Not done)
* Conjugate gradient (Not done)
* [(Banach) Fixed-point iteration -- the simple case](./FPA/FixedPointTheory.ipynb)
* Lasso (Not done)
* Elastic-net (Not done)
* Simplex method (Not done)
* Least-squares (Not done)
* Levenberg-Marquardt -- Nonlinear least-squares (Not done)
* Non-negative least-squares (Not done)
* Dogleg method (Not done)
* BFGS (Not done)
* Bisection (Not done)
* Newton's method (Not done)
* Quasi-Newton (Not done)
* Alternating direction of method of multiplier (Not done)
* Primal-dual minimization (Not done)
* Interior point methods (Not done)
* Gauss-Newton (Not done)
* (Projected) Landweber (Not done)
* POCS -- Projection On Convex Sets (Not done)
* (Block) coordinate descent


Some maths that may be required in at least more than one algorithm
* Global overview of convex optimization 
* KKT conditions 
* Augmented Lagrangian 
* Proximal operators, Moreau envelope, Moreau decomposition

Some applications
-----------------

(This is still work in progress and the list of applications will follow as I continue developing the other parts)

* Image denoising 
* Bundle adjustment and camera calibration
* Learning with proximal methods and regularization 

Some readings of interest
-------------------------

* Nocedal, Wright, _Numerical Optimization_, Springer, 2006. 
* Hastie, Tibshirani, Wainwright, _Statistical learning with sparsity: The Lasso and generalizations_, Chapman & Hall, 2015. 
* Boyd, Vandenberghe, _Convex optimization_. Cambridge University Press, 2004.

