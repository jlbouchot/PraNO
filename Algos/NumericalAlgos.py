import numpy as np

__author__ = ["Jean-Luc Bouchot"]
__copyright__ = "Jean-Luc Bouchot"
__credits__ = ["Jean-Luc Bouchot"]
__license__ = "GPLv3"
__version__ = "0.1.0-dev"
__maintainer__ = "Jean-Luc Bouchot"
__email__ = "jlbmathit@gmail.com"
__status__ = "Development"
__lastmodified__ = "2020/03/30"
__created__ = "2020/03/19"

class NumericalAlgos():
    """Generic class for a numerical solver."""
    def __init__(self, anOperator): 
        self.lhs = anOperator
        self.estimates = [] # This will contain the various estimations, in case of iterative algorithms
        self.xstar = None

    def print(self): 
        print("Results from {} is {}.".format(self.algoName, self.xstar))

    def solve(self):
        return self.solve()

