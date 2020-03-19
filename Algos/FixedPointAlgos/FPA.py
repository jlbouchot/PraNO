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

class FixedPointProblems(NumericalSolver):
    """Generic class for solving fixed point problems."""

    self.type = "Fixed point problem"

    def __init__(self, anOperator): 
        super().__init__(anOperator)



