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
__lastmodified__ = "2020/03/30"
__created__ = "2020/03/19"

class FPA(NumericalAlgos):
    """Generic class for solving fixed point problems."""

    algoType = "Fixed point problem"

    def __init__(self, anOperator): 
        super().__init__(anOperator)



