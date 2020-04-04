__author__ = "Jean-Luc Bouchot"
__copyright__ = "Copyright 2020, Jean-Luc Bouchot"
__credits__ = "Jean-Luc Bouchot"
__license__ = "GPLv3"
__version__ = "0.1.0-dev"
__maintainer__ = "Jean-Luc Bouchot"
__email__ = "jlbmathit@gmail.com"
__status__ = "Development"
__created__ = "2020/03/30"
__lastmodified__ = "2020/04/04"

__all__ = [
    'FixedPointAlgos', 
    'OptimisationAlgos', 
    'RootFindingAlgos', 
    'LinearSystems'
]

import numpy as np
eps = np.finfo(float).eps

from .FixedPointAlgos import *
from .OptimisationAlgos import *
from .RootFindingAlgos import *
from .LinearSystems import *
