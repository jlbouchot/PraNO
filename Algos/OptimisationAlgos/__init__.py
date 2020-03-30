__author__ = "Jean-Luc Bouchot"
__copyright__ = "Copyright 2020, Jean-Luc Bouchot"
__credits__ = "Jean-Luc Bouchot"
__license__ = "GPLv3"
__version__ = "0.1.0-dev"
__maintainer__ = "Jean-Luc Bouchot"
__email__ = "jlbmathit@gmail.com"
__status__ = "Development"
__lastmodified__ = "2020/03/30"
__created__ = "2020/03/22"

__all__ = [
    'GradientDescent', 
    'NewtonsIterations'
]

import numpy as np
eps = np.finfo(float).eps

from .GradientDescent import *
from .NewtonsIterations import *
