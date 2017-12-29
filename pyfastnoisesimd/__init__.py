# -*- coding: utf-8 -*-
########################################################################
#
#       PyFastNoiseSIMD
#       License: BSD 3-clause
#       Created: August 13, 2017
#       Library Author: Jordan Peck - https://github.com/Auburns
#       Python Extension Author:  Robert A. McLeod - robbmcleod@gmail.com
#
########################################################################

from pyfastnoisesimd.version import __version__

from pyfastnoisesimd.helpers import (
    Noise, emptyCoords,
    generate, setNumWorkers, 
    NoiseType, FractalType, PerturbType, 
    CellularReturnType, CellularDistanceFunction)

import pyfastnoisesimd.extension as _ext

from pyfastnoisesimd.test_fns import test

from pyfastnoisesimd.cpuinfo import get_cpu_info
cpu_info = get_cpu_info()