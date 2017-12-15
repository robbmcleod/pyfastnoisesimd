# -*- coding: utf-8 -*-
########################################################################
#
#       PyFastNoiseSIMD
#       License: BSD
#       Created: August 13, 2017
#       Library Author: Jordan Peck - https://github.com/Auburns
#       Python Extension Author:  Robert A. McLeod - robbmcleod@gmail.com
#
########################################################################

from pyfastnoisesimd.version import __version__
from pyfastnoisesimd import extension as _ext

from pyfastnoisesimd.helpers import generate, setNumWorkers

from pyfastnoisesimd.cpuinfo import get_cpu_info
cpu_info = get_cpu_info()
setNumWorkers( cpu_info['count'] )
