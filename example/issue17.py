import numpy as np
import numpy.testing as npt
import pyfastnoisesimd as fns

noise = fns.Noise()
noise.numWorkers = 4
grid_origin = noise.genAsGrid(shape=[64,64,64], start=[0,0,0])
grid_offset = noise.genAsGrid(shape=[64,64,64], start=[50,50,50])
try:
    npt.assert_array_almost_equal(grid_origin, grid_offset)
    raise ValueError('Arrays are equal, that means `start` parameter is not working')
except AssertionError:
    print('Issue 17 is ok')
