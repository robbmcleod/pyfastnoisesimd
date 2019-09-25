# -*- coding: utf-8 -*-
"""
PyFastNoiseSIMD unit testing
@author: Robert A. McLeod
"""

import pyfastnoisesimd as fns
import numpy as np
import numpy.testing as npt
import os, os.path, sys
import tempfile
import unittest
from logging import Logger
log = Logger(__name__)

# X is block size.
# For AVX512, if we want to have 4 workers we should have at least 16 x 4 
# blocks.
SIMD_LEN= fns.extension.SIMD_ALIGNMENT // np.dtype(np.float32).itemsize
CHUNK = fns.helpers._MIN_CHUNK_SIZE
CHUNK2 = int(np.ceil(np.sqrt(CHUNK) / SIMD_LEN) * SIMD_LEN)
CHUNK3 = int(np.ceil(np.cbrt(CHUNK) / SIMD_LEN) * SIMD_LEN)

class FNS_Tests(unittest.TestCase):
    
    def setUp(self):
        pass

    def grid_3d(self, size=CHUNK3, numWorkers=1):
        noise = fns.Noise(seed=None, numWorkers=numWorkers)
        noise.frequency = 0.15
        assert(noise.frequency == 0.15)
        noise.axesScales = [0.9, 0.85, 0.9]
        assert(noise.axesScales == [0.9, 0.85, 0.9])

        result = noise.genAsGrid(shape=[size,size,size])
        assert(result.shape == (size,size,size))


    def grid_2d(self, size=CHUNK2, numWorkers=1):
        noise = fns.Noise(seed=None, numWorkers=numWorkers)
        # Test 2D
        result = noise.genAsGrid(shape=[size,size])
        assert(result.shape == (size,size))


    def grid_1d(self, size=CHUNK, numWorkers=1):
        # Test 1D
        noise = fns.Noise(seed=None, numWorkers=numWorkers)
        result = noise.genAsGrid(shape=size)
        assert(result.shape == (size,))

    def coords(self, size=CHUNK, numWorkers=1):
        noise = fns.Noise(seed=None, numWorkers=numWorkers)
        coords = fns.empty_coords(size)
        coords[0,:] = np.arange(size)
        coords[1,:] = np.arange(-size//2, size//2)
        coords[2,:] = np.linspace(-1.0, 1.0, size)
        noise.genFromCoords(coords)

        # Re-use coords to make sure the array isn't accidentally free'd 
        # in FastNoiseSIMD
        noise2 = fns.Noise(seed=None, numWorkers=numWorkers)
        noise2.genFromCoords(coords)
        return

    def test_grid_1thread(self):
        self.grid_3d(size=CHUNK3, numWorkers=1)
        self.grid_2d(size=CHUNK2, numWorkers=1)
        self.grid_1d(size=CHUNK,  numWorkers=1)

    def test_grid_4thread(self):
        self.grid_3d(size=CHUNK3, numWorkers=4)
        self.grid_2d(size=CHUNK2, numWorkers=4)
        self.grid_1d(size=CHUNK,  numWorkers=4)

    def test_coords_1(self):
        self.coords(size=CHUNK, numWorkers=1)

    def test_coords_4(self):
        self.coords(size=CHUNK, numWorkers=1)

    def test_noise_type(self, size=CHUNK, numWorkers=1):
        noise = fns.Noise(seed=None, numWorkers=numWorkers)
        # Iterate through NoiseType
        for newNoise in fns.NoiseType:
            noise.noiseType = newNoise
            assert(noise.noiseType == newNoise)
            result = noise.genAsGrid(shape=size)

    def fractal(self, size=CHUNK, numWorkers=1):
        # Iterate through FractalType
        noise = fns.Noise(seed=None, numWorkers=numWorkers)
        noise.fractal.octaves = 2
        assert(noise.fractal.octaves == 2)
        noise.fractal.lacunarity = 1.9
        assert(noise.fractal.lacunarity == 1.9)
        noise.fractal.gain = 0.7
        assert(noise.fractal.gain == 0.7)
        for newFractal in fns.FractalType:
            noise.fractal.fractalType = newFractal
            assert(noise.fractal.fractalType == newFractal)
            noise.genAsGrid(shape=size)

    def test_perturb(self, size=CHUNK, numWorkers=1):
        # Iterate through PeturbType
        noise = fns.Noise(seed=None, numWorkers=numWorkers)
        noise.perturb.amp = 1.1
        assert(noise.perturb.amp == 1.1)
        noise.perturb.frequency = 0.44
        assert(noise.perturb.frequency == 0.44)
        noise.perturb.octaves = 2
        assert(noise.perturb.octaves == 2)
        noise.perturb.lacunarity = 2.2
        assert(noise.perturb.lacunarity == 2.2)
        noise.perturb.gain = 0.6
        assert(noise.perturb.gain == 0.6)
        for newPeturb in fns.PerturbType:
            noise.perturb.perturbType = newPeturb
            assert(noise.perturb.perturbType == newPeturb)
            noise.genAsGrid(shape=size)

    def test_cell(self, size=CHUNK, numWorkers=1):
        noise = fns.Noise(seed=None, numWorkers=numWorkers)
        # Iterate through cellular options
        noise.cell.jitter = 0.7
        assert(noise.cell.jitter == 0.7)
        noise.cell.lookupFrequency = 0.3
        assert(noise.cell.lookupFrequency == 0.3)
        noise.cell.distanceIndices = [0, 2]
        assert(noise.cell.distanceIndices == [0, 2])

        noise.cell.returnType = fns.CellularReturnType.NoiseLookup
        for newNoise in fns.NoiseType:
            noise.cell.noiseLookupType = newNoise
            assert(noise.cell.noiseLookupType == newNoise)
            noise.genAsGrid(shape=size)
        for retType in fns.CellularReturnType:
            noise.cell.returnType = retType
            assert(noise.cell.returnType == retType)
            noise.genAsGrid(shape=size)
        for distFunc in fns.CellularDistanceFunction:
            noise.cell.distanceFunc = distFunc
            assert(noise.cell.distanceFunc == distFunc)
            noise.genAsGrid(shape=size)
    

def test(verbosity=2):
    '''
    test(verbosity=2)

    Run ``unittest`` suite for ``pyfastnoisesimd`` package.
    '''

    log.info( "pyfastnoisesimd tests for version: {}".format(fns.__version__))
    
    theSuite = unittest.TestSuite()

    theSuite.addTest(unittest.makeSuite(FNS_Tests))

    return unittest.TextTestRunner(verbosity=verbosity).run(theSuite)
    
if __name__ == "__main__":
    # For continuous integraton, call as:
    # python -c \"import sys;import pyfastnoisesimd as fns;sys.exit(0 if fns.test().wasSuccessful() else 1)\""
    test_result = test()
    print( "Tests result: {}".format(test_result))
  
    

    



                     
