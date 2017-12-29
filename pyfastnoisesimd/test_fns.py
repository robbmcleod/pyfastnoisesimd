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

X = 32
        
class FNS_Tests(unittest.TestCase):
    
    def setUp(self):
        pass

    def grid(self, numWorkers):
        noise = fns.Noise(seed=None, numWorkers=numWorkers)
        
        noise.frequency = 0.15
        assert(noise.frequency == 0.15)
        noise.axesScales = [0.9, 0.85, 0.9]
        assert(noise.axesScales == [0.9, 0.85, 0.9])

        # Iterate through NoiseType
        for newNoise in fns.NoiseType:
            noise.noiseType = newNoise
            assert(noise.noiseType == newNoise)
            result = noise.genAsGrid(shape=[X,X,X])
            assert(result.shape == (X,X,X))
        
        # Test different dimensionalities
        result = noise.genAsGrid(shape=[1,X,X])
        assert(result.shape == (1,X,X) )
        result = noise.genAsGrid(shape=[1,1,X])
        assert(result.shape == (1,1,X))

        # Iterate through FractalType
        noise.fractal.octaves = 2
        assert(noise.fractal.octaves == 2)
        noise.fractal.lacunarity = 1.9
        assert(noise.fractal.lacunarity == 1.9)
        noise.fractal.gain = 0.7
        assert(noise.fractal.gain == 0.7)
        for newFractal in fns.FractalType:
            noise.fractal.fractalType = newFractal
            assert(noise.fractal.fractalType == newFractal)
            noise.genAsGrid(shape=[1,1,X])

        # Iterate through PeturbType
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
            noise.genAsGrid(shape=[1,1,X])

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
            noise.genAsGrid(shape=[1,1,X])
        for retType in fns.CellularReturnType:
            noise.cell.returnType = retType
            assert(noise.cell.returnType == retType)
            noise.genAsGrid(shape=[1,1,X])
        for distFunc in fns.CellularDistanceFunction:
            noise.cell.distanceFunc = distFunc
            assert(noise.cell.distanceFunc == distFunc)
            noise.genAsGrid(shape=[1,1,X])
        
        return

    def coords(self, numWorkers):
        noise = fns.Noise(seed=None, numWorkers=numWorkers)
        coords = fns.emptyCoords(X)
        coords[0,:] = np.arange(X)
        coords[1,:] = np.arange(-16,16)
        coords[2,:] = np.linspace(-1.0, 1.0, X)
        noise.genFromCoords(coords)
        return

    def test_grid_1(self):
        self.grid(1)

    def test_grid_4(self):
        self.grid(4)

    def test_coords_1(self):
        self.coords(1)
    

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
  
    

    



                     
