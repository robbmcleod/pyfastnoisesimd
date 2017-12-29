import pyfastnoisesimd.extension as ext
import concurrent.futures as cf
import numpy as np
from enum import Enum
from pyfastnoisesimd.cpuinfo import get_cpu_info
cpu_info = get_cpu_info()

class NoiseType(Enum):
    ''' The high-level class of noise generated.
    Valid enums: Value, ValueFractal, Perlin, PerlinFractal, Simplex, 
    SimplexFractal, WhiteNoise, Cellular, Cubic, CubicFractal
    '''
    Value          = 0
    ValueFractal   = 1
    Perlin         = 2
    PerlinFractal  = 3
    Simplex        = 4
    SimplexFractal = 5
    WhiteNoise     = 6
    Cellular       = 7
    Cubic          = 8
    CubicFractal   = 9

class FractalType(Enum):
    '''Fractal noise types also have an additional fractal type. 
    Valid enums: FBM, Billow, RigidMulti'''
    FBM            = 0
    Billow         = 1
    RigidMulti     = 2

class PerturbType(Enum):
    ''' The enumerator for the class of Perturbation.
    Valid enums: NoPeturb, Gradient, GradientFractal, Normalise, Gradient_Normalise, 
    GradientFractal_Normalise'''
    NoPertrub                 = 0
    Gradient                  = 1
    GradientFractal           = 2
    Normalise                 = 3
    Gradient_Normalise        = 4
    GradientFractal_Normalise = 5

class CellularDistanceFunction(Enum):
    '''The distance function for cellular noise.
    Valid enums: Euclidean, Manhattan, Natural'''
    Euclidean = 0
    Manhattan = 1
    Natural   = 2

class CellularReturnType(Enum):
    ''' The functional filter to apply to the distance function to generate the 
    return from cellular noise.
    Valid enums: CellValue, Distance, Distance2, Distance2Add, Distance2Sub, 
    Distance2Mul, Distance2Div, NoiseLookup, Distance2Cave'''
    CellValue     = 0
    Distance      = 1
    Distance2     = 2
    Distance2Add  = 3
    Distance2Sub  = 4
    Distance2Mul  = 5
    Distance2Div  = 6
    NoiseLookup   = 7
    Distance2Cave = 8

def emptyCoords(size):
    '''
    Generate an empty array of length `size` coordinates in [Z,Y,X], for use in 
    `Noise.genFromCoords()`.  

    The returned array will be of shape (3, size), however `size` may be somewhat
    larger to accommodate an integer number of SIMD instructions along its axis.
    '''
    size = ext.AlignedSize(int(size))
    empty = ext.EmptySet(size*3).reshape((3,size))
    return empty

class FractalClass(object):
    ''' Holds properties related to `NoiseType.<...>Fractal`. 

    Do not instantiate this class separately from `Noise`.'''
    def __init__(self, fns):
        self._fns = fns
        self._octaves = 3
        self._lacunarity = 2.0
        self._gain = 0.5
        self._fractalType = FractalType.FBM

    @property
    def fractalType(self):
        ''' The type of fractal for fractal NoiseTypes.
        Type: pyfastnoisesimd.FractalType, default: `FractalType.FBM`'''
        return self._fractalType

    @fractalType.setter
    def fractalType(self, new):
        if isinstance(new, FractalType):
            pass
        elif isinstance(new, int):
            new = FractalType(int)
        elif isinstance(new, str):
            new = FractalType[new]
        else:
            raise TypeError('Unparsable type for fractalType: {}'.format(type(new)))

        self._fractalType = new
        self._fns.SetFractalType(new.value)
        
    @property
    def octaves(self):
        '''Octave count for all fractal noise types.
	    type: `int`, default: 3'''
        return self._octaves

    @octaves.setter
    def octaves(self, new):
        self._octaves = int(new)
        self._fns.SetFractalOctaves(int(new))

    @property
    def lacunarity(self):
        '''Octave lacunarity for all fractal noise types
	    Type: `float`, default: 2.0'''
        return self._lacunarity

    @lacunarity.setter
    def lacunarity(self, new):
        self._lacunarity = float(new)
        self._fns.SetFractalLacunarity(float(new))

    @property
    def gain(self):
        '''Octave gain for all fractal noise types
	    type: `float`, default: 0.5'''
        return self._gain

    @gain.setter
    def gain(self, new):
        self._gain = float(new)
        self._fns.SetFractalGain(float(new))


class CellularClass(object):
    ''' Holds properties related to `NoiseType.Cellular`. 

    Do not instantiate this class separately from `Noise`.'''
    def __init__(self, fns):
        self._fns = fns

        self._returnType = CellularReturnType.Distance
        self._distanceFunc = CellularDistanceFunction.Euclidean
        self._noiseLookupType = NoiseType.Simplex
        self._lookupFrequency = 0.2
        self._jitter = 0.45
        self._distanceIndices = (0.0, 1.0)
    
    @property
    def returnType(self):
        ''' The return type for cellular (Voronoi) noise.
        Type: pyfastnoisesimd.CellularReturnType, default: `CellularReturnType.Distance`'''
        return self._returnType

    @returnType.setter
    def returnType(self, new):
        if isinstance(new, CellularReturnType):
            pass
        elif isinstance(new, int):
            new = CellularReturnType(int)
        elif isinstance(new, str):
            new = CellularReturnType[new]
        else:
            raise TypeError('Unparsable type for returnType: {}'.format(type(new)))

        self._returnType = new
        self._fns.SetCellularReturnType(new.value)

    @property
    def distanceFunc(self):
        return self._distanceFunc

    @distanceFunc.setter
    def distanceFunc(self, new):
        ''' The distance function for cellular (Voronoi) noise.
        Type: pyfastnoisesimd.CellularDistanceFunction, default: `CellularDistanceFunction.Euclidean`'''
        if isinstance(new, CellularDistanceFunction):
            pass
        elif isinstance(new, int):
            new = CellularDistanceFunction(int)
        elif isinstance(new, str):
            new = CellularDistanceFunction[new]
        else:
            raise TypeError('Unparsable type for distanceFunc: {}'.format(type(new)))

        self._distanceFunc = new
        self._fns.SetCellularDistanceFunction(new.value)

    @property
    def noiseLookupType(self):
        '''  Sets the type of noise used if cellular return type is set as `CellularReturnType.NoiseLookup`.
        Type: pyfastnoisesimd.NoiseType, default: `NoiseType.Simplex` '''
        return self._noiseLookupType

    @noiseLookupType.setter
    def noiseLookupType(self, new):
        if isinstance(new, NoiseType):
            pass
        elif isinstance(new, int):
            new = NoiseType(int)
        elif isinstance(new, str):
            new = NoiseType[new]
        else:
            raise TypeError('Unparsable type for noiseLookupType: {}'.format(type(new)))
        self._noiseLookupType = new
        self._fns.SetCellularNoiseLookupType(new.value)

    @property
    def lookupFrequency(self):
        ''' Relative frequency on the cellular noise lookup return type.
        Type: `float`, default: `0.2`'''
        return self._lookupFrequency

    @lookupFrequency.setter
    def lookupFrequency(self, new):
        self._lookupFrequency = float(new)
        self._fns.SetCellularNoiseLookupFrequency(float(new))

    @property
    def jitter(self):
        ''' The maximum distance a cellular point can move from it's grid 
        position. Setting this high will make artifacts more common.
	    Type: `float`, default: `0.45`'''
        return self._jitter

    @jitter.setter
    def jitter(self, new):
        self._jitter = float(new)
        self._fns.SetCellularJitter(float(new))

    @property
    def distanceIndices(self):
        '''Sets the 2 distance indices used for distance2 return types
	    Type: 2-element tuple/list of `int`, default: 0, 1
	    Note: index0 should be lower than index1
	    Both indices must be >= 0, index1 must be < 4'''
        return self._distanceIndices

    @distanceIndices.setter
    def distanceIndices(self, new):
        if not hasattr(new, '__len__') or len(new) != 2:
            raise ValueError( 'distanceIndices must be a length 2 array/list/tuple' )
        new = list(new)
        if new[0] < 0:
            new[0] = 0
        if new[1] < 0:
            new[0] = 0
        if new[1] >= 4:
            new[1] = 3
        if new[0] >= new[1]:
            new[0] = new[1]-1

        self._distanceIndices = new
        return self._fns.SetCellularDistance2Indices(*new)

class PerturbClass(object):
    ''' Holds properties related to the perturbation applied to noise.

    Do not instantiate this class separately from `Noise`.'''

    def __init__(self, fns):
        self._fns = fns

        self._perturbType = PerturbType.NoPertrub
        self._amp = 1.0
        self._frequency = 0.5
        self._octaves = 3
        self._lacunarity = 2.0
        self._gain = 2.0
        self._normaliseLength = 1.0

    @property
    def perturbType(self):
        ''' The class of perturbation. 
        Type: pyfastnoisesimd.PerturbType, default: `PerturbType.NoPeturb`'''
        return self._perturbType

    @perturbType.setter
    def perturbType(self, new):
        if isinstance(new, PerturbType):
            pass
        elif isinstance(new, int):
            new = PerturbType(int)
        elif isinstance(new, str):
            new = PerturbType[new]
        else:
            raise TypeError('Unparsable type for perturbType: {}'.format(type(new)))
        self._perturbType = new
        return self._fns.SetPerturbType(new.value)

    @property
    def amp(self):
        ''' The maximum distance the input position can be perturbed.
        Type: `float`, default: `1.0`'''
        return self._amp

    @amp.setter
    def amp(self, new):
        self._amp = float(new)
        return self._fns.SetPerturbAmp(float(new))

    @property
    def frequency(self):
        ''' The relative frequency for the perturbation gradient.
        Type: `float`, default: `0.5`'''
        return self._frequency

    @frequency.setter
    def frequency(self, new):
        self._frequency = float(new)
        return self._fns.SetPerturbFrequency(float(new))

    @property
    def octaves(self):
        ''' The octave count for fractal perturbation types,
        Type: `int`, default: `3` '''
        return self._octaves

    @octaves.setter
    def octaves(self, new):
        self._octaves = int(new)
        return self._fns.SetPerturbFractalOctaves(int(new))

    @property
    def lacunarity(self):
        '''The octave lacunarity (gap-fill) for fractal perturbation types.
        Type: `float`, default: `2.0`'''
        return self._lacunarity

    @lacunarity.setter
    def lacunarity(self, new):
        self._lacunarity = float(new)
        return self._fns.SetPerturbFractalLacunarity(float(new))

    @property
    def gain(self):
        '''The octave gain for fractal perturbation types.
        Type: `float`, default: `0.5`'''
        return self._gain

    @gain.setter
    def gain(self, new):
        self._gain = float(new)
        return self._fns.SetPerturbFractalGain(float(new))

    @property
    def normaliseLength(self):
        '''
        The length for vectors after perturb normalising 
        Type: `float`, default: `1.0`
        '''
        return self._normaliseLength

    @normaliseLength.setter
    def normaliseLength(self, new):
        self._normaliseLength = float(new)
        return self._fns.SetPerturbNormaliseLength(float(new))

def _chunk_noise_grid(fns, chunk, chunkStart, chunkAxis):
    '''
    For use by `concurrent.futures` to multi-thread `Noise.genAsGrid()` calls.
    '''
    dataPtr = chunk.__array_interface__['data'][0]
    # print( 'pointer: {}, start: {}, axis{}'.format(chunk, chunkStart, chunkAxis) )
    if chunkAxis == 0:
        fns.FillNoiseSet(dataPtr, chunkStart, 0, 0, *chunk.shape)
    elif chunkAxis == 1:
        fns.FillNoiseSet(dataPtr, 0, chunkStart, 0, *chunk.shape)
    else:
        fns.FillNoiseSet(dataPtr, 0, 0, chunkStart, *chunk.shape)

class Noise(object):

    def __init__(self, seed=None, numWorkers=None):
        '''
        If `seed == None` a random integer is generated as the seed.

        If `numWorkers == None` the number of virtual cores found by `cpuinfo.py` 
        is used.
        '''
        self._fns = ext.FNS()
        if bool(numWorkers):
            self._asyncExecutor = cf.ThreadPoolExecutor(max_workers = numWorkers)
        else:
            self._asyncExecutor = cf.ThreadPoolExecutor(max_workers = cpu_info['count'])

        # Sub-classed object handles
        self.fractal = FractalClass(self._fns)
        self.cell = CellularClass(self._fns)
        self.perturb = PerturbClass(self._fns)

        if bool(seed):
            self.seed = seed # calls setter
        else:
            self.seed = np.random.randint(-2147483648, 2147483647)

        # Syncronizers for property getters should use the default values as
        # stated in `FastNoiseSIMD.h`
        self._noiseType = NoiseType.Simplex
        self._frequency = 0.01
        self._axesScales = (1.0, 1.0, 1.0)
        
    @property
    def numWorkers(self):
        return self._asyncExecutor._max_workers

    @numWorkers.setter
    def numWorkers(self, N_workers):
        ''' Sets the maximum number of thread workers that will be used for 
        generating noise. Generally should be the number of physical CPU cores 
        on the machine. '''
        N_workers = int(N_workers)
        if N_workers <= 0:
            raise ValueError('numWorkers must be greater than 0')
        # if self._asyncExecutor._max_workers == N_workers:
        #     return

        self._asyncExecutor = cf.ThreadPoolExecutor(max_workers = N_workers)

    @property
    def SIMDLevel(self):
        ''' A text string identifying the SIMD level `FastNoiseSIMD` was 
        compiled with. '''
        levels = {
            5: 'ARM NEON',
            4: 'AVX512',
            3: 'AVX2 & FMA3',
            2: 'SSE4.1',
            1: 'SSE2',
            0: 'Fallback, no SIMD support'
        }
        return levels[self._fns.GetSIMDLevel()]

    @property 
    def seed(self):
        '''
        The random-number seed used for generation of noise. 
        Type: `np.int32`, default: `numpy.random.randint()`
        '''
        return self._fns.GetSeed()

    @seed.setter
    def seed(self, new):
        return self._fns.SetSeed(int(np.int32(new)))

    @property
    def frequency(self):
        ''' The frequency of the noise, lower values result in larger noise features.
        Type: `float`, default: `0.01`'''
        return self._frequency

    @frequency.setter
    def frequency(self, new):
        self._frequency = float(new)
        return self._fns.SetFrequency(float(new))

    @property
    def noiseType(self):
        ''' The class of noise. 
        Type: pyfastnoisesimd.NoiseType, default: `NoiseType.Simplex` '''
        return self._noiseType

    @noiseType.setter
    def noiseType(self, new):
        if isinstance(new, NoiseType):
            pass
        elif isinstance(new, int):
            new = NoiseType(int)
        elif isinstance(new, str):
            new = NoiseType[new]
        else:
            raise TypeError('Unparsable type for noiseType: {}'.format(type(new)))
        self._noiseType = new
        return self._fns.SetNoiseType(new.value)

    @property
    def axesScales(self):
        '''Sets the FastNoiseSIMD axes scales, which allows for non-square 
        voxels. Indirectly affects `frequency` by changing the voxel pitch.
        Type: 3-element tuple of floats, e.g. (1.0, 1.0, 10.0).  '''
        return self._axesScales

    @axesScales.setter
    def axesScales(self, new):
        if not hasattr(new, '__len__') or len(new) != 3:
            raise ValueError( 'axesScales must be a length 3 array/list/tuple' )
        
        self._axesScales = new
        return self._fns.SetAxesScales(*new)

    def genAsGrid(self, shape=[1,1024,1024], start=[0,0,0]):
        '''
        genAsGrid(self, shape=[1,1024,1024], start=[0,0,0])

        Generates noise according to the set properties along a rectilinear 
        (evenly-spaced) grid.  

        * `shape`:  the shape of the output noise volume.
        * `start`: the starting coordinates for generation of the grid.
          I.e. the coordinates are essentially `start: start + shape`

        Usage example::

            import numpy as np
            import pyfastnoisesimd as fns 
            noise = fns.Noise()
            result = noise.genFromGrid(shape=[256,256,256], start=[0,0,0])
            nextResult = noise.genFromGrid(shape=[256,256,256], start=[256,0,0])
        '''
        if self._asyncExecutor._max_workers <= 1:
            return self._fns.GetNoiseSet( *start, *shape )

        # else run in threaded mode.
        # Create a full shape empty array
        noise = ext.EmptySet( *shape )
        # It would be nice to be able to split both on Z and Y if needed...
        if shape[0] > 1:
            chunkAxis = 0
        elif shape[1] > 1:
            chunkAxis = 1
        else:
            chunkAxis = 2

        numChunks = np.minimum( self._asyncExecutor._max_workers, shape[chunkAxis] ) 

        chunkedNoise = np.array_split( noise, numChunks, axis=chunkAxis )
        chunkIndices = [ start[0] for start in np.array_split( np.arange(shape[chunkAxis]), numChunks )]

        workers = []
        for I, (chunk, chunkIndex) in enumerate( zip(chunkedNoise,chunkIndices) ):
            if chunk.shape == 0: # Empty array indicates we have more threads than chunks
                continue
            workers.append( self._asyncExecutor.submit( _chunk_noise_grid, self._fns, chunk, chunkIndex, chunkAxis ) )

        for peon in workers:
            peon.result()
        return noise

    def genFromCoords(self, coords):
        '''
        genFromCoords(self, coords)

        Generate noise from supplied coordinates, rather than a rectilinear grid.
        Useful for complicated shapes, such as tesselated surfaces.

        Usage example::

            import numpy as np
            import pyfastnoisesimd as fns 
            numCoords = 256
            coords = fns.emptyCoords(numCoords)
            # <Set the coordinate values, it is a (3, numCoords) array
            coords[0,:] = np.linspace(-np.pi, np.pi, numCoords)
            coords[1,:] = np.linspace(-1.0, 1.0, numCoords)
            coords[2,:] = np.zeros(numCoords)
            noise = fns.Noise()
            result = noise.genFromCoords(coords)

        '''

        # Check that coords is C-aligned, contiguous, shape (3,N), N is 
        # evenly divisible by the simdSize
        if not isinstance(coords, np.ndarray):
            raise TypeError('coords must be a `numpy.ndarray`')
        if coords.ndim != 2:
            raise ValueError('coords must be a 2D array')
        shape = coords.shape
        if shape[0] != 3:
            raise ValueError('coords.shape[0] must equal 3')
        simdLen = ext.AlignedSize(1)
        size = shape[1]
        if size%simdLen != 0:
            raise ValueError('coord.shape[1] must be evenly divisible by the SIMD instruction length: {}'.format(simdLen))
        
        # coords = np.require(coords, dtype='float32', requirements=['C', 'A'])
        noise = ext.EmptySet( size )
        noisePtr = noise.__array_interface__['data'][0]
        zPtr = coords.__array_interface__['data'][0]
        yPtr = zPtr + 4*size
        xPtr = yPtr + 4*size

        self._fns.NoiseFromCoords(noisePtr, zPtr, yPtr, xPtr, size, 0)
        return noise
        '''
        # TODO: fix seg-faults in multi-threaded operation.
        if self._asyncExecutor._max_workers <= 1:
            self._fns.NoiseFromCoords(noisePtr, zPtr, yPtr, xPtr, size, 0)
            return noise

        simdBlocks = size // simdLen
        numWorkers = np.minimum( self._asyncExecutor._max_workers, simdBlocks)
        print( 'Parallel genFromCoords using {} workers'.format(numWorkers) )

        chunkLen = (simdBlocks // numWorkers) * simdLen
        chunkBytes = 4 * chunkLen

        print( 'simdBlocks = {}, chunkLen = {}, chunkBytes = {}'.format(simdBlocks, chunkLen, chunkBytes) )
        print( 'Noise extent: {:X} to {:X}'.format(noise.__array_interface__['data'][0], noise.__array_interface__['data'][0]+noise.nbytes) )
        print( 'Coords extent: {:X} to {:X}'.format(coords.__array_interface__['data'][0], coords.__array_interface__['data'][0]+coords.nbytes)) 
        print( 'Coords Z: {:X}'.format(coords[0,:].__array_interface__['data'][0]) )
        print( 'Coords Y: {:X}'.format(coords[1,:].__array_interface__['data'][0]) )
        print( 'Coords X: {:X}'.format(coords[2,:].__array_interface__['data'][0]) )
        workers = []
        for I in range(numWorkers-1):
            print( '{}: noisePtr = {:X}'.format(I, noisePtr + I*chunkBytes) )
            print( '{}: zPtr = {:X}'.format(I, zPtr + I*chunkBytes) )
            print( '{}: yPtr = {:X}'.format(I, yPtr + I*chunkBytes) )
            print( '{}: xPtr = {:X}'.format(I, xPtr + I*chunkBytes) )
            
            workers.append( 
                self._asyncExecutor.submit(
                    self._fns.NoiseFromCoords, 
                    noisePtr, zPtr, yPtr, xPtr, chunkLen, I*chunkLen ))
      
        # Last worker takes any odd simdBlocks to the end of the array
        lastChunkLen = size - (numWorkers-1)*chunkLen
        I += 1
        print( 'lastChunkLen = {}'.format(lastChunkLen))
        print( 'last{}: noisePtr = {:X}'.format(I, noisePtr + I*chunkBytes) )
        print( 'last{}: zPtr = {:X}'.format(I, zPtr + I*chunkBytes) )
        print( 'last{}: yPtr = {:X}'.format(I, yPtr + I*chunkBytes) )
        print( 'last{}: xPtr = {:X}'.format(I, xPtr + I*chunkBytes) )
        
        workers.append( 
            self._asyncExecutor.submit(
                self._fns.NoiseFromCoords,
                noisePtr, zPtr, yPtr, xPtr, lastChunkLen, I*chunkLen ))

        for peon in workers:
            peon.result()
        print( "Peons done")
        return noise
        '''


    





#######################################################
######### DEPRECATED `kitchen-sink` interface #########
#######################################################

_factory = ext.FNS()
_factoryExecutor = cf.ThreadPoolExecutor( max_workers = 1 )
def setNumWorkers( N_workers ):
    '''
    setNumWorkers( N_workers )

    ===DEPRECATED===

    Sets the maximum number of thread workers that will be used for generating
    noise.
    '''
    N_workers = int( N_workers )
    if N_workers <= 0:
        raise ValueError('N_workers must be greater than 0')
    if _factoryExecutor._max_workers == N_workers:
        return

    _factoryExecutor._max_workers = N_workers
    _factoryExecutor._adjust_thread_count()

def generate( size=[1,1024,1024], start=[0,0,0],
              seed=42, freq=0.01, noiseType='Simplex', axesScales=[1.0,1.0,1.0], 
              fracType='FBM', fracOctaves=4, 
              fracLacunarity=2.0, fracGain=0.5, 
              cellReturnType='Distance', cellDistFunc='Euclidean',
              cellNoiseLookup='Simplex', cellNoiseLookupFreq=0.2, 
              cellDist2Ind=[0,1], cellJitter=0.2,
              perturbType=None, perturbAmp=1.0, perturbFreq=0.5, perturbOctaves=3,
              perturbLacunarity=2.0, perturbGain=0.5, perturbNormLen=1.0,   ):
    '''
    def generate( size=[1,1024,1024], start=[0,0,0], 
              seed=42, freq=0.01, noiseType='Simplex', axesScales=[1.0,1.0,1.0], 
              fracType='FBM', fracOctaves=4, 
              fracLacunarity=2.0, fractalGain=0.5, 
              cellReturnType='Distance', cellDistFunc='Euclidean',
              cellNoiseLookup='Simplex', cellNoiseLookupFreq='0.2', 
              cellDist2Ind=[0,1], cellJitter=0.2,
              perturbType=None, perturbAmp=1.0, perturbFreq=0.5, perturbOctaves=3,
              perturbLacunarity=2.0, perturbGain=0.5, perturbNormLen=1.0,   )

    ===DEPRECATED===
    '''
    print( 'generate() is deprecated, please use Noise() class.' )
    _factory.SetSeed( seed )
    _factory.SetFrequency( freq )
    _factory.SetNoiseType( ext.noiseType[noiseType] )
    _factory.SetAxesScales( axesScales[0], axesScales[1], axesScales[2] )
    _factory.SetFractalOctaves( fracOctaves )
    _factory.SetFractalLacunarity( fracLacunarity )
    _factory.SetFractalGain( fracGain )
    _factory.SetFractalType( ext.fractalType[fracType] )
    _factory.SetCellularReturnType( ext.cellularReturnType[cellReturnType] )
    _factory.SetCellularDistanceFunction( ext.cellularDistanceFunction[cellDistFunc]  )
    _factory.SetCellularNoiseLookupType( ext.noiseType[cellNoiseLookup] )
    _factory.SetCellularNoiseLookupFrequency( cellNoiseLookupFreq )
    _factory.SetCellularDistance2Indices( *cellDist2Ind  )
    _factory.SetCellularJitter( cellJitter )
    _factory.SetPerturbType( ext.perturbType[perturbType] )
    _factory.SetPerturbAmp( perturbAmp )
    _factory.SetPerturbFrequency( perturbFreq )
    _factory.SetPerturbFractalOctaves( perturbOctaves )
    _factory.SetPerturbFractalLacunarity( perturbLacunarity )
    _factory.SetPerturbFractalGain( perturbGain )
    _factory.SetPerturbNormaliseLength( perturbNormLen )

    if _factoryExecutor._max_workers <= 1:
        return _factory.GetNoiseSet( *start, *size )

    # else run in threaded mode.
    # Create a full size empty array
    noise = ext.EmptySet( *size )
    # It would be nice to be able to split both on Z and Y if needed...
    if size[0] > 1:
        chunkAxis = 0
    elif size[1] > 1:
        chunkAxis = 1
    else:
        chunkAxis = 2

    numChunks = np.minimum( _factoryExecutor._max_workers, size[chunkAxis] )

    chunkedNoise = np.array_split( noise, numChunks, axis=chunkAxis )
    chunkIndices = [ start[0] for start in np.array_split( np.arange(size[chunkAxis]), numChunks )]

    workers = []
    for I, (chunk, chunkIndex) in enumerate( zip(chunkedNoise,chunkIndices) ):
        if chunk.size == 0: # Empty array indicates we have more threads than chunks
            continue
        
        workers.append( _factoryExecutor.submit( _chunked_gen, chunk, chunkIndex, chunkAxis ) )

    for peon in workers:
        peon.result()
    return noise

def _chunked_gen( chunk, chunkStart, chunkAxis ):
    pointer = chunk.__array_interface__['data'][0]
    # print( 'pointer: {}, start: {}, axis{}'.format(chunk, chunkStart, chunkAxis) )
    if chunkAxis == 0:
        _factory.FillNoiseSet( pointer, chunkStart, 0, 0, *chunk.shape )
    elif chunkAxis == 1:
        _factory.FillNoiseSet( pointer, 0, chunkStart, 0, *chunk.shape )
    else:
        _factory.FillNoiseSet( pointer, 0, 0, chunkStart, *chunk.shape )