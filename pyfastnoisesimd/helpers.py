import pyfastnoisesimd.extension as ext
import concurrent.futures as cf
import numpy as np
from enum import Enum

_MIN_CHUNK_SIZE = 8192

def empty_aligned(shape, dtype=np.float32, n_byte=ext.SIMD_ALIGNMENT):
    """
    Provides an memory-aligned array for use with SIMD accelerated instructions.
    Should be used to build 

    Adapted from: https://github.com/hgomersall/pyFFTW/blob/master/pyfftw/utils.pxi

    Args:
        shape: a sequence (typically a tuple) of array axes.
        dtype: NumPy data type of the underlying array. Note FastNoiseSIMD supports 
               only `np.float32`. Seg faults may occur if this is changed.
        n_byte: byte alignment. Should always use the `pyfastnoisesimd.extension.SIMD_ALIGNMENT`
                value or seg faults may occur.
    """
    dtype = np.dtype(dtype)
    itemsize = dtype.itemsize

    if not isinstance(shape, (int, np.integer)):
        array_length = 1
        for each_dimension in shape:
            array_length *= each_dimension
    else:
        array_length = shape

    # Allocate a new array that will contain the aligned data
    buffer = np.empty(array_length * itemsize + n_byte, dtype='int8')

    offset = (n_byte - buffer.ctypes.data) % n_byte
    aligned = buffer[offset:offset-n_byte].view(dtype).reshape(shape)
    return aligned

def full_aligned(shape, fill, dtype=np.float32, n_byte=ext.SIMD_ALIGNMENT):
    """
    As per `empty_aligned`, but returns an array initialized to a constant value.

    Args:
        shape: a sequence (typically a tuple) of array axes.
        fill: the value to fill each array element with.
        dtype: NumPy data type of the underlying array. Note FastNoiseSIMD supports 
               only `np.float32`. Seg faults may occur if this is changed.
        n_byte: byte alignment. Should always use the `pyfastnoisesimd.extension.SIMD_ALIGNMENT`
                value or seg faults may occur.
    """
    aligned = empty_aligned(shape, dtype=dtype, n_byte=n_byte)
    aligned.fill(fill)
    return aligned

def empty_coords(length, dtype=np.float32, n_byte=ext.SIMD_ALIGNMENT):
    """
    """
    dtype = np.dtype(dtype)
    itemsize = dtype.itemsize

    # We need to expand length to be a multiple of the vector size
    vect_len = max(ext.SIMD_ALIGNMENT // itemsize, 1)
    aligned_len = int(vect_len*np.ceil(length/vect_len))
    shape = (3, aligned_len)

    coords = empty_aligned(shape)

    # Lots of trouble with passing sliced views out, with over-running the 
    # array and seg-faulting on an invalid write.
    # return coords[:,:length]
    return coords

def check_alignment(array):
    """
    Verifies that an array is aligned correctly for the supported SIMD level.

    Args:
        array: numpy.ndarray

    Returns:
        truth: bool
    """
    return ((ext.SIMD_ALIGNMENT - array.ctypes.data) % ext.SIMD_ALIGNMENT) == 0

def check_coords(array):
    """
    Verifies that an array is aligned correctly for the supported SIMD level.

    Args:
        array: numpy.ndarray

    Returns:
        truth: bool
    """
    # print('  Array shape: ', array.shape)
    # if array.base is not None:
    #     print('  Base shape: ', array.base.shape)

    base = array if array.base is None else array.base
    # print('  Check alignment: ', check_alignment(array))
    # print('  Base alignment error: ', (array.shape[1]*array.dtype.itemsize)%ext.SIMD_ALIGNMENT)
    return check_alignment(array) \
            and (array.shape[1] * array.dtype.itemsize) % ext.SIMD_ALIGNMENT == 0 == 0

def aligned_chunks(array, n_chunks, axis=0):
    """
    An generator that divides an array into chunks that have memory
    addresses compatible with SIMD vector length.

    Args:
        array: numpy.ndarray
            the array to chunk
        n_chunks: int
            the desired number of chunks, the returned number _may_ be less.
        axis: int
            the axis to chunk on, similar to `numpy` axis commanes.

    Returns
        chunk: numpy.ndarray
        start: Tuple[int]
            the start indices of the chunk, in the `array`.
    """
    block_size = 1
    if array.ndim > axis + 1:
        block_size = np.product(array.shape[axis:])
    # print(f'Got blocksize of {block_size}')

    vect_len = max(ext.SIMD_ALIGNMENT // array.dtype.itemsize, 1)

    if block_size % vect_len == 0:
        # Iterate at-will, the underlying blocks have the correct shape
        slice_size =  int(np.ceil(array.shape[axis] / n_chunks))
    else:
        # Round slice_size up to nearest vect_len
        slice_size = int(vect_len*np.ceil(array.shape[axis] / n_chunks /vect_len))
    # print('Slice size: ', slice_size)

    offset = 0
    chunk_index = 0
    while(chunk_index < n_chunks):
        # Dynamic slicing is pretty clumsy, unfortunately:
        # https://stackoverflow.com/questions/24398708/slicing-a-numpy-array-along-a-dynamically-specified-axis#47859801
        # so just use some nested conditions.
        if axis == 0:
            if array.ndim == 1:
                chunk = array[offset:offset+slice_size]
            else:
                chunk = array[offset:offset+slice_size, ...]
        elif axis == 1:
            if array.ndim == 2:
                chunk = array[:, offset:offset+slice_size]
            else:
                chunk = array[:, offset:offset+slice_size, ...]
        elif axis == 2:
            chunk = array[:, :, offset:offset+slice_size]

        # print(f'Chunk has memory addr: {chunk.ctypes.data}, remain: {chunk.ctypes.data%ext.SIMD_ALIGNMENT}')
        yield chunk, offset
        offset += slice_size
        chunk_index += 1


def num_virtual_cores():
    """
    Detects the number of virtual cores on a system without importing 
    ``multiprocessing``. Borrowed from NumExpr 2.6.
    """
    import os, subprocess
    # Linux, Unix and MacOS
    if hasattr(os, 'sysconf'):
        if 'SC_NPROCESSORS_ONLN' in os.sysconf_names:
            # Linux & Unix:
            ncpus = os.sysconf('SC_NPROCESSORS_ONLN')
            if isinstance(ncpus, int) and ncpus > 0:
                return ncpus
        else:  # OSX:
            return int(subprocess.check_output(['sysctl', '-n', 'hw.ncpu']))
    # Windows
    if 'NUMBER_OF_PROCESSORS' in os.environ:
        ncpus = int(os.environ['NUMBER_OF_PROCESSORS'])
        if ncpus > 0:
            return ncpus
        else:
            return 1
    # TODO: need method for ARM7/8
    return 1  # Default


class NoiseType(Enum):
    """
    The class of noise generated.

    Enums: ``{Value, ValueFractal, Perlin, PerlinFractal, Simplex, SimplexFractal, WhiteNoise, Cellular, Cubic, CubicFractal}``
    """
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
    """
    Enum: Fractal noise types also have an additional fractal type. 

    Values: ``{FBM, Billow, RigidMulti}``"""
    FBM            = 0
    Billow         = 1
    RigidMulti     = 2

class PerturbType(Enum):
    """
    Enum: The enumerator for the class of Perturbation.

    Values: ``{NoPeturb, Gradient, GradientFractal, Normalise, Gradient_Normalise, GradientFractal_Normalise}``
    """
    NoPerturb                 = 0
    Gradient                  = 1
    GradientFractal           = 2
    Normalise                 = 3
    Gradient_Normalise        = 4
    GradientFractal_Normalise = 5

class CellularDistanceFunction(Enum):
    """
    Enum: The distance function for cellular noise.

    Values: ``{Euclidean, Manhattan, Natural}``"""
    Euclidean = 0
    Manhattan = 1
    Natural   = 2

class CellularReturnType(Enum):
    """
    Enum: The functional filter to apply to the distance function to generate the 
    return from cellular noise.

    Values: ``{CellValue, Distance, Distance2, Distance2Add, Distance2Sub, Distance2Mul, Distance2Div, NoiseLookup, Distance2Cave}``
    """
    CellValue     = 0
    Distance      = 1
    Distance2     = 2
    Distance2Add  = 3
    Distance2Sub  = 4
    Distance2Mul  = 5
    Distance2Div  = 6
    NoiseLookup   = 7
    Distance2Cave = 8


class FractalClass(object):
    """ 
    Holds properties related to noise types that include fractal octaves.

    Do not instantiate this class separately from `Noise`.
    """
    def __init__(self, fns):
        self._fns = fns
        self._octaves = 3
        self._lacunarity = 2.0
        self._gain = 0.5
        self._fractalType = FractalType.FBM

    @property
    def fractalType(self) -> FractalType:
        """
        The type of fractal for fractal NoiseTypes.

        Default: ``FractalType.FBM``"""
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
    def octaves(self) -> int:
        """
        Octave count for all fractal noise types, i.e. the number of 
        log-scaled frequency levels of noise to apply. Generally ``3`` is 
        sufficient for small textures/sprites (256x256 pixels), use larger 
        values for larger textures/sprites.

	    Default: ``3``
        """
        return self._octaves

    @octaves.setter
    def octaves(self, new):
        self._octaves = int(new)
        self._fns.SetFractalOctaves(int(new))

    @property
    def lacunarity(self) -> float:
        """
        Octave lacunarity for all fractal noise types.

	    Default: ``2.0``
        """
        return self._lacunarity

    @lacunarity.setter
    def lacunarity(self, new):
        self._lacunarity = float(new)
        self._fns.SetFractalLacunarity(float(new))

    @property
    def gain(self) -> float:
        """
        Octave gain for all fractal noise types. Reflects the ratio 
        of the underlying noise to that of the fractal.  Values > 0.5 up-weight 
        the fractal.

	    Default: ``0.5``
        """
        return self._gain

    @gain.setter
    def gain(self, new):
        self._gain = float(new)
        self._fns.SetFractalGain(float(new))


class CellularClass(object):
    """
    Holds properties related to `NoiseType.Cellular`. 

    Do not instantiate this class separately from ``Noise``.
    """
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
        """
        The return type for cellular (cubic Voronoi) noise.

        Default: ``CellularReturnType.Distance``
        """
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
        """
        The distance function for cellular (cubic Voronoi) noise.

        Default: ``CellularDistanceFunction.Euclidean``
        """
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
    def noiseLookupType(self) -> NoiseType:
        """
        Sets the type of noise used if cellular return type.
        
        Default: `NoiseType.Simplex`
        """
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
        """
        Relative frequency on the cellular noise lookup return type.

        Default: ``0.2``
        """
        return self._lookupFrequency

    @lookupFrequency.setter
    def lookupFrequency(self, new):
        self._lookupFrequency = float(new)
        self._fns.SetCellularNoiseLookupFrequency(float(new))

    @property
    def jitter(self):
        """ 
        The maximum distance a cellular point can move from it's grid 
        position. The value is relative to the cubic cell spacing of ``1.0``. 
        Setting ``jitter > 0.5`` can generate wrapping artifacts.

	    Default: ``0.45``
        """
        return self._jitter

    @jitter.setter
    def jitter(self, new):
        self._jitter = float(new)
        self._fns.SetCellularJitter(float(new))

    @property
    def distanceIndices(self) -> tuple:
        """
        Sets the two distance indices used for ``distance2X`` return types
	    Default: ``(0, 1)``

	    .. note: * index0 should be lower than index1
	             * Both indices must be ``>= 0``
                 * index1 must be ``< 4``
        """
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
    """
    Holds properties related to the perturbation applied to noise.

    Do not instantiate this class separately from ``Noise``.
    """

    def __init__(self, fns):
        self._fns = fns

        self._perturbType = PerturbType.NoPerturb
        self._amp = 1.0
        self._frequency = 0.5
        self._octaves = 3
        self._lacunarity = 2.0
        self._gain = 2.0
        self._normaliseLength = 1.0

    @property
    def perturbType(self) -> PerturbType:
        """
        The class of perturbation.

        Default: ``PerturbType.NoPeturb``
        """
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
    def amp(self) -> float:
        """
        The maximum distance the input position can be perturbed. The 
        reasonable values of ``amp`` before artifacts are apparent increase with 
        decreased ``frequency``. The default value of ``1.0`` is quite high.

        Default: ``1.0``
        """
        return self._amp

    @amp.setter
    def amp(self, new):
        self._amp = float(new)
        return self._fns.SetPerturbAmp(float(new))

    @property
    def frequency(self) -> float:
        """
        The relative frequency for the perturbation gradient. 

        Default: ``0.5``
        """
        return self._frequency

    @frequency.setter
    def frequency(self, new):
        self._frequency = float(new)
        return self._fns.SetPerturbFrequency(float(new))

    @property
    def octaves(self) -> int:
        """
        The octave count for fractal perturbation types, i.e. the number of 
        log-scaled frequency levels of noise to apply. Generally ``3`` is 
        sufficient for small textures/sprites (256x256 pixels), use larger values for 
        larger textures/sprites.

        Default: ``3``
        """
        return self._octaves

    @octaves.setter
    def octaves(self, new):
        self._octaves = int(new)
        return self._fns.SetPerturbFractalOctaves(int(new))

    @property
    def lacunarity(self) -> float:
        """
        The octave lacunarity (gap-fill) for fractal perturbation types. 
        Lacunarity increases the fineness of fractals.  The appearance of 
        graininess in fractal noise occurs when lacunarity is too high for 
        the given frequency.

        Default: ``2.0``
        """
        return self._lacunarity

    @lacunarity.setter
    def lacunarity(self, new):
        self._lacunarity = float(new)
        return self._fns.SetPerturbFractalLacunarity(float(new))

    @property
    def gain(self) -> float:
        """
        The octave gain for fractal perturbation types. Reflects the ratio 
        of the underlying noise to that of the fractal.  Values > 0.5 up-weight 
        the fractal.

        Default: ``0.5``
        """
        return self._gain

    @gain.setter
    def gain(self, new):
        self._gain = float(new)
        return self._fns.SetPerturbFractalGain(float(new))

    @property
    def normaliseLength(self) -> float:
        """
        The length for vectors after perturb normalising 

        Default: ``1.0``
        """
        return self._normaliseLength

    @normaliseLength.setter
    def normaliseLength(self, new):
        self._normaliseLength = float(new)
        return self._fns.SetPerturbNormaliseLength(float(new))

def _chunk_noise_grid(fns, chunk, chunkStart, chunkAxis, start=[0,0,0]):
    """
    For use by ``concurrent.futures`` to multi-thread ``Noise.genAsGrid()`` calls.
    """
    dataPtr = chunk.__array_interface__['data'][0]
    # print( 'pointer: {:X}, start: {}, shape: {}'.format(dataPtr, chunkStart, chunk.shape) )
    if chunkAxis == 0:
        fns.FillNoiseSet(chunk, chunkStart+start[0], start[1], start[2], *chunk.shape)
    elif chunkAxis == 1:
        fns.FillNoiseSet(chunk, start[0], chunkStart+start[1], start[2], *chunk.shape)
    else:
        fns.FillNoiseSet(chunk, start[0], start[1], chunkStart+start[2], *chunk.shape)

class Noise(object):
    """
    ``Noise`` encapsulates the C++ SIMD class ``FNSObject`` and enables get/set 
    of all relative properties via Python properties.  

    Args:
        seed: The random number (int32) that seeds the random-number generator
            If ``seed == None`` a random integer is generated as the seed.
        numWorkers: The number of threads used for parallel noise generation. 
            If ``numWorkers == None``, the default applied by
            `concurrent.futures.ThreadPoolExecutor` is used.
    """

    def __init__(self, seed: int=None, numWorkers: int=None):

        self._fns = ext.FNS()
        if numWorkers is not None:
            self._num_workers = int(numWorkers)
        else:
            self._num_workers = num_virtual_cores()
        self._asyncExecutor = cf.ThreadPoolExecutor(max_workers = self._num_workers)

        # Sub-classed object handles
        self.fractal = FractalClass(self._fns)
        self.cell = CellularClass(self._fns)
        self.perturb = PerturbClass(self._fns)

        if seed is not None:
            self.seed = seed # calls setter
        else:
            self.seed = np.random.randint(-2147483648, 2147483647)

        # Syncronizers for property getters should use the default values as
        # stated in `FastNoiseSIMD.h`
        self._noiseType = NoiseType.Simplex
        self._frequency = 0.01
        self._axesScales = (1.0, 1.0, 1.0)
        
    @property
    def numWorkers(self) -> int:
        """
        Sets the maximum number of thread workers that will be used for 
        generating noise. Generally should be the number of physical CPU cores 
        on the machine. 
        
        Default: Number of virtual cores on machine.
        """
        return self._num_workers

    @numWorkers.setter
    def numWorkers(self, N_workers) -> int:
        N_workers = int(N_workers)
        if N_workers <= 0:
            raise ValueError('numWorkers must be greater than 0')
        self._num_workers = N_workers
        self._asyncExecutor = cf.ThreadPoolExecutor(max_workers = N_workers)

    @property 
    def seed(self) -> int:
        """
        The random-number seed used for generation of noise. 

        Default: ``numpy.random.randint()``
        """
        return self._fns.GetSeed()

    @seed.setter
    def seed(self, new):
        return self._fns.SetSeed(int(np.int32(new)))

    @property
    def frequency(self) -> float:
        """
        The frequency of the noise, lower values result in larger noise features.

        Default: ``0.01``
        """
        return self._frequency

    @frequency.setter
    def frequency(self, new):
        self._frequency = float(new)
        return self._fns.SetFrequency(float(new))

    @property
    def noiseType(self) -> NoiseType:
        """
        The class of noise. 
        
        Default: ``NoiseType.Simplex`` 
        """
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
    def axesScales(self) -> tuple:
        """
        Sets the FastNoiseSIMD axes scales, which allows for non-square 
        voxels. Indirectly affects `frequency` by changing the voxel pitch.

        Default: ``(1.0, 1.0, 1.0)`` 
        """
        return self._axesScales

    @axesScales.setter
    def axesScales(self, new: tuple):
        if not hasattr(new, '__len__') or len(new) != 3:
            raise ValueError( 'axesScales must be a length 3 array/list/tuple' )
        
        self._axesScales = new
        return self._fns.SetAxesScales(*new)

    def genAsGrid(self, shape=[1,1024,1024], start=[0,0,0]) -> np.ndarray:
        """
        Generates noise according to the set properties along a rectilinear 
        (evenly-spaced) grid.  

        Args:
            shape: Tuple[int]
                the shape of the output noise volume. 
            start: Tuple[int]
                the starting coordinates for generation of the grid.
                I.e. the coordinates are essentially `start: start + shape`

        Example::

            import numpy as np
            import pyfastnoisesimd as fns 
            noise = fns.Noise()
            result = noise.genFromGrid(shape=[256,256,256], start=[0,0,0])
            nextResult = noise.genFromGrid(shape=[256,256,256], start=[256,0,0])
        """
        if isinstance(shape, (int, np.integer)):
            shape = (shape,)

        # There is a minimum array size before we bother to turn on futures.
        size = np.product(shape)
        # size needs to be evenly divisible by ext.SIMD_ALINGMENT
        if np.remainder(size, ext.SIMD_ALIGNMENT/np.dtype(np.float32).itemsize) != 0.0:
            raise ValueError('The size of the array (in bytes) must be evenly divisible by the SIMD vector length')

        result = empty_aligned(shape)

        # Shape could be 1 or 2D, so we need to expand it with singleton 
        # dimensions for the FillNoiseSet call
        if len(start) == 1:
            start = [start[0], 0, 0]
        elif len(start) == 2:
            start = [start[0], start[1], 1]
        else:
            start = list(start)
        start_zero = start[0]

        if self._num_workers <= 1 or size < _MIN_CHUNK_SIZE:
            # print('Grid single-threaded')
            if len(shape) == 1:
                shape = (*shape, 1, 1)
            elif len(shape) == 2:
                shape = (*shape, 1)
            else:
                shape = shape

            self._fns.FillNoiseSet(result, *start, *shape)
            return result

        # else run in threaded mode.
        n_chunks = np.minimum(self._num_workers, shape[0]) 
        # print(f'genAsGrid using {n_chunks} chunks')
        
        # print('Grid multi-threaded')
        workers = []
        for I, (chunk, consumed) in enumerate(aligned_chunks(result, n_chunks, axis=0)):
            # print(f'{I}: Got chunk of shape {chunk.shape} with {consumed} consumed')

            if len(chunk.shape) == 1:
                chunk_shape = (*chunk.shape, 1, 1)
            elif len(chunk.shape) == 2:
                chunk_shape = (*chunk.shape, 1)
            else:
                chunk_shape = chunk.shape

            start[0] = start_zero + consumed
            # print('len start: ', len(start), ', len shape: ', len(chunk_shape))
            peon = self._asyncExecutor.submit(self._fns.FillNoiseSet, 
                        chunk, *start, *chunk_shape)
            workers.append(peon)

        for peon in workers:
            peon.result()
        # For memory management we have to tell NumPy it's ok to free the memory
        # region when it is dereferenced.
        # self._fns._OwnSplitArray(noise)
        return result

    def genFromCoords(self, coords: np.ndarray) -> np.ndarray:
        """
        Generate noise from supplied coordinates, rather than a rectilinear grid.
        Useful for complicated shapes, such as tesselated surfaces.

        Args:
            coords: 3-D coords as generated by ``fns.empty_coords()`` 
                and filled with relevant values by the user.
            
        Returns: 
            noise: a shape (N,) array of the generated noise values.

        Example::

            import numpy as np
            import pyfastnoisesimd as fns 
            numCoords = 256
            coords = fns.empty_coords(3,numCoords)
            # <Set the coordinate values, it is a (3, numCoords) array
            coords[0,:] = np.linspace(-np.pi, np.pi, numCoords)
            coords[1,:] = np.linspace(-1.0, 1.0, numCoords)
            coords[2,:] = np.zeros(numCoords)
            noise = fns.Noise()
            result = noise.genFromCoords(coords)

        """

        if not isinstance(coords, np.ndarray):
            raise TypeError('`coords` must be of type `np.ndarray`, not type: ', type(coords))
        if coords.ndim != 2:
            raise ValueError('`coords` must be a 2D array')
        shape = coords.shape
        if shape[0] != 3:
            raise ValueError('`coords.shape[0]` must equal 3')
        if not check_alignment(coords):
            raise ValueError('Memory alignment of `coords` is not valid')
        if coords.dtype != np.float32:
            raise ValueError('`coords` must be of dtype `np.float32`')
        if np.remainder(coords.shape[1], ext.SIMD_ALIGNMENT/np.dtype(np.float32).itemsize) != 0.0:
            raise ValueError('The number of coordinates must be evenly divisible by the SIMD vector length')

        itemsize = coords.dtype.itemsize
        result = empty_aligned(shape[1])
        if self._num_workers <= 1 or shape[1] < _MIN_CHUNK_SIZE:
            self._fns.NoiseFromCoords(result, 
                    coords[0,:], coords[1,:], coords[2,:], shape[1], 0)
            return result

        n_chunks = np.minimum(self._num_workers, 
                shape[1] * itemsize / ext.SIMD_ALIGNMENT)

        workers = []
        # for I, ((result_chunk, r_offset), (coord_chunk, offset)) in enumerate(zip(
        #             aligned_chunks(result, self._num_workers, axis=0),
        #             aligned_chunks(coords, self._num_workers, axis=1))):
        for I, (result_chunk, offset) in enumerate(
                    aligned_chunks(result, self._num_workers, axis=0)):

            # aligned_size = int(vect_len*np.ceil(result_chunk.size/vect_len))
            # print(f'    {I}: Got chunk of length {result_chunk.size}, AlignedSize would be: {aligned_size}')
            # print('    Offset: ', offset, ', offset error: ', offset % 8)

            # zPtr = (coords[0,:].ctypes.data + offset) % 8
            # yPtr = (coords[1,:].ctypes.data + offset) % 8
            # xPtr = (coords[2,:].ctypes.data + offset) % 8
            # print(f'    Pointer alignment: {zPtr, yPtr, xPtr}')
            # peon = self._asyncExecutor.submit(self._fns.NoiseFromCoords, result, 
            #     coords[0,:], coords[1,:], coords[2,:], aligned_size, offset)
            peon = self._asyncExecutor.submit(self._fns.NoiseFromCoords, result, 
                coords[0,:], coords[1,:], coords[2,:], result_chunk.size, offset)
            workers.append(peon)

        for peon in workers:
            peon.result()

        return result
        
