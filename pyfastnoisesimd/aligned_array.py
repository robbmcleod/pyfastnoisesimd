import numpy as np
from threading import Lock

import pyfastnoisesimd.extension as ext

_MIN_CHUNK_SIZE = 4096
del_count = 0

class AlignedArray(np.ndarray):
    """
    A `numpy.ndarray` mapping to memory aligned for fast SIMD vectorized 
    instructions. `dtype` is always `np.float32`.

    Parameters
    ----------
    shape: Tuple[*int]
        a tuple ints giving of array dimensions. If array dimensions are not 
        modulo zero relative to the SIMD instruction length the array will be 
        grown. 
    pieces: int
        The number of pieces to break the array into for multi-threaded processing.
    """

    def __new__(subtype, shape):
        if len(shape) > 3 or len(shape) == 0:
            raise ValueError('FastNoiseSIMD is implemented for 1 to 3D data')
        elif len(shape) == 1:
            shape = (shape[0], 0, 0)
        elif len(shape) == 2:
            shape = (shape[0], shape[1], 0)

        # Zero for any dimension length indicates that dimension should not
        # exist in the returned array.
        array = ext.EmptySet(*shape).view(subtype)

        # As __del__ can be called multiple times, protect the actual array 
        # deletion with a mutex.
        setattr(array, '_delete_mutex', Lock())
        return array

    def __array_finalize__(self, obj):
        if obj is None:
            return

        self._delete_mutex = getattr(obj, '_delete_mutex', Lock())
        print('__array_finalize__ at ', hex(self.__array_interface__['data'][0]))


    def __del__(self):
        # In CPython __del__ can be called multiple times; if we free twice, 
        # that's a fault, so we use a mutex to protect it.
        print('__del__ on array at ', hex(self.__array_interface__['data'][0]))
        if self._delete_mutex.acquire(blocking=False):
            ext.FreeSet(self)

    # def chunks(self, n_chunks):
    #     """
    #     TODO: make an iterator. Should return chunk and start.
    #     """

