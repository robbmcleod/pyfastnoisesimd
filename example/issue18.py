import numpy as np
import pyfastnoisesimd as fns
import pyfastnoisesimd.extension as ext
import psutil, gc, sys, os
from sys import getrefcount
# import matplotlib.pyplot as plt

process = psutil.Process(os.getpid())
kB = 1024
n_cycles = 24
mem = np.zeros(n_cycles+2)
'''
print('\n\n=== Starting genAsGrid ===')
n = fns.Noise()
n.numWorkers = 1
mem[0] = process.memory_info().rss / kB
for I in range(n_cycles):
    print(f"    Iteration #{I}")
    # Repeat as necessary, but be careful not to OOM yourself
    result = n.genAsGrid((641,473,5))
    mem[I+1] = process.memory_info().rss / kB

del n, result
mem[-1] = process.memory_info().rss / kB

plt.figure()
plt.plot(mem)
plt.ylabel('Memory usage (kB)')
plt.title('genAsGrid')

print('genAsGrid: Leaking {} kB per call'.format((mem[0] - mem[-1])/n_cycles))
'''

def emptyAligned(shape): 
    """
    For high performance PyFastNoiseSIMD's backing library requires data that 
    is exactly aligned in memory address to the SIMD vector instruction length.
    So the starting address of the memory needs to be exactly divisible by 
    the SIMD vector size.

    Args:
        shape: a sequence of len 1-3 with the desired array dimensions.
    Returns:
        Returns a `numpy.ndarray` of dtype `np.float32` that is aligned to the 
        current SIMD level.
    """
    simd_len = ext.AlignedSize(1) # SIMD vector length in bytes
    dim_mult = simd_len // 4

    print('SIMD length: ', simd_len, ", dim mult: ", dim_mult)
    # First try: make each shape axis an integer number of SIMD vectors
    if not hasattr(shape, '__len__'):
        shape = [shape]
    if len(shape) > 3:
        raise ValueError('PyFastNoiseSIMD can only generate 1-3D noise.')

    
    ext_shape = [int(np.ceil(x/dim_mult))*dim_mult for x in shape]
    # We don't have to extend the Z-axis, however
    # ext_shape[0] = shape[0]
    # ext_shape[-1] = ext_shape[-1] + 2
    print(f'Promoting shape {shape} to {ext_shape}')

    # ext_shape = [x for x in shape]

    n_elements = np.product(ext_shape)

    n_bytes = n_elements * 4

    array = np.empty(n_bytes + simd_len, dtype=np.uint8)

    array_start = hex(array.ctypes.data)
    array_end = hex(array.ctypes.data + n_bytes)

    align_error = array.ctypes.data % simd_len
    print(f'Total allocated range: {array_start} to {array_end}, alignment error {align_error} bytes')
    
    offset = 0 if align_error == 0 else (simd_len - align_error)
    
    # Is there a problem here because we convert to float32 early?
    # Do we need to reshape first?
    view = array[offset: offset + n_bytes]
    # Now we need to convert to float32
    view = view.view(np.float32)
    # And then reshape
    view = view.reshape(ext_shape)
    # And then slice
    if len(shape) == 1:
        return view[:shape[0]]
    elif len(shape) == 2:
        return view[:shape[0], :shape[1]]
    else:
        return view[:shape[0], :shape[1], :shape[2]]


print('\n\n=== Starting genFromCoords ===')
n = fns.Noise()
print('SIMD level: ', n.SIMDLevel)
n.numWorkers = 1
mem[0] = process.memory_info().rss / kB
len_coords = 347
print('--Alloc result--')
result = emptyAligned(len_coords)

print('--Alloc coords--')
# coords = emptyAligned((3,len_coords))
# coords = np.empty((3,len_coords), dtype='float32')
# x = emptyAligned(len_coords)
# y = emptyAligned(len_coords)
# z = emptyAligned(len_coords)
x = np.empty(len_coords, dtype=np.float32)
y = np.empty(len_coords, dtype=np.float32)
z = np.empty(len_coords, dtype=np.float32)

# print(coords.shape)
for I in range(n_cycles):
    print(f"\n    Iteration #{I}")
    print( '    -------------')
    # Repeat as necessary, but be careful not to OOM yourself
    # coords[0,:] = np.pi
    # coords[1,:] = 0.5
    # coords[2,:] = np.exp(1)
    # n._fns.NoiseFromCoords(result, coords[0,:], coords[1,:], coords[2,:], len_coords, 0)

    x[:] = np.pi
    y[:] = 0.5
    z[:] = np.exp(1)
    n._fns.NoiseFromCoords(result, x, y, z, len_coords, 0)
    # print(result)
    mem[I+1] = process.memory_info().rss / kB

del n, result
mem[-1] = process.memory_info().rss / kB

# plt.figure()
# plt.plot(mem)
# plt.ylabel('Memory usage (kB)')
# plt.title('genFromCoords')

print('DONE')
# plt.show()
