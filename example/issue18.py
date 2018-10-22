import numpy as np
import pyfastnoisesimd as fns
import pyfastnoisesimd.extension as ext
import psutil, gc, sys, os
from sys import getrefcount
import matplotlib.pyplot as plt

process = psutil.Process(os.getpid())
kB = 1024
n_cycles = 64
mem = np.zeros(n_cycles+2)

print('\n\n=== Starting genAsGrid ===')
n = fns.Noise()
mem[0] = process.memory_info().rss / kB
for I in range(n_cycles):
    # print(f"    Iteration #{I}")
    result = n.genAsGrid((641, 473, 5))
    mem[I+1] = process.memory_info().rss / kB

del n, result
mem[-1] = process.memory_info().rss / kB

print('\n\n=== Starting genFromCoords ===')
n = fns.Noise()
mem2 = np.zeros(n_cycles+2)
mem2[0] = process.memory_info().rss / kB
len_coords = 3470435

coords = fns.empty_aligned((3,len_coords))
for I in range(n_cycles):
    # print(f"\n    Iteration #{I}")
    coords[0,:len_coords] = np.pi
    coords[1,:len_coords] = 0.5
    coords[2,:len_coords] = np.exp(1)
    result = n.genFromCoords(coords)

    mem2[I+1] = process.memory_info().rss / kB

del n, result, coords
mem2[-1] = process.memory_info().rss / kB


plt.figure(figsize=(16,8))
plt.subplot(121)
plt.plot(mem)
plt.ylabel('Memory usage (kB)')
plt.title('genAsGrid')
plt.subplot(122)
plt.plot(mem2)
plt.ylabel('Memory usage (kB)')
plt.title('genFromCoords')

print('genAsGrid  : Leaking {} kB per call'.format((mem[0] - mem[-1])/n_cycles))
print('genAsCoords: Leaking {} kB per call'.format((mem2[0] - mem2[-1])/n_cycles))
print('DONE')
plt.show()
