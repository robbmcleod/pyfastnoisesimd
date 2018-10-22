import pyfastnoisesimd as fns
import numpy as np
import time
from hihi import color_print as cp

n = fns.Noise()
n.numWorkers = 4
n_cycles = 12

print(cp.red('Is Numpy resizing the arrays after we exit scope?'))
coords = [None] * n_cycles
for I in range(n_cycles):
    len_coords = 4096 + I + 1
    coords[I] = fns.empty_coords(len_coords)
    print(cp.yellow("\n    Iteration #{I} with len: {len_coords}"))
    # print(f"\n\n    Iteration #{I} with len: {len_coords}")
    coords[I][0,:] = np.pi
    coords[I][1,:] = 0.5
    coords[I][2,:] = np.exp(1)
    result = n.genFromCoords(coords[I])

    # time.sleep(0.2)

print(cp.magenta('== Done =='))
# print('== Done ==')