import pyfastnoisesimd as fns
import numpy as np

n = fns.Noise()
n.numWorkers = 8
len_coords = 32*32 + 2
n_cycles = 5

for I in range(n_cycles):
    print(f"\n    Iteration #{I}")
    # result = n.genAsGrid(shape=[123, 111, 107])
    # result = n.genAsGrid(shape=[123, 111])
    result = n.genAsGrid(shape=[11111])
print('== Done ==')