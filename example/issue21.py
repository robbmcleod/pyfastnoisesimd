import numpy as np
import pyfastnoisesimd as fns


# Num workers does not seem to matter.
# It only seems to be an issue if the middle dimension is not divisible 
# by the SIMD length?
n = fns.Noise(numWorkers=1)
shape = [27, 127, 1]
for I in range(10):
    # print(I)
    res = n.genAsGrid(shape)
print('Finished successfully')