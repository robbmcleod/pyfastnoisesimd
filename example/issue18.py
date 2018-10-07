import numpy as np
import pyfastnoisesimd as fns
import psutil, gc, sys

gc.set_debug(gc.DEBUG_UNCOLLECTABLE)
# gc.disable()

n = fns.Noise()
n.numWorkers = 1
n_cycles = 128
mem = np.zeros(n_cycles)
print('Starting genAsGrid')
for I in range(n_cycles):  
    # Repeat as necessary, but be careful not to OOM yourself
    result = n.genAsGrid((1024, 1024, 1))
    mem[I] = psutil.virtual_memory().available

print('Result flags: ', result.flags)
print('Result has {} ref counts'.format(sys.getrefcount(result)))
print('Leaking {} kB per call'.format((mem[0] - mem[-1])/n_cycles/1024))
print('Garbage: ', gc.garbage)
