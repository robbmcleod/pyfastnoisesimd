import numpy as np
import pyfastnoisesimd as fns

X = 4096
numWorkers = 1

print("instant Noise")
noise = fns.Noise(seed=None, numWorkers=numWorkers)
print("emptyCoords")
coords = fns.emptyCoords(X)
coords[0,:] = np.arange(X)
coords[1,:] = np.arange(-X//2, X//2)
coords[2,:] = np.linspace(-1.0, 1.0, X)
print("genFromCoords")
noise.genFromCoords(coords)

# Re-use coords to make sure the array isn't accidentally free'd 
# in FastNoiseSIMD
print("Re-use noise")
noise2 = fns.Noise(seed=None, numWorkers=numWorkers)
print("genFromCoords #2")
noise2.genFromCoords(coords)