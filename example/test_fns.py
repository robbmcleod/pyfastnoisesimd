import numpy as np
import matplotlib.pyplot as plt
import pyfastnoisesimd as fns
import time

N = [4,1024,1024]

t0 = time.perf_counter()
# Plot cellular noise with a gradient perturbation
# In this case all possible function keyword arguments are shown.
cellular = fns.generate( size=N, start=[0,0,0], 
              seed=np.random.randint(2**31), freq=0.005, noiseType='Cellular', axisScales=[N[-1]/N[0],1.0,1.0], 
              fracType='FBM', fracOctaves=4, 
              fracLacunarity=3.0, fracGain=0.5, 
              cellReturnType='Distance', cellDistFunc='Euclidean',
              cellNoiseLookup='Simplex', cellNoiseLookupFreq=0.2, 
              cellDist2Ind=[0,1], cellJitter=0.5,
              perturbType='Gradient', perturbAmp=1.0, perturbFreq=0.7, perturbOctaves=5,
              perturbLacunarity=2.0, perturbGain=0.5, perturbNormLen=1.0,   )
t1 = time.perf_counter()
# Plot Perlin noise without any perturbation
perlin = fns.generate( size=N, start=[0,0,0], 
              seed=np.random.randint(2**31), freq=0.02, noiseType='PerlinFractal', axisScales=[N[-1]/N[0],1.0,1.0], 
              fracType='FBM', fracOctaves=4, 
              fracLacunarity=2.0, fracGain=0.5,
              perturbType=None )
t2 = time.perf_counter()

print( "Computed {} voxels cellular noise in {} s".format( np.prod(N), t1-t0) )
print( "    {} ns/voxel".format( 1E9*(t1-t0)/np.prod(N) ) )
print( "Computed {} voxels Perlin noise in {} s".format( np.prod(N), t2-t1) )
print( "    {} ns/voxel".format( 1E9*(t2-t1)/np.prod(N) ) )

# Simple plotting.  matplotlib can also make movies.
plt.figure()
figManager = plt.get_current_fig_manager()
# Tkinter
figManager.window.state('zoomed')
# Qt
# figManager.window.showMaximized()
for J in range(N[0]):
    plt.imshow( cellular[J,:,:] )
    plt.title( "Cellular #{}".format(J) )
    plt.pause(0.5)

for J in range(N[0]):
    plt.imshow( perlin[J,:,:] )
    plt.title( "Perlin #{}".format(J) )
    plt.pause(0.5)