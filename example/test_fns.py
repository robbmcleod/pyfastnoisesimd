import numpy as np
import matplotlib.pyplot as plt
import pyfastnoisesimd as fns
import time
import numpy.testing as npt

# N_thread = fns.cpu_info['count']
N_thread = 8
N = [8,1024,1024]

SEED_CELL = np.random.randint(2**31)
SEED_PERLIN = np.random.randint(2**31)

fns.setNumWorkers(1)
t0 = time.perf_counter()
# Plot cellular noise with a gradient perturbation
# In this case all possible function keyword arguments are shown.
cellular_s = fns.generate( size=N, start=[0,0,0], 
              seed=SEED_CELL, freq=0.005, noiseType='Cellular', axisScales=[N[-1]/N[0],1.0,1.0], 
              fracType='FBM', fracOctaves=4, 
              fracLacunarity=3.0, fracGain=0.5, 
              cellReturnType='Distance', cellDistFunc='Euclidean',
              cellNoiseLookup='Simplex', cellNoiseLookupFreq=0.2, 
              cellDist2Ind=[0,1], cellJitter=0.5,
              perturbType='Gradient', perturbAmp=1.0, perturbFreq=0.7, perturbOctaves=5,
              perturbLacunarity=2.0, perturbGain=0.5, perturbNormLen=1.0,   )
t1 = time.perf_counter()
# Plot Perlin noise without any perturbation
perlin_s = fns.generate( size=N, start=[0,0,0], 
              seed=SEED_PERLIN, freq=0.02, noiseType='PerlinFractal', axisScales=[N[-1]/N[0],1.0,1.0], 
              fracType='FBM', fracOctaves=4, 
              fracLacunarity=2.0, fracGain=0.5,
              perturbType=None )
t2 = time.perf_counter()

print( "#### Single threaded mode ####")
print( "Computed {} voxels cellular noise in {:.3f} s".format( np.prod(N), t1-t0) )
print( "    {:.1f} ns/voxel".format( 1E9*(t1-t0)/np.prod(N) ) )
print( "Computed {} voxels Perlin noise in {:.3f} s".format( np.prod(N), t2-t1) )
print( "    {:.1f} ns/voxel".format( 1E9*(t2-t1)/np.prod(N) ) )
print( "" )


fns.setNumWorkers(N_thread)
t3 = time.perf_counter()
# Plot cellular noise with a gradient perturbation
# In this case all possible function keyword arguments are shown.
cellular = fns.generate( size=N, start=[0,0,0], 
              seed=SEED_CELL, freq=0.005, noiseType='Cellular', axisScales=[N[-1]/N[0],1.0,1.0], 
              fracType='FBM', fracOctaves=4, 
              fracLacunarity=3.0, fracGain=0.5, 
              cellReturnType='Distance', cellDistFunc='Euclidean',
              cellNoiseLookup='Simplex', cellNoiseLookupFreq=0.2, 
              cellDist2Ind=[0,1], cellJitter=0.5,
              perturbType='Gradient', perturbAmp=1.0, perturbFreq=0.7, perturbOctaves=5,
              perturbLacunarity=2.0, perturbGain=0.5, perturbNormLen=1.0,   )
t4 = time.perf_counter()
# Plot Perlin noise without any perturbation
perlin = fns.generate( size=N, start=[0,0,0], 
              seed=SEED_PERLIN, freq=0.02, noiseType='PerlinFractal', axisScales=[N[-1]/N[0],1.0,1.0], 
              fracType='FBM', fracOctaves=4, 
              fracLacunarity=2.0, fracGain=0.5,
              perturbType=None )
t5 = time.perf_counter()

print( "#### Multi-threaded ({} threads) mode ####".format(N_thread) )
print( "Computed {} voxels cellular noise in {:.3f} s".format( np.prod(N), t4-t3) )
print( "    {:.1f} ns/voxel".format( 1E9*(t4-t3)/np.prod(N) ) )
print( "    {:.1f} % thread scaling".format( (t1-t0)/(t4-t3)*100.0  ) )
print( "Computed {} voxels Perlin noise in {:.3f} s".format( np.prod(N), t5-t4) )
print( "    {:.1f} ns/voxel".format( 1E9*(t5-t4)/np.prod(N) ) )
print( "    {:.1f} % thread scaling".format( (t2-t1)/(t5-t4)*100.0  ) )


# Check that the results are the same from single and multi-threading
npt.assert_array_almost_equal( cellular, cellular_s )
npt.assert_array_almost_equal( perlin, perlin_s )

'''
# Simple plotting.  matplotlib can also make movies.
plt.figure()
figManager = plt.get_current_fig_manager()
# Tkinter
# figManager.window.state('zoomed')
# Qt
figManager.window.showMaximized()
for J in range(N[0]):
    plt.subplot(121)
    plt.imshow( cellular[J,:,:] )
    plt.title( "Cellular #{}".format(J) )
    plt.subplot(122)
    plt.imshow( perlin[J,:,:] )
    plt.title( "Perlin #{}".format(J) )
    plt.pause(0.5)

'''
