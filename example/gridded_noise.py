import numpy as np
import pyfastnoisesimd as fns
import time
import numpy.testing as npt


N_thread = fns.num_virtual_cores()
shape = [8, 1024, 1024]
print( 'Array shape: {}'.format(shape) )

# Plot cellular noise with a gradient perturbation
cellular = fns.Noise(numWorkers=1)
print( 'SIMD level supported: {}'.format(fns.extension.SIMD_LEVEL))
# The Noise class uses properties to map to the FastNoiseSIMD library Set<...> functions
cellular.seed = 42
cellular.noiseType = fns.NoiseType.Cellular
cellular.frequency = 0.005
cellular.axesScales = [shape[-1]/shape[0],1.0,1.0]

cellular.cell.returnType = fns.CellularReturnType.Distance
cellular.cell.distanceFunc = fns.CellularDistanceFunction.Euclidean
cellular.cell.noiseLookupType = fns.NoiseType.Simplex
cellular.cell.lookupFrequency = 0.2
cellular.cell.jitter = 0.5
cellular.cell.distanceIndices = (0,1)

cellular.fractal.octaves = 4

cellular.perturb.perturbType = fns.PerturbType.Gradient
cellular.perturb.amp = 1.0
cellular.perturb.frequency = 0.7
cellular.perturb.octaves = 5
cellular.perturb.lacunarity = 2.0
cellular.perturb.gain = 0.5
cellular.perturb.normaliseLength = 1.0

# Plot PerlinFractal noise without any perturbation
perlin = fns.Noise(numWorkers=1)
perlin.noiseType = fns.NoiseType.PerlinFractal
perlin.frequency = 0.02
perlin.axesScales = [shape[-1]/shape[0],1.0,1.0]

perlin.fractal.fractalType = fns.FractalType.FBM
perlin.fractal.octaves = 4
perlin.fractal.lacunarity = 2.1
perlin.fractal.gain = 0.45

perlin.perturb.perturbType = fns.PerturbType.NoPerturb

t0 = time.perf_counter()
cell_single = cellular.genAsGrid(shape=shape, start=[0,0,0])
t1 = time.perf_counter()
perlin_single = perlin.genAsGrid(shape=shape, start=[0,0,0])
t2 = time.perf_counter()

print( '#### Single threaded mode ####')
print( 'Computed {} voxels cellular noise in {:.3f} s'.format( np.prod(shape), t1-t0) )
print( '    {:.1f} ns/voxel'.format( 1E9*(t1-t0)/np.prod(shape) ) )
print( 'Computed {} voxels Perlin noise in {:.3f} s'.format( np.prod(shape), t2-t1) )
print( '    {:.1f} ns/voxel\shape'.format( 1E9*(t2-t1)/np.prod(shape) ) )

cellular.numWorkers = N_thread
perlin.numWorkers = N_thread
t3 = time.perf_counter()
cell_multi = cellular.genAsGrid(shape=shape, start=[0,0,0])
t4 = time.perf_counter()
perlin_multi = perlin.genAsGrid(shape=shape, start=[0,0,0])
t5 = time.perf_counter()

print( '#### Multi-threaded ({} threads) mode ####'.format(N_thread) )
print( 'Computed {} voxels cellular noise in {:.3f} s'.format( np.prod(shape), t4-t3) )
print( '    {:.1f} ns/voxel'.format( 1E9*(t4-t3)/np.prod(shape) ) )
print( '    {:.1f} % thread scaling'.format( (t1-t0)/(t4-t3)*100.0  ) )
print( 'Computed {} voxels Perlin noise in {:.3f} s'.format( np.prod(shape), t5-t4) )
print( '    {:.1f} ns/voxel'.format( 1E9*(t5-t4)/np.prod(shape) ) )
print( '    {:.1f} % thread scaling'.format( (t2-t1)/(t5-t4)*100.0  ) )

# Check that the results are the same from single and multi-threading
npt.assert_array_almost_equal(cell_single, cell_multi)
npt.assert_array_almost_equal(perlin_single, perlin_multi)

# Simple plotting.  matplotlib can also make movies.
'''
import matplotlib.pyplot as plt

plt.figure()
figManager = plt.get_current_fig_manager()
# Tkinter
figManager.window.state('zoomed')
# Qt
# figManager.window.showMaximized()
for J in range(shape[0]):
    plt.subplot(121)
    plt.imshow( cell_single[J,:,:] )
    plt.title( 'Cellular #{}'.format(J) )
    plt.subplot(122)
    plt.imshow( perlin_single[J,:,:] )
    plt.title( 'Perlin #{}'.format(J) )
    plt.pause(0.5)
'''

