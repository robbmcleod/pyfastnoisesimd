import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pyfastnoisesimd as fns
import pyfastnoisesimd.extension as ext
from time import perf_counter
import numpy.testing as npt

# Let us make a Gall projected random-noise heightmap, to demonstrate
# correct wrapping behavior at the edges.
# https://en.wikipedia.org/wiki/Gall%E2%80%93Peters_projection
#
# It is somewhat better behaved numerically than the classic Mercator projection.
# https://en.wikipedia.org/wiki/Mercator_projection

freq = 0.1
shape = (1333, 2000)
# Make a sphere with a radius of 40.0 `units`
radius = 40.0 
inverseRadius = 1.0/radius
# Generate an evenly spaced 2D mesh of projection X-Y coordinates over its surface.
X, Y = np.meshgrid( np.linspace(-np.pi*radius, np.pi*radius, shape[1], endpoint=False), 
                    np.linspace(-np.pi*radius, np.pi*radius, shape[0], endpoint=False))
X = X.astype('float32').ravel()
Y = Y.astype('float32').ravel()
# Generate lambda (meridian) and phi (parallel) angles in radians, from the 
# projection formula
meridian = X * inverseRadius
parallel = np.arcsin((0.5*inverseRadius) * Y)

# Show the sampling (essentially Tissot's indicatrices):
'''
plt.figure()
plt.plot( meridian, parallel, 'k.' )
plt.xlabel( 'meridian (rad)' )
plt.ylabel( 'Parallel (rad)' )
plt.title( 'meridian and parallel sampling space')
'''

# Make your Noise object
bumps = fns.Noise(seed=42, numWorkers=1)
bumps.noiseType = fns.NoiseType.Perlin
bumps.frequency = freq
print( 'FastNoiseSIMD maximum supported SIMD instruction level: {}'.format(bumps.SIMDLevel) )

# Generate an empty-array of 3D cartesian coordinates. You can use NumPy
# arrays from other sources but see the warnings in `Noise.genFromCoords()`
coords = fns.emptyCoords(meridian.size)

# Fill coords with Cartesian coordinates in 3D
# (Note that normally zenith starts at 0.0 rad, whereas we want the equator to be 
# 0.0 rad, so we swap cos/sin for the `parallel` axis)
coords[0,:meridian.size] = radius*np.sin(parallel)                   # Z
coords[1,:meridian.size] = radius*np.cos(parallel)*np.sin(meridian)  # Y
coords[2,:meridian.size] = radius*np.cos(parallel)*np.cos(meridian)  # X

# Check that we have spherical coordinates as expected:
'''
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter( coords[2,:], coords[1,:], coords[0,:], 'k.' )
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D coordinate sampling')
'''

t2 = perf_counter()
result = bumps.genFromCoords(coords)
t3 = perf_counter()
print( 'Generated noise from {} coordinates with {} workers in {:.3e} s ({:.1f} ns/pixel)'.format(meridian.size, 1, t3-t2, 1e9*(t3-t2)/result.size) )

# Unravel
projection = result.reshape(shape)

# Here we plot our projections, with both the conventional view, and a view 
# rolled to see the far side to observe that the junction is smooth (i.e. that 
# the heightmap is wrapped )
plt.figure()
plt.subplot(121)
plt.imshow(projection, cmap='Greys')
plt.title('Gall-Peters Projection')
plt.subplot(122)
plt.imshow( np.roll(projection, shape[1]//2, axis=1), cmap='Greys' )
plt.title('Rotated 180\u00b0 to see junction')
plt.show()