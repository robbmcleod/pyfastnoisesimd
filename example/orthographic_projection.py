import numpy as np
from time import perf_counter
import pyfastnoisesimd as fns

import matplotlib.pyplot as plt
plt.style.use('dark_background')
from mpl_toolkits.mplot3d import Axes3D

N_thread = fns.cpu_info['count']
N_thread = 1

def orthoProject(noise, tile2: int=512, p0: float=0., l0: float=0.) -> np.ndarray:
    '''
    Render noise onto a spherical surface with an Orthographic projection.

    Args:
        noise: a `pyfastnoisesimd.Noise` object.
        tile2: the half-width of the returned array, i.e. return will have shape ``(2*tile2, 2*tile2)``.
        p0: the central parallel (i.e. latitude)
        l0: the central meridian (i.e. longitude)

    See also: 
    
        https://en.wikipedia.org/wiki/Orthographic_projection_in_cartography

    '''
    t0 = perf_counter()
    # We use angular coordinates as there's a lot of trig identities and we 
    # can make some simplifications this way.
    xVect = np.linspace(-0.5*np.pi, 0.5*np.pi, 2*tile2, endpoint=True).astype('float32')
    xMesh, yMesh = np.meshgrid(xVect, xVect)
    p0 = np.float32(p0)
    l0 = np.float32(l0)

    # Our edges are a little sharp, one could make an edge filter from the mask
    # mask = xMesh*xMesh + yMesh*yMesh <= 0.25*np.pi*np.pi
    # plt.figure()
    # plt.imshow(mask)
    # plt.title('Mask')

    # Check an array of coordinates that are inside the disk-mask
    valids = np.argwhere(xMesh*xMesh + yMesh*yMesh <= 0.25*np.pi*np.pi)
    xMasked = xMesh[valids[:,0], valids[:,1]]
    yMasked = yMesh[valids[:,0], valids[:,1]]
    maskLen = xMasked.size

    # These are simplified equations from the linked Wikipedia article.  We 
    # have to back project from 2D map coordinates [Y,X] to Cartesian 3D 
    # noise coordinates [W,V,U]
    # TODO: one could accelerate these calculations with `numexpr` or `numba`
    one = np.float32(0.25*np.pi*np.pi) 
    rhoStar = np.sqrt(one - xMasked*xMasked - yMasked*yMasked)
    muStar =  yMasked*np.cos(p0) + rhoStar*np.sin(p0)
    conjMuStar = rhoStar*np.cos(p0) - yMasked*np.sin(p0)
    alphaStar = l0 + np.arctan2(xMasked, conjMuStar) 
    sqrtM1MuStar2 = np.sqrt(one - muStar*muStar)

    coords = fns.emptyCoords(maskLen) # ask fastnoisesimd for a properly-shaped array
    coords[0,:maskLen] = muStar                             # W
    coords[1,:maskLen] = sqrtM1MuStar2 * np.sin(alphaStar)  # V
    coords[2,:maskLen] = sqrtM1MuStar2 * np.cos(alphaStar)  # U

    # Check our coordinates in 3-D to make sure the shape is correct:
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter( coords[2,:maskLen], coords[1,:maskLen], coords[0,:maskLen], 'k.' )
    # ax.set_xlabel('U')
    # ax.set_ylabel('V')
    # ax.set_zlabel('W')
    # ax.set_title('3D coordinate sampling')

    pmap = np.full( (2*tile2, 2*tile2), -np.inf, dtype='float32')
    t1 = perf_counter()
    result = noise.genFromCoords(coords)
    t2 = perf_counter()
    pmap[valids[:,0], valids[:,1]] = result[:maskLen]
    print("Generated {} coords in {:.2e} s".format(maskLen, t1-t0))
    print("Generated noise for {} coords with {} workers in {:.3e} s".format(maskLen, noise.numWorkers, t2-t1))
    print("    {:.1f} ns/pixel".format(1e9*(t2-t1)/maskLen) )
    return pmap

if __name__ == '__main__':
    # Let's set the view-parallel so we can see the top of the sphere
    p0 = np.pi-0.3
    # the view-meridian isn't so important, but if you wanted to rotate the 
    # view, this is how you do it.
    l0 = 0.0

    # Now create a Noise object and populate it with intelligent values. How to 
    # come up with 'intelligent' values is left as an exercise for the reader.
    gasy = fns.Noise(numWorkers=N_thread)
    gasy.frequency = 1.8
    gasy.axesScales = (1.0,0.06,0.06)

    gasy.fractal.octaves = 5
    gasy.fractal.lacunarity = 1.0
    gasy.fractal.gain = 0.33

    gasy.perturb.perturbType = fns.PerturbType.GradientFractal
    gasy.perturb.amp = 0.5
    gasy.perturb.frequency = 1.2
    gasy.perturb.octaves = 5
    gasy.perturb.lacunarity = 2.5
    gasy.perturb.gain = 0.5

    gasy_map = orthoProject(gasy, tile2=1024, p0=p0, l0=l0)

    t3 = perf_counter()
    # fig = plt.figure()
    # fig.patch.set_facecolor('black')
    # plt.imshow(gasy_map, cmap='inferno')
    # # plt.savefig('gasy_map.png', bbox_inches='tight', dpi=200)
    # plt.show(block=False)
    t4 = perf_counter()

    print( "Matplotlib showed plot in {:.3e} s".format(t4-t3))