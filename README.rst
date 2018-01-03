PyFastNoiseSIMD
===============

PyFastNoiseSIMD is a wrapper around Jordan Peck's synthetic noise library 
https://github.com/Auburns/FastNoise-SIMD which has been 
accelerated with SIMD instruction sets. It may be installed via pip:

    pip install pyfastnoisesimd
    
Source and Windows Python 3.6 wheels are provided.

Here is a simple example to generate Perlin-style noise on a 3D rectilinear 
grid::

    import pyfastnoisesimd as fns
    import numpy as np
    shape = [512,512,512]
    seed = np.random.randint(2**31)
    N_threads = 4

    perlin = fns.Noise(seed=seed, numWorkers=N_threads)
    perlin.frequency = 0.02
    perlin.noiseType = fns.NoiseType.Perlin
    perlin.fractal.octaves = 4
    perlin.fractal.lacunarity = 2.1
    perlin.fractal.gain = 0.45
    perlin.perturb.perturbType = fns.PerturbType.NoPertrub
    result = perlin.genAsGrid(shape)

where ``result`` is a 3D ``numpy.ndarray`` of dtype ``'float32'``. Alternatively, 
the user can provide coordinates, which is helpful for tasks such as 
custom bump-mapping a tessellated surface, via ``Noise.getFromCoords(coords)``. 

More extensive examples are found in the ``examples`` folder on the Github repository.

Parallelism is further enhanced by the use of ``concurrent.futures`` to multi-thread
the generation of noise for large arrays. Thread scaling is generally in the 
range of 50-90 %, depending largely on the vectorized instruction set used. 
The number of threads, defaults to the number of virtual cores on the system. The 
ideal number of threads is typically the number of physical cores, irrespective 
of Intel Hyperthreading®.


Benchmarks
---------- 

The first test is used the default mode, a cubic grid, ``Noise.genAsGrid()``.

**Array shape**: [8,1024,1024]
**CPU**: Intel i7-7820X Skylake-X (8 cores, 3.6 GHz), Windows 7
**SIMD level supported**: AVX2 & FMA3

Single-threaded mode
~~~~~~~~~~~~~~~~~~~
Computed 8388608 voxels cellular noise in 0.298 s
    35.5 ns/voxel
Computed 8388608 voxels Perlin noise in 0.054 s
    6.4 ns/voxel


Multi-threaded (8 threads) mode
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Computed 8388608 voxels cellular noise in 0.044 s
    5.2 ns/voxel
    685.0 % thread scaling
Computed 8388608 voxels Perlin noise in 0.013 s
    1.5 ns/voxel
    431.3 % thread scaling


The alternative mode is ``Noise.getFromCoords()`` where the user provides the 
coordinates in Cartesian-space:

Single-threaded mode
~~~~~~~~~~~~~~~~~~~
* Generated noise from 2666000 coordinates with 1 workers in 1.935e-02 s (7.3 ns/pixel)

Multi-threaded (8 threads) mode
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Not implemented at present

    
Release Notes
-------------

**0.2.2**

* Added `orthographic_projection.py` to `examples/`.
* Updated doc-strings to accommodate `sphinx.napoleon` formatting.
* Added Sphinx-docs in the `doc` directory.
* Corrected spelling error `PerturbType.NoPetrub` -> `PerturbType.NoPerturb`
* Stopped `fastnoisesimd` from freeing memory for `coords` argument of 
  `Noise.genFromCoords(coords)`.  It should now be possible to reuse 
  coords without seg-faulting.

**0.2.1**

* Drop explicit Python 3.4 support as we cannot test it for Windows on MSVC2010
  and in any case it wouldn't have AVX2 instruction support.
* Start tagging, see `RELEASING_GUIDE.txt` for notes.

**0.2.0**

* Added the capability to provide coordinates 
* Added ``examples/projection.py`` to demonstrate noise generation by supplied 
  coordinates as applied to a Gall-Peters cylindrical projection of a sphere 
  (i.e. a world map).
* Added ``Noise`` object-oriented interface.  ``Noise`` uses Python properties to 
  expose the ``Set/Get`` functions in ``FastNoiseSIMD``.
* Added ``unittest`` support.
* Deprecated 'kitchen sink' ``pyfastnoisesimd.generate()`` function.
* Changed README from markdown to rich-structured text.
* Fixed a bug in the deprecated ``pyfastnoisesimd.generate()`` that always set 
  the seed to 42.
* Fixed spelling errors: ``axisScales`` -> ``axesScales``, ``indicies`` -> ``indices``

**0.1.5**

* Using all lower-case directories for *nix.

**0.1.4**

* Fixed bug on multithreading; current approach splits arrays up to min(threads, array.shape[0])

**0.1.2**

* Added MANIFEST.in file for source distribution on PyPI


FastNoiseSIMD library
---------------------

If you want a more direct interface with the underlying library you may use the 
``pyfastsimd._ext`` module, which is a function-for-function mapping to the C++ 
code.

FastNoiseSIMD is implemented by Jordan Peck, and may be found at: 

https://github.com/Auburns/FastNoiseSIMD

It aims to provide faster performance through the use of intrinsic(SIMD) CPU 
functions. Vectorisation of the code allows noise functions to process data in 
sets of 4/8/16 increasing performance by 700% in some cases (Simplex).

See the Wiki for usage information on the noise types:

https://github.com/Auburns/FastNoiseSIMD/wiki

Download links for a GUI-based reference noise generator may be found at:

https://github.com/Auburns/FastNoiseSIMD/releases

