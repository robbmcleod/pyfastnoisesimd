PyFastNoiseSIMD
===============

PyFastNoiseSIMD is a wrapper around Jordan Peck's synthetic noise library 
https://github.com/Auburns/FastNoise-SIMD which has been 
accelerated with SIMD instruction sets. It may be installed via pip:

    pip install pyfastnoisesimd
    
Parallelism is further enhanced by the use of ``concurrent.futures`` to multi-thread
the generation of noise for large arrays. Thread scaling is generally in the 
range of 50-90 %, depending largely on the vectorized instruction set used. 
The number of threads, defaults to the number of virtual cores on the system. The 
ideal number of threads is typically the number of physical cores, irrespective 
of Intel Hyperthreading®. 

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
    perlin.perturb.perturbType = fns.PerturbType.NoPerturb
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

Documentation
-------------

Check it out at:

http://pyfastnoisesimd.readthedocs.io

Installation
------------

`pyfastnoisesimd` is available on PyPI, and may be installed via `pip`::

    pip install --upgrade pip
    pip install --upgrade setuptools
    pip install -v pyfastnoisesimd

On Windows, a wheel is provided for Python 3.6 only. Building from source or 
compiling the extension for 3.5 will require either MS Visual Studio 2015 or 
MSVC2015 Build Tools:

http://landinghub.visualstudio.com/visual-cpp-build-tools

No Python versions compile with MSVC2017 yet, which is the newest version to 
support AVX512. Only Python 3.5/3.6 support AVX2 on Windows.

On Linux or OSX, only a source distribution is provided and installation 
requires `gcc` or `clang`. For AVX512 support with GCC, GCC7.2+ is required, lower 
versions will compile with AVX2/SSE4.1/SSE2 support only. GCC earlier than
4.7 disables AVX2 as well. Note that `pip` does not respect the `$CC` environment
variable, so to clone and build from source with `gcc-7`:

    git clone https://github.com/robbmcleod/pyfastnoisesimd.git
    alias gcc=gcc-7; alias g++=g++-7
    pip install -v ./pyfastnoisesimd

Installing GCC7.2 on Ubuntu (with `sudo` or as root)::

    add-apt-repository ppa:ubuntu-toolchain-r/test
    apt update
    apt install gcc-7 g++-7

Benchmarks
---------- 

Generally speaking thread scaling is higher on machines with SSE4 support only, 
as most CPUs throttle clock speed down to limit heat generation with AVX2. 
As such, AVX2 is only about 1.5x faster than SSE4 whereas on a pure SIMD 
instruction length basis (4 versus 8) you would expect it to be x2 faster.

The first test is used the default mode, a cubic grid, ``Noise.genAsGrid()``, 
from ``examples\gridded_noise.py``:

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
coordinates in Cartesian-space, from ``examples\GallPeters_projection.py``:

Single threaded mode
~~~~~~~~~~~~~~~~~~~~
Generated noise from 2666000 coordinates with 1 workers in 1.766e-02 s
    6.6 ns/pixel

Multi-threaded (4 threads) mode
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Generated noise from 2666000 coordinates with 4 workers in 6.161e-03 s
    2.3 ns/pixel
    286.6 % thread scaling
    
Release Notes
-------------
**0.4.2**

- 

**0.4.1**

* Support for Python 3.7 now official. On Windows AVX512 is still disabled as 
  even with MSVC2017.3 some of the required SIMD instructions are unavailable.

**0.4.0**

* Fixed aligned memory location on Windows and enabled multi-threaded processing 
  for both generators.
* renamed `emptyCoords` function to `empty_coords`.

**0.3.2**

* Disabled aligned memory allocation on Windows, due to it causing seg-faults.
* Thanks to Luke H-W for finding and fixing a memory leak in `genAsGrid`.
* Thanks to Enderlook for reporting that the `start` parameter was not working 
  in multi-threading mode for calls to `genAsGrid`.

**0.3.1**

* Changes to calling convention to avoid pointer size confusion between 64- and 
  32-bit OSs.

**0.3.0**

* Elliott Sales de Andrade fixed a number of issues with installation to 
  build cleanly and better handle CPU SIMD capabilities.
* Added multi-threaded operation to `Noise.genFromCoords()`.
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


Authors
-------

Robert A. McLeod wrote the Python wrapper, implemented multi-threading, and 
wrote the documentation.

Elliott Sales de Andrade contributed a number of fixes to allow building to 
succeed on many platforms.

Jordan Peck wrote the underlying library `FastNoiseSIMD`.
