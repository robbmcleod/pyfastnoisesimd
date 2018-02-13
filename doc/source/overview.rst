Overview of PyFastNoiseSIMD
===========================

PyFastNoiseSIMD (`pyfastnoisesimd`) is a Python wrapper around Jordan Peck's 
`FastNoiseSIMD` (https://github.com/Auburns/FastNoise-SIMD) synthetic noise 
generation library, `FastNoiseSIMD`. 

`pyfastnoisesimd` can generate noise in a 1-3D grid, via the `Noise.genAsGrid()` 
or the user can provide arbitrary coordinates in 3D Cartesian space with 
`Noise.genFromCoords()`

`FastNoiseSIMD` is also extremely fast due to its use of advanced x64 SIMD 
vectorized instruction sets, including SSE4.1, AVX2, and AVX512, depending 
on your CPU capabilities and the compiler used.  

Parallelism in `pyfastnoisesimd` is further enhanced by the use of 
``concurrent.futures`` to multi-thread the generation of noise for large arrays. 
Thread scaling is generally in the range of 50-90 %, depending largely on the 
vectorized instruction set used. The number of threads, defaults to the number 
of virtual cores on the system. The ideal number of threads is typically the 
number of physical cores, irrespective of Intel Hyperthreading®.

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

The combination of the optimized, SIMD-instruction level C library, and 
multi-threading, means that `pyfastnoisesimd` is very, very fast. Generally 
speaking thread scaling is higher on machines with SSE4 support only, 
as most CPUs throttle clock speed down to limit heat generation with AVX2. 
As such, AVX2 is only about 1.5x faster than SSE4 whereas on a pure SIMD 
instruction length basis (4 versus 8) you would expect it to be x2 faster.

Configuration
~~~~~~~~~~~~~

- **CPU**: Intel i7-7820X Skylake-X (8 cores, 3.6 GHz), Windows 7
- **SIMD level supported**: AVX2 & FMA3

With ``Noise.genAsGrid()``
~~~~~~~~~~~~~~~~~~~~~~~~~~

The first test is used the default mode, a cubic grid, ``Noise.genAsGrid()``, 
from ``examples\gridded_noise.py``:

- **Array shape**: [8,1024,1024]

**Single-threaded mode**

Computed 8388608 voxels cellular noise in 0.298 s
    35.5 ns/voxel
Computed 8388608 voxels Perlin noise in 0.054 s
    6.4 ns/voxel

**Multi-threaded (8 threads) mode**

Computed 8388608 voxels cellular noise in 0.044 s
    5.2 ns/voxel
    685.0 % thread scaling
Computed 8388608 voxels Perlin noise in 0.013 s
    1.5 ns/voxel
    431.3 % thread scaling

With ``Noise.getFromCoords()``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The alternative mode is ``Noise.getFromCoords()`` where the user provides the 
coordinates in Cartesian-space, from ``examples\GallPeters_projection.py``:
- ``noiseType = Simplex``
- ``peturbType = GradientFractal``

**Single-threaded mode**
Generated noise from 2666000 coordinates with 1 workers in 1.766e-02 s
    6.6 ns/pixel

**Multi-threaded (4 threads) mode**
Generated noise from 2666000 coordinates with 4 workers in 6.161e-03 s
    2.3 ns/pixel
    286.6 % thread scaling





