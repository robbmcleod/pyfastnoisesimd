Overview of PyFastNoiseSIMD
===========================

PyFastNoiseSIMD (`pyfastnoisesimd`) is a Python wrapper around Jordan Peck's 
`FastNoiseSIMD` (https://github.com/Auburns/FastNoise-SIMD) synthetic noise 
generation library.  `FastNoiseSIMD` 

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

Benchmarks
----------

The combination of the optimized, SIMD-instruction level C library, and 
multi-threading, means that `pyfastnoisesimd` is very, very fast.

Configuration
~~~~~~~~~~~~~

- **CPU**: Intel i7-7820X Skylake-X (8 cores, 3.6 GHz), Windows 7
- **SIMD level supported**: AVX2 & FMA3

With ``Noise.genAsGrid()``
~~~~~~~~~~~~~~~~~~~~~~~~~~

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

- ``noiseType = Simplex``
- ``peturbType = GradientFractal``

**Single-threaded mode**

Generated noise from 2666000 coordinates with 1 workers in 0.01935 s (7.3 ns/pixel)

**Multi-threaded (8 threads) mode**

Not implemented at present





