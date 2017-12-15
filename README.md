# PyFastNoiseSIMD

PyFastNoiseSIMD is a wrapper around Jordan Peck's synthetic noise library [FastNoise SIMD](https://github.com/Auburns/FastNoise-SIMD) which has been accelerated with SIMD
instruction sets.  It may be installed via pip:

    pip install pyfastnoisesimd
    
Source and Windows Python 3.6 wheels are provided.

I have further accelerated it by multi-threading the generator.  The number of 
threads used is set by (Defaults to the total number of virtual cores found on the 
system).

    pyfastnoisesimd.setNumWorkers( N_workers )

There is generator function exposed `pyfastnoisesimd.generate` which works as 
follows:

    import pyfastnoisesimd as fns
    N = [512,512,512]
    seed = np.random.randint(2**31)
    fns.setNumWorkers( fns.cpu_info['count'] )
    cellular = fns.generate( size=N, start=[0,0,0], 
              seed=seed, freq=0.005, noiseType='Cellular', axisScales=[N[-1]/N[0],1.0,1.0], 
              fracType='FBM', fracOctaves=4, 
              fracLacunarity=3.0, fracGain=0.5, 
              cellReturnType='Distance', cellDistFunc='Euclidean',
              cellNoiseLookup='Simplex', cellNoiseLookupFreq=0.2, 
              cellDist2Ind=[0,1], cellJitter=0.5,
              perturbType='Gradient', perturbAmp=1.0, perturbFreq=0.7, perturbOctaves=5,
              perturbLacunarity=2.0, perturbGain=0.5, perturbNormLen=1.0 )

The return is a `numpy.ndarray`.  A more extensive example is found in 
`example/test_fns.py`.

### Benckmark single-threaded (1 core, i5-3570K @ 3.5 GHz)
    array of shape [4,1024,1024]
    Computed 4194304 voxels cellular noise in 0.289 s
        69.0 ns/voxel
    Computed 4194304 voxels Perlin noise in 0.192 s
        45.7 ns/voxel

### Benchmark multi-threaded (4 cores, i5-3570K @ 3.5 GHz)
    array of shape [4,1024,1024]
    Computed 4194304 voxels cellular noise in 0.079 s
        18.8 ns/voxel
        365 % thread scaling
    Computed 4194304 voxels Perlin noise in 0.050 s
        11.9 ns/voxel
        384 % thread scaling

Valid strings for `noiseType` and `cellNoiseLookup`:
    [ 'Value', 'ValueFractal', 'Perlin', 'PerlinFractal', 'Simplex', 'SimplexFractal', 'WhiteNoise', 'Cellular', 
        'Cubic', 'CubicFractal' ]

Valid strings for `fractType`:
    ['FBM', 'Billow', 'RigidMulti']

Valid strings for `perturbType`:
    ['Gradient','GradientFractal', 'Normalise', 'Gradient_Normalise', 'GradientFractal_Normalise']

Valid strings for `cellReturnType`:
    ['CellValue', 'Distance', 'Distance2', 'Distance2Add', 'Distance2Sub', 'Distance2Mul', 'Distance2Div', 'NoiseLookup', 
        'Distance2Cave' ]

Valid strings for `callDistFunc`:
    ['Euclidean', 'Manhattan', 'Natural']

If you want a more direct interface with the underlying library you may use the
`pyfastsimd._ext` module, which is a function-for-function mapping to the C++ 
code.

## Release Notes

### 0.1.5

* Using all lower-case directories for *nix.

### 0.1.4

* Fixed bug on multithreading; current approach splits arrays up to min(threads, array.shape[0])

### 0.1.2

* Added MANIFEST.in file for source distribution on PyPI



# C-Interface 

See below:

# FastNoise SIMD
FastNoise SIMD is the SIMD implementation of my noise library [FastNoise](https://github.com/Auburns/FastNoise). It aims to provide faster performance through the use of intrinsic(SIMD) CPU functions. Vectorisation of the code allows noise functions to process data in sets of 4/8/16 increasing performance by 700% in some cases (Simplex).

After releasing FastNoise I got in contact with the author of [FastNoise SIMD](https://github.com/jackmott/FastNoise-SIMD) (naming is coincidence) and was inspired to work with SIMD functions myself. Through his code and discussions with him I created my implementation with even more optimisation thanks to the removal of lookup tables. 

Runtime detection of highest supported instruction set ensures the fastest possible performance with only 1 compile needed. If no support is found it will fallback to standard types (float/int).

## Features

- Value Noise 3D
- Perlin Noise 3D
- Simplex Noise 3D
- Cubic Noise 3D
- Multiple fractal options for all of the above
- White Noise 3D
- Cellular Noise 3D
- Perturb input coordinates in 3D space
- Integrated up-sampling
- Easy to use 3D cave noise

Credit to [CubicNoise](https://github.com/jobtalle/CubicNoise) for the cubic noise algorithm

## Supported Instruction Sets
- ARM NEON
- AVX512
- AVX2 - FMA3
- SSE4.1
- SSE2

## Tested Compilers
- MSVC v120/v140
- Intel 16.0
- GCC 4.7 Linux
- Clang MacOSX

## Wiki
[Docs](https://github.com/Auburns/FastNoiseSIMD/wiki)

# FastNoise SIMD Preview

I have written a compact testing application for all the features included in FastNoiseSIMD with a visual representation. I use this for development purposes and testing noise settings used in terrain generation. The fastest supported instruction set is also reported.

Download links can be found in the [Releases Section](https://github.com/Auburns/FastNoiseSIMD/releases).

![Simplex Fractal](http://i.imgur.com/45JkT5j.png)

# Performance Comparisons
Using default noise settings on FastNoise SIMD and matching those settings across the other libraries where possible.

Timings below are x1000 ns to generate 32x32x32 points of noise on a single thread.

- CPU: Intel Xeon Skylake @ 2.0Ghz
- Compiler: Intel 17.0 x64

| Noise Type  | AVX512 | AVX2 | SSE4.1 | SSE2 | FastNoise | LibNoise |
|-------------|--------|------|--------|------|-----------|----------|
| White Noise | 7      | 9    | 16     | 29   | 141       |          |
| Value       | 92     | 152  | 324    | 436  | 642       |          |
| Perlin      | 147    | 324  | 592    | 795  | 1002      | 1368     |
| Simplex     | 129    | 294  | 548    | 604  | 1194      |          |
| Cellular    | 851    | 1283 | 2679   | 2959 | 2979      | 58125    |
| Cubic       | 615    | 952  | 1970   | 3516 | 2979      |          |

Comparision of fractals and sampling performance [here](https://github.com/Auburns/FastNoiseSIMD/wiki/In-depth-SIMD-level).

# Examples
### Cellular Noise
![Cellular Noise](http://i.imgur.com/RshUkoe.png)

![Cellular Noise](http://i.imgur.com/PjPYBXu.png)

![Cellular Noise](http://i.imgur.com/hyKjIuH.png)

[Cave noise example](https://www.youtube.com/watch?v=Df4Hidvq11M)

### Fractal Noise
![Simplex Fractal Billow](http://i.imgur.com/gURJtpc.png)

![Perlin Fractal Billow](http://i.imgur.com/IcjbpYz.png)

### Value Noise
![Value Noise](http://i.imgur.com/Ss22zRs.png)

### White Noise
![White Noise](http://i.imgur.com/wcTlyek.png)

### Perturb
![Perturbed Cellular Noise](http://i.imgur.com/xBKGo1E.png)

