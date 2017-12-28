import pyfastnoisesimd.extension as ext
import concurrent.futures as cf
import numpy as np

#_DEFAULT_NOISE = 'Simplex'
#_DEFAULT_FRACTAL = 'FBM'
#_DEFAULT_PERTURB = None
#_DEFAULT_CELL_DISTANCE = 'Euclidean'
#_DEFAULT_CELL_RET = 'Distance'

# Let's see if we need a pool of _factories or not?
_factory = ext.FNS()

# Maybe have more specialized factory functions?
_asyncExecutor = cf.ThreadPoolExecutor( max_workers = 1 )
def setNumWorkers( N_workers ):
    """
    setNumWorkers( N_workers )

    Sets the maximum number of thread workers that will be used for generating
    noise.
    """
    N_workers = int( N_workers )
    if N_workers <= 0:
        raise ValueError("N_workers must be greater than 0")
    if _asyncExecutor._max_workers == N_workers:
        return

    _asyncExecutor._max_workers = N_workers
    _asyncExecutor._adjust_thread_count()


def generate( size=[1,1024,1024], start=[0,0,0],
              seed=42, freq=0.01, noiseType='Simplex', axisScales=[1.0,1.0,1.0], 
              fracType='FBM', fracOctaves=4, 
              fracLacunarity=2.0, fracGain=0.5, 
              cellReturnType='Distance', cellDistFunc='Euclidean',
              cellNoiseLookup='Simplex', cellNoiseLookupFreq=0.2, 
              cellDist2Ind=[0,1], cellJitter=0.2,
              perturbType=None, perturbAmp=1.0, perturbFreq=0.5, perturbOctaves=3,
              perturbLacunarity=2.0, perturbGain=0.5, perturbNormLen=1.0,   ):
    '''
    def generate( size=[1,1024,1024], start=[0,0,0], 
              seed=42, freq=0.01, noiseType='Simplex', axisScales=[1.0,1.0,1.0], 
              fracType='FBM', fracOctaves=4, 
              fracLacunarity=2.0, fractalGain=0.5, 
              cellReturnType='Distance', cellDistFunc='Euclidean',
              cellNoiseLookup='Simplex', cellNoiseLookupFreq='0.2', 
              cellDist2Ind=[0,1], cellJitter=0.2,
              perturbType=None, perturbAmp=1.0, perturbFreq=0.5, perturbOctaves=3,
              perturbLacunarity=2.0, perturbGain=0.5, perturbNormLen=1.0,   )
    '''
    _factory.SetSeed( 42 )
    _factory.SetFrequency( freq )
    _factory.SetNoiseType( ext.noiseType[noiseType] )
    _factory.SetAxisScales( axisScales[0], axisScales[1], axisScales[2] )
    _factory.SetFractalOctaves( fracOctaves )
    _factory.SetFractalLacunarity( fracLacunarity )
    _factory.SetFractalGain( fracGain )
    _factory.SetFractalType( ext.fractalType[fracType] )
    _factory.SetCellularReturnType( ext.cellularReturnType[cellReturnType] )
    _factory.SetCellularDistanceFunction( ext.cellularDistanceFunction[cellDistFunc]  )
    _factory.SetCellularNoiseLookupType( ext.noiseType[cellNoiseLookup] )
    _factory.SetCellularNoiseLookupFrequency( cellNoiseLookupFreq )
    _factory.SetCellularDistance2Indicies( *cellDist2Ind  )
    _factory.SetCellularJitter( cellJitter )
    _factory.SetPerturbType( ext.perturbType[perturbType] )
    _factory.SetPerturbAmp( perturbAmp )
    _factory.SetPerturbFrequency( perturbFreq )
    _factory.SetPerturbFractalOctaves( perturbOctaves )
    _factory.SetPerturbFractalLacunarity( perturbLacunarity )
    _factory.SetPerturbFractalGain( perturbGain )
    _factory.SetPerturbNormaliseLength( perturbNormLen )

    if _asyncExecutor._max_workers <= 1:
        # TODO: do we need scaleMod?  What's it do?
        return _factory.GetNoiseSet( *start, *size )
    # else run in threaded mode.
    # Create a full size empty array
    noise = ext.EmptySet( *size )
    # It would be nice to be able to split both on Z and Y if needed...
    if size[0] > 1:
        chunkAxis = 0
    elif size[1] > 1:
        chunkAxis = 1
    else:
        chunkAxis = 2

    numChunks = np.minimum( _asyncExecutor._max_workers, size[chunkAxis] )

    chunkedNoise = np.array_split( noise, numChunks, axis=chunkAxis )
    chunkIndices = [ start[0] for start in np.array_split( np.arange(size[chunkAxis]), numChunks )]

    workers = []
    for I, (chunk, chunkIndex) in enumerate( zip(chunkedNoise,chunkIndices) ):
        if chunk.size == 0: # Empty array indicates we have more threads than chunks
            continue
        
        workers.append( _asyncExecutor.submit( _chunked_gen, chunk, chunkIndex, chunkAxis ) )

    for peon in workers:
        peon.result()
    return noise

def _chunked_gen( chunk, chunkStart, chunkAxis ):
    pointer = chunk.__array_interface__['data'][0]
    # print( "pointer: {}, start: {}, axis{}".format(chunk, chunkStart, chunkAxis) )
    if chunkAxis == 0:
        _factory.FillNoiseSet( pointer, chunkStart, 0, 0, *chunk.shape )
    elif chunkAxis == 1:
        _factory.FillNoiseSet( pointer, 0, chunkStart, 0, *chunk.shape )
    else:
        _factory.FillNoiseSet( pointer, 0, 0, chunkStart, *chunk.shape )