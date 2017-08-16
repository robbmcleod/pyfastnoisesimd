import pyfastnoisesimd.extension as ext


#_DEFAULT_NOISE = 'Simplex'
#_DEFAULT_FRACTAL = 'FBM'
#_DEFAULT_PERTURB = None
#_DEFAULT_CELL_DISTANCE = 'Euclidean'
#_DEFAULT_CELL_RET = 'Distance'

# Instead of having a global here, we could create a pool and
_factory = ext.FNS()

# Maybe have more specialized factory functions?

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
    
    # TODO: do we need scaleMod?  What's it do?
    return _factory.GetNoiseSet( *start, *size )
    