import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from Waves import Waves
from ImageUtils import ImageUtils
from ArrayAmpSlice import ArrayAmpSlice

#variables to test
path ="./datasets/self/upnalabIconsInv3"
amp=50
ampMod=False
nMux=4

#Simulation parameters
arraySize = 0.16
emitterApperture = 0.009
emittersPerSide = 16
distToOutput = 0.16
c = 343
fr = 40000
slicePx = 256

#derived variables
wavelength = c/fr 
k = 2*np.pi/wavelength
nEmitters = emittersPerSide * emittersPerSide

#flat array
emitterPositions = Waves.planeGridZ(0,0,0, arraySize, arraySize, emittersPerSide, emittersPerSide)
normals = Waves.constNormals(emitterPositions, [0,0,1])
outputPositions = Waves.planeGridZ(0,0,distToOutput, arraySize, arraySize, slicePx, slicePx)
#circular array
#emitterPositions = Waves.circleGrid(0,0,0, 0.1, 64)
#normals = Waves.pointToNormals(emitterPositions, [0,0,0])
#outputPositions = Waves.planeGridZ(0,0,distToOutput, 0.1, 0.1, slicePx, slicePx)

propagators = Waves.calcPropagatorsPistonsToPoints(emitterPositions, normals,outputPositions, k, emitterApperture)


opti = ArrayAmpSlice()
opti.showLossEvery = opti.iters+1 #do not show it
target = ImageUtils.loadNorm(path + ".png", slicePx)
opti.nMux = nMux
_, amps, phases, outputField = opti.optimizeAmpSlice(propagators, target * amp, ampMod)


virtualField =  tf.reshape( outputField, [slicePx,slicePx]).numpy()
    
composingFields = []
for i in range(nMux):
    currentCosAmps = tf.acos( amps[i:i+1,:] )
    compossingFieldFlat = ArrayAmpSlice.outputField(currentCosAmps, phases[i, :], propagators, 1)
    compossingField = tf.reshape( compossingFieldFlat , [slicePx,slicePx]).numpy()
    composingFields.append( compossingField )
    
    
#plot virtual field
plt.imshow(virtualField, cmap = 'gist_heat')
plt.show()

#plot compossing fields
for i in range(nMux):
    plt.imshow(composingFields[i], cmap = 'gist_heat')
    plt.show()