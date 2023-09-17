import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from Waves import Waves
from ImageUtils import ImageUtils
from ArrayAmpSlice import ArrayAmpSlice

cPath="./datasets/"

#variables to test
targets=("thickA","thinA","pi","star","domino","helmet")
ampMods=(True,False)

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
#outputPositions = Waves.planeGridZ(0,0,0, 0.1, 0.1, slicePx, slicePx)


#propagators = Waves.calcPropagatorsPointsToPoints(emittersPositions, outputPositions, k)
propagators = Waves.calcPropagatorsPistonsToPoints(emitterPositions, normals,outputPositions, k, emitterApperture)


avgAmp = {}
stdAmp = {}
fields = {}

opti = ArrayAmpSlice()
opti.normalizeOutputAmp = True
opti.showLossEvery = opti.iters+1 #do not show it

for path,ampMod in [(path,ampMod) for path in targets for ampMod in ampMods]:
    print ("calc for", path, ampMod)
    target = ImageUtils.loadNorm(cPath + path + ".png", slicePx)
    _, _, _, flatOutput = opti.optimizeAmpSlice(propagators, target, ampMod)
    fields[path,ampMod] = flatOutput
    avgAmp[path,ampMod] = tf.reduce_mean( flatOutput ).numpy()
    stdAmp[path,ampMod] = tf.math.reduce_std( flatOutput ).numpy()


# plot the amps
avgAmpPhase = [avgAmp[key] for key in avgAmp.keys() if key[1] == True]
avgPhase = [avgAmp[key] for key in avgAmp.keys() if key[1] == False]
avgAmpPhaseStd = [stdAmp[key] for key in stdAmp.keys() if key[1] == True]
avgPhaseStd = [stdAmp[key] for key in stdAmp.keys() if key[1] == False]
indices = np.arange(len(targets))  # the label locations
barWidth = 0.20
fig, ax = plt.subplots(figsize=(7, 5))
rects1 = ax.bar(indices - barWidth/2, avgAmpPhase, barWidth, yerr=avgAmpPhaseStd , label='Amp&Phase')
rects2 = ax.bar(indices + barWidth/2, avgPhase, barWidth, yerr=avgPhaseStd, label='Only Phase')
ax.set_ylabel('Avg Amp')
ax.set_xticks(indices)
ax.set_xticklabels(targets)
ax.legend()
fig.tight_layout()
plt.show()

#see the fields
for path,ampMod in [(path,ampMod) for path in targets for ampMod in ampMods]:
    outputAmp= tf.reshape( fields[path, ampMod], [slicePx,slicePx])
    plt.imshow(outputAmp, cmap = 'gist_heat')
    plt.colorbar()
    plt.title( path + " " + ("Amp&Phase" if ampMod else "Only Phase") )
    plt.show()
    
#amplitude histograms
for path,ampMod in [(path,ampMod) for path in targets for ampMod in ampMods]:
    plt.hist( fields[path, ampMod][0,:], bins=25 )
    plt.title( path + " " + ("Amp&Phase" if ampMod else "Only Phase") )
    plt.show()