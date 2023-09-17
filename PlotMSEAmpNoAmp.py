import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from Waves import Waves
from ImageUtils import ImageUtils
from ArrayAmpSlice import ArrayAmpSlice

cPath="./datasets/"

#variables to test
targets=("thinA","pi","star","domino","helmet")
amps=range(10,150,10)
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
#emitterPositions = Waves.planeGridZ(0,0,0, arraySize, arraySize, emittersPerSide, emittersPerSide)
#normals = Waves.constNormals(emitterPositions, [0,0,1])
#outputPositions = Waves.planeGridZ(0,0,distToOutput, arraySize, arraySize, slicePx, slicePx)

#circular array
emitterPositions = Waves.circleGrid(0,0,0, 0.1, 64)
normals = Waves.pointToNormals(emitterPositions, [0,0,0])
outputPositions = Waves.planeGridZ(0,0,distToOutput, 0.1, 0.1, slicePx, slicePx)


propagators = Waves.calcPropagatorsPistonsToPoints(emitterPositions, normals,outputPositions, k, emitterApperture)


MSEs = {}
avgAmp = {}
stdAmp = {}

opti = ArrayAmpSlice()
opti.showLossEvery = opti.iters+1 #do not show it

for path,amp,ampMod in [(path,amp,ampMod) for path in targets for amp in amps for ampMod in ampMods]:
    print ("calc for", path, amp, ampMod)
    target = ImageUtils.loadNorm(cPath + path + ".png", slicePx)
    mse, _, _, flatOutput = opti.optimizeAmpSlice(propagators, target * amp, ampMod)
    div = slicePx*slicePx * amp*amp
    
    MSEs[path,amp,ampMod] = mse / div
    avgAmp[path,amp,ampMod] = tf.reduce_mean( flatOutput )
    stdAmp[path,amp,ampMod] = tf.math.reduce_std( flatOutput )

#plot all results
for path in targets:
    withAmp = [MSEs[key] for key in MSEs.keys() if key[0] == path and key[2] == True]
    noAmp   = [MSEs[key] for key in MSEs.keys() if key[0] == path and key[2] == False]
    fig, ax = plt.subplots()
    ax.plot(amps, withAmp, color='blue', linestyle='--', label='amp&phase')
    ax.plot(amps, noAmp, color='blue', linestyle='-', label='onlyPhase')
    ax.legend()
    ax.set_xlabel('target amplitude')
    ax.set_ylabel('nMSE')
    ax.set_title(path)
    plt.show()