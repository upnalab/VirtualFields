import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from Waves import Waves
from ImageUtils import ImageUtils
from ArrayAmpSlice import ArrayAmpSlice

cPath="./datasets/"

#variables to test
#targets=("misc/6letters", "misc/12letters", "misc/ABCD")
targets=["self/upnalabIconsInv3"]
amps=[25]
ampMods=[True]
nMuxes=[1,2,8]

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


MSEs = {}
outputFields = {}

opti = ArrayAmpSlice()
opti.showLossEvery = opti.iters+1 #do not show it

for path,amp,ampMod,nMux in [(path,amp,ampMod, nMux) for path in targets for amp in amps for ampMod in ampMods for nMux in nMuxes]:
    print ("calc for", path, amp, ampMod, nMux)
    target = ImageUtils.loadNorm(cPath + path + ".png", slicePx)
    opti.nMux = nMux
    mse, _, _, outputField = opti.optimizeAmpSlice(propagators, target * amp, ampMod)
    div = slicePx*slicePx * amp*amp
    MSEs[path,amp,ampMod,nMux] = mse / div
    outputFields[path,amp,ampMod,nMux] =  tf.reshape( outputField, [slicePx,slicePx]).numpy()

colors = ["black","blue", "red", "green", "orange"]
#plot all results
for path in targets:
    fig, ax = plt.subplots()
    for nMux in nMuxes:
        withAmp = [MSEs[key] for key in MSEs.keys() if key[0] == path  and key[3] == nMux and key[2] == True]
        noAmp   = [MSEs[key] for key in MSEs.keys() if key[0] == path  and key[3] == nMux and key[2] == False]
        ax.plot(amps, withAmp, color=colors[nMux], linestyle='--', label='amp&phase mux' + str(nMux))
        ax.plot(amps, noAmp, color=colors[nMux], linestyle='-', label='onlyPhase mux' + str(nMux))
    ax.legend()
    ax.set_xlabel('target amplitude')
    ax.set_ylabel('nMSE')
    ax.set_title(path)
    plt.show()
    

#plot fields comparing nMuxes
if False: 
    target = targets[0]
    ampMod = True
    for amp in amps:
        f, axarr = plt.subplots(2,2)
        axarr[0,0].imshow( outputFields[target, amp, ampMod, nMuxes[0]] , cmap = 'gist_heat')
        axarr[0,1].imshow( outputFields[target, amp, ampMod, nMuxes[1]] , cmap = 'gist_heat')
        axarr[1,0].imshow( outputFields[target, amp, ampMod, nMuxes[2]] , cmap = 'gist_heat')
        axarr[1,1].imshow( outputFields[target, amp, ampMod, nMuxes[3]] , cmap = 'gist_heat')
        plt.show()
        input("Press Enter to continue...") 


        