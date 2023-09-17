import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from Waves import Waves
from ImageUtils import ImageUtils
from ArrayAmpSlice import ArrayAmpSlice
import time

start = time.time()

cPath="./datasets/self/"

# Variables to test
targets = ("sheet","upnalabInv","upnalab","upna")
ampMods = (True,False)
nMuxes = (1,2,4,6,8)
emitters = (30,60,90)

# Simulation parameters
arraySize = 0.16
distToOutput = 0.16
c = 343
fr = 40000
slicePx = 256

#derived variables
wavelength = arraySize/30
k = 2*np.pi/wavelength

for path,ampMod,nMux,emittersPerSide in [(path,ampMod,nMux,emittersPerSide) for path in targets for ampMod in ampMods for nMux in nMuxes for emittersPerSide in emitters]:
    # print(ampMod,nMux,emittersPerSide)
    nEmitters = emittersPerSide * emittersPerSide
    targetAmplitude = 25 * (nEmitters/(16*16))
    
    emittersPositions = Waves.planeGridZ(0,0,0, arraySize, arraySize, emittersPerSide, emittersPerSide)
    outputPositions = Waves.planeGridZ(0,0,distToOutput, arraySize, arraySize, slicePx, slicePx)
    normals = Waves.constNormals(emittersPositions, [0,0,1])
    propagators = Waves.calcPropagatorsPistonsToPoints(emittersPositions,normals,outputPositions,k,arraySize/emittersPerSide)
    
    targetPath= cPath + path + ".png"
    target = ImageUtils.loadNorm(targetPath, slicePx) * targetAmplitude
    
    opti = ArrayAmpSlice()
    
    # opti.nMux = 1
    # mse, amps, phases, field = opti.optimizeAmpSlice(propagators, target, modAmp )
    # #plot amplitude field
    # outputAmp= tf.reshape( field, [slicePx,slicePx])
    # plt.imshow(outputAmp, cmap = 'gist_heat')
    # plt.colorbar()
    # plt.show()
    
    opti.nMux = nMux
    mse, amps, phases, field = opti.optimizeAmpSlice(propagators, target, ampMod)
    #plot amplitude field
    outputAmp= tf.reshape( field, [slicePx,slicePx])
    graph_title = path + '_' + str(emittersPerSide) + 'x' + str(emittersPerSide) + '_mux' + str(nMux)
    if ampMod:
        graph_title = graph_title + '_ampmod'
    plt.imshow(outputAmp, cmap = 'gist_heat')
    plt.title(graph_title)
    plt.colorbar()
    plt.show()
    fig_path = './results/HD/' + graph_title + '.png'
    plt.savefig(graph_title)
    fig_path = './results/HD/' + graph_title + '.pdf'
    plt.savefig(graph_title)
    
    end = time.time()
    print(end-start)