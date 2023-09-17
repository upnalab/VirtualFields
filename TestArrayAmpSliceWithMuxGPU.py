import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from Waves import Waves
from ImageUtils import ImageUtils
from ArrayAmpSlice import ArrayAmpSlice
import time

cPath="./datasets/"

# Variables to test
targets = ("thickA", "star")
ampMods = (True,False)
nMuxes = (2,4)
emitters = (30,60)
devices = ('/gpu:0','/device:CPU:0')

# Simulation parameters
arraySize = 0.16
distToOutput = 0.16
c = 343
fr = 40000
slicePx = 256
print(tf.config.list_physical_devices())
#print(tf.test.is_gpu_available())
#derived variables
wavelength = arraySize/30
k = 2*np.pi/wavelength

for device in devices:
    with tf.device(device):
        for path,ampMod,nMux,emittersPerSide in [(path,ampMod,nMux,emittersPerSide) for path in targets for ampMod in ampMods for nMux in nMuxes for emittersPerSide in emitters]:
            print("Starting on "+device+". path='"+path+"' ampMod:"+str(ampMod)+" nMux:"+str(nMux)+" emittersPerSide:"+str(emittersPerSide))
            t = time.time()
            nEmitters = emittersPerSide * emittersPerSide
            targetAmplitude = 40 * (nEmitters/(16*16))
            
            emittersPositions = Waves.planeGridZ(0,0,0, arraySize, arraySize, emittersPerSide, emittersPerSide)
            outputPositions = Waves.planeGridZ(0,0,distToOutput, arraySize, arraySize, slicePx, slicePx)
            normals = Waves.constNormals(emittersPositions, [0,0,1])
            propagators = Waves.calcPropagatorsPistonsToPoints(emittersPositions,normals,outputPositions,k,arraySize/emittersPerSide)
            
            targetPath= cPath + path + ".png"
            target = ImageUtils.loadNorm(targetPath, slicePx) * targetAmplitude
            
            opti = ArrayAmpSlice()
            opti.showLossEvery=100;
            
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
            #plt.imshow(outputAmp, cmap = 'gist_heat')
            #plt.colorbar()
            #plt.show()
            
            end = time.time()
            print("Finished on "+device+". path='"+path+"' ampMod:"+str(ampMod)+" nMux:"+str(nMux)+" emittersPerSide:"+str(emittersPerSide)+". [It took "+ str(end-t)+" s]")