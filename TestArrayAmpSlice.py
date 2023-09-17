import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from Waves import Waves
from ImageUtils import ImageUtils
from ArrayAmpSlice import ArrayAmpSlice
from datetime import datetime

#parameters from the model
arraySize = 0.16
emittersPerSide = 16
distToOutput = 0.16
c = 343
fr = 40000
slicePx = 256
targetPath = "./datasets/thickA.png"
targetAmplitude = 100
#derived variables
wavelength = c/fr 
k = 2*np.pi/wavelength
nEmitters = emittersPerSide * emittersPerSide

emittersPositions = Waves.planeGridZ(0,0,0, arraySize, arraySize, emittersPerSide, emittersPerSide)
outputPositions = Waves.planeGridZ(0,0,distToOutput, arraySize, arraySize, slicePx, slicePx)
#propagators = Waves.calcPropagatorsPointsToPoints(emittersPositions, outputPositions, k)
normals = Waves.constNormals(emittersPositions, [0,0,1])
propagators = Waves.calcPropagatorsPistonsToPoints(emittersPositions, normals,outputPositions, k,  0.01)


target = ImageUtils.loadNorm(targetPath, slicePx) * targetAmplitude
#target = ImageUtils.loadNorm(targetPath, slicePx)


opti = ArrayAmpSlice()
mse, amps, phases, field = opti.optimizeAmpSlice(propagators, target, True )
#opti.normalizeOutputAmp = True

timestamp = datetime.now().strftime("%Y.%m.%d.%H.%M.%S")
#plot amplitude field
outputAmp= tf.reshape( field, [slicePx,slicePx])
plt.imshow(outputAmp, cmap = 'gist_heat')
plt.colorbar()
#plt.savefig("./PDF_figures/"+timestamp+"_amplitudeField.pdf", format="pdf", bbox_inches="tight")
plt.show()

#histogram of the amplitudes
plt.hist( tf.reshape( amps , [nEmitters]), bins=25) 
#plt.savefig("./PDF_figures/"+timestamp+"_amplitudesHistogram.pdf", format="pdf", bbox_inches="tight")
plt.show()
#histogram of phases
plt.hist( tf.reshape( phases, [nEmitters] ), bins=25 )
#plt.savefig("./PDF_figures/"+timestamp+"_phasesHistogram.pdf", format="pdf", bbox_inches="tight")
plt.show()