import tensorflow as tf
import numpy as np
#You can edit from here
class ArrayAmpSlice:
    def __init__(self, its=400):
        self.iters = its
        self.showLossEvery = 20
        self.learningRate = 0.1
        self.normalizeOutputAmp = False
        self.slicePx = 256
        self.nMux = 1 #multiplexacion
        
    def outputField(cosAmps, phases, propagators, nMux):
        amps = tf.cos( cosAmps )
        emitters =  tf.complex(amps * tf.cos(phases), amps * tf.sin(phases) )
        
        if nMux == 1:
            field = emitters @ propagators
            ampField = tf.abs( field )
            return ampField
        else:
            nFieldPoints = propagators.shape[1]
            ampField = tf.zeros( [ nFieldPoints ] )
            for iField in range(nMux):
                field = emitters[iField:iField+1, :] @ propagators
                fieldAmp = tf.abs(field)
                ampField = ampField + fieldAmp
            return ampField / nMux
        
    
    def targetFunction(cosAmps, phases, propagators, targetFlat, normalizeOutput, nMux, slicePx):
        ampField = ArrayAmpSlice.outputField(cosAmps, phases, propagators, nMux)
        if normalizeOutput:
            maxAmp = tf.reduce_max( ampField )
            ampField = ampField / maxAmp
        loss = tf.reduce_mean(tf.square(ampField - targetFlat)) #mse   
        return loss
    
    def optimizeAmpSlice(self, propagators, targetAmpSlice, useAmpMod=True):
        propShape = propagators.shape
        nEmitters = propShape[0]
        nPoints = propShape[1]
        targetShape = targetAmpSlice.shape
        assert( nPoints ==  targetShape[0] * targetShape[1])
        targetFlat = tf.constant( np.reshape(targetAmpSlice, [nPoints]) )
        
        #initial amplitudes and phases
        if useAmpMod:
            cosAmps = tf.Variable( tf.random.uniform([self.nMux,nEmitters], minval=-np.pi, maxval=np.pi, dtype="float32"), trainable=True  )
        else:
            cosAmps = tf.Variable( tf.zeros([self.nMux,nEmitters], dtype="float32"), trainable=True )
        phases = tf.Variable( tf.random.uniform([self.nMux,nEmitters], minval=-np.pi, maxval=np.pi, dtype="float32"), trainable=True )
        
        propagatorsConst = tf.constant( propagators, dtype="complex64" )
    
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learningRate)
        for i in range(self.iters):
            with tf.GradientTape() as tape:
                loss = ArrayAmpSlice.targetFunction(cosAmps, phases, propagatorsConst, targetFlat, self.normalizeOutputAmp, self.nMux, self.slicePx)
                if useAmpMod:
                   grads = tape.gradient(loss, [cosAmps, phases])
                   optimizer.apply_gradients(zip(grads, [cosAmps, phases]))
                else:  
                    grads = tape.gradient(loss, [phases])
                    optimizer.apply_gradients(zip(grads, [phases]))
            if i+1 % self.showLossEvery == 0:
                print("Loss at iteration {}: {}".format(i, loss.numpy()))
                
                    
        ampOutput = ArrayAmpSlice.outputField(cosAmps, phases, propagatorsConst, self.nMux);
        return loss.numpy(), tf.cos( cosAmps ).numpy(), phases.numpy(), ampOutput
    