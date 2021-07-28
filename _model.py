# -*- coding: utf-8 -*-
"""
Created on Sat Jan  9 22:25:58 2021

@author: atteb
"""

import tensorflow as tf
from tensorflow.keras.layers import Dense, Reshape
from tensorflow.keras.layers.experimental.preprocessing import Normalization

tf.keras.backend.set_floatx('float32')
tf.config.optimizer.set_jit(True)

'''
==============================================================================
                            The famous TDBP
            
            
                            
'''
class TD_net(tf.keras.Model):

    #TODO;
    #   get rid of gast crap in @tf.function
    
    def __init__(self,
                 output_shape : tuple,
                 nodes_per_layer : int = 32,
                 N_hidden_layers : int = 5,
                 hidden_activation = "relu",
                 output_activation = "linear",
                 row_wise_output = False ):
                        
        super().__init__()
        self.model = NeuralNetwork(output_shape,
                                   nodes_per_layer,
                                   N_hidden_layers,
                                   hidden_activation,
                                   output_activation,
                                   row_wise_output
                                   )
                        
    def fit(self,
            x,
            z,
            lr=None,
            lamb=0.3,
            batch_size=1,
            epochs=1,
            verbose=1,
            mask = None,

            #keras stuff
            callbacks=None,
            validation_split=0.,
            validation_data=None,
            shuffle=True,
            class_weight=None,
            initial_epoch=0,
            steps_per_epoch=None,
            validation_steps=None,
            validation_batch_size=None,
            validation_freq=1,
            max_queue_size=10,
            workers=1,
            use_multiprocessing=False):

        # Set hyperparameters lambda and alpha
        self.lamb = tf.convert_to_tensor([lamb],dtype=tf.float32)
        if lr is not None:
            tf.keras.backend.set_value( self.optimizer.learning_rate, lr )
        
        # Set axes by which the gradients are calculated in update_step
        x,z = [ tf.convert_to_tensor(v, dtype = tf.float32) for v in [x,z] ]
        while tf.rank(z) < 2:
            z = tf.expand_dims(z,-1)
        self.gradient_axes = [ [i for i in range(z.ndim)] ] * 2
        while tf.rank(x) < 3:
            x = tf.expand_dims(x,-1)
        
        # Get mean and var across flat time dimension
        s = tf.shape(x)
        self.model.normalization.adapt( tf.reshape( x, (s[0]*s[1],*s[2:]) ) )
        
        # Create mask for the inputs
        if mask is None:
            mask = tf.ones( (s[0],s[1]), dtype=tf.float32)
        mask=tf.convert_to_tensor(tf.cast(mask,tf.float32),dtype=tf.float32)
        assert all( tf.shape(mask).numpy() == (s[0],s[1]) ),\
            f"Expected mask with shape{(s[0],s[1])}, got {mask.shape}"
        
        return super().fit(x,z,
                           sample_weight = mask,
                           batch_size=batch_size,
                           epochs=epochs,
                           verbose=verbose,
                           callbacks=callbacks,
                           validation_split=validation_split,
                           validation_data=validation_data,
                           shuffle=shuffle,
                           class_weight=class_weight,
                           initial_epoch = initial_epoch,
                           steps_per_epoch = steps_per_epoch,
                           validation_steps = validation_steps,
                           validation_batch_size = validation_batch_size,
                           validation_freq = validation_freq,
                           max_queue_size=max_queue_size,
                           workers=workers, 
                           use_multiprocessing=use_multiprocessing 
                           )

    @tf.function
    def train_step(self, data):
        
        # Zero-initialize eligibility trace 
        trace = [ tf.zeros((1)) * layer for layer in self.trainable_variables ]
        
        # Unpack data
        x, z, mask = data
        x_by_t = tf.unstack( x, axis = 1 )
        mask_by_t = tf.unstack( mask, axis = 1 )
        
        # Temporal difference updates
        for x_t, x_tplusone, mask_t in zip(x_by_t[:-1],x_by_t[1:],mask_by_t):
            
            # Predict
            P_tplusone = self(x_tplusone,training = True)
                        
            # Update
            trace = self.update_step( x_t, P_tplusone, trace, mask_t )
            
            # patch to embed z 
            trace = self.update_step( x_tplusone, z, trace, mask_t ) 
            
            # Decay
            trace = [ tf.multiply(t, self.lamb) for t in trace ]
        
        # Final update
        self.update_step( x_tplusone, z, trace, mask_t )
                
        # Update and return metrics
        return self.test_step( data = (x,z) )
    
    @tf.function
    def update_step(self, predictor, target, trace, mask):
        
        train_vars = self.trainable_variables
        
        # Get gradients
        with tf.GradientTape() as g_tape:
            P_t = self(predictor,training = True)
        jacobian = g_tape.jacobian( P_t, train_vars )
        
        # Add gradients to eligibility trace
        trace = [ tf.add(t, j) for t, j in zip( trace, jacobian ) ]
                
        # Get prediction error
        d_P = tf.subtract( P_t, target )
        
        # Compute weight updates, null if mask is 0
        mask = tf.reduce_prod(mask)
        delta_w = [mask*tf.tensordot(t,d_P,self.gradient_axes) for t in trace]
        
        # Update weights
        self.optimizer.apply_gradients( zip( delta_w, train_vars ) )
        
        return trace
    
    def test_step(self,data):
        
        # Unpack data
        x, z = data
        z_hat = tf.map_fn( self, x )
        self.compiled_metrics.update_state( z, z_hat )       
        return {m.name: m.result() for m in self.metrics}

    def predict( self, x ):
        
        # Convert to rank two tensor
        x = tf.convert_to_tensor( x, dtype = tf.float32 )
        while tf.rank( x ) < 2:
            x = tf.expand_dims(x, 0)
            
        z_hat = self( x )
        return z_hat
            
    def call(self, x ):
        return self.model(x)

'''
==============================================================================
                        Neural network models
'''

class NeuralNetwork( tf.keras.layers.Layer ):
    
    def __init__(self, 
                 output_shape : tuple,
                 nodes_per_layer : int = 32,
                 N_hidden_layers : int = 5,
                 hidden_activation = "relu",
                 output_activation = "linear",
                 row_wise_output = False ):
        
        super().__init__()
        
        self.normalization = Normalization(axis=-1,dtype=tf.float32)
        
        self.hidden_layers = [Dense(nodes_per_layer,hidden_activation)
                              for __layer__ in range(N_hidden_layers)]
        
        if len(output_shape) == 0:
            self.out = Dense( 1, output_activation )
        
        elif row_wise_output:
            self.out = multiclass_output(output_shape,output_activation)
            
        else:
            self.out = matrix_output(output_shape,output_activation)
            
    def call(self,x):
        x = self.normalization(x)
        for layer in self.hidden_layers:
            x = layer(x)
        x = self.out(x)
        return x

class matrix_output( tf.keras.layers.Layer ):
    
    def __init__( self, shape, activation ):
        super().__init__()
        nodes = tf.reduce_prod(shape)
        self.flat_layer = Dense(nodes, activation)
        self.reshape = Reshape( shape )
    
    def call(self,x):
        x = self.flat_layer(x)
        x = self.reshape(x)
        return x
        
class multiclass_output( tf.keras.layers.Layer ):
    
    def __init__( self, shape, activation ):
        super().__init__()
        self.rows = [Dense(shape[0],activation)
                     for row in range( shape[1])]

    def call(self,x):
        x = tf.stack( [ row(x) for row in self.rows ], axis = 1 )
        return x

