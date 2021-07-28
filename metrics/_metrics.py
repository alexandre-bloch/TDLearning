# -*- coding: utf-8 -*-
"""
Created on Wed Jul 28 21:38:22 2021

@author: atteb
"""

'''
==============================================================================
                            Custom metrics
'''

import tensorflow as tf

def Temporal_MAE(y_true, y_pred,sample_weight=None):
    z_t = tf.roll( y_pred, axis = 1, shift = 1 )[:,1:]
    z_t_plusone = tf.roll( y_pred, axis = 1, shift = -1 )[:,:-1]
    return tf.reduce_mean(tf.abs(z_t-z_t_plusone))

def Prediction_MAE(y_true, y_pred,sample_weight=None):
    z_repeated = tf.expand_dims(y_true,1)
    return tf.reduce_mean(tf.abs(z_repeated-y_pred))
