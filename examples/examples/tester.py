# -*- coding: utf-8 -*-
"""
Created on Wed Jul 28 21:41:07 2021

@author: atteb
"""
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from TDLearning import TD_net, metrics

#%%
''' make walks '''
sample_size = 500
walk_steps = 10

s0 = 100
mu = 0.1
sigma = 0.2
T = 30/255

# make some random walks
dt = T * np.concatenate( [[0], np.ones(walk_steps)] )[None] 
drift = ( mu - 0.5 * sigma**2 ) * dt
noise = sigma * np.sqrt(dt) * np.random.normal(0,1, (sample_size,walk_steps+1))
increments = np.exp( drift + noise ) 
walks = s0 * np.cumprod( increments, 1 )

# define payoff
histogram_range = tuple(f(walks[...,-1]) for f in (min,max))
histogram_bins = 20  
hist = lambda value: np.histogram(value, 
                                  bins = histogram_bins, 
                                  range = histogram_range)[0] 
payoffs = np.array( [*map( hist, walks[...,-1] )] )

# add a dim for time
time_dim = np.cumsum( np.ones_like(walks), axis = 1 )
walks_with_time = np.stack([ walks,time_dim ],-1 )


#%% 
''' meet the TDBP '''

# teach the great TD
nodes_per_layer = 64
N_hidden_layers = 10
hidden_activation = tf.keras.activations.swish
output_activation = tf.keras.activations.swish

# TD MODEL
td_model = TD_net(

    #hidden layers
    nodes_per_layer = nodes_per_layer,
    N_hidden_layers = N_hidden_layers,
    hidden_activation = hidden_activation,
    
    #output
    output_shape = (histogram_bins,),          
    output_activation = output_activation,
    row_wise_output = False
)


td_model.compile( 
    optimizer = tf.keras.optimizers.SGD(), 
    metrics = [ metrics.Temporal_MAE, metrics.Prediction_MAE ]
)

#%%
''' train the TDBP '''

for lr in [.05,.01,.005,.001]:
    td_model.fit(
        walks_with_time, 
        payoffs, 
        epochs = 15,
        lr = lr,
        lamb = 0.1,
        batch_size = 1,
    )
    
#%%
''' plot preds '''

def plot_pred( t, spot, ax ):
    
    if ax is None:
        _,ax = plt.subplots()
        
    pred = td_model.predict( [[spot,walk_steps*t/T]]).numpy()
    
    ax.plot(
        np.linspace( *histogram_range, histogram_bins ), 
        np.squeeze( pred ), 
        label = "E[Price at t=%0.2f] | t=%0.2f, s_t = %0.2f" % (T, t, spot) 
    )
    
    ax.legend(loc="best",fontsize=10)
    
    return ax

ax = None
ax = plot_pred(0 /255, 100, ax)
ax = plot_pred(30 /255, 100, ax)

