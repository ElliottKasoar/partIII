#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 17 16:38:32 2019

@author: Elliott
"""

#DLLk set1: batch_size=256, epochs=50, lr=0.0001, b1=0.5, frac=0.05, noise=100 
#Works ok. Set1.1: epochs=250

#DLLk set2: epochs=200, frac=0.1, lr=0.0002, noise=150......  
#Worked ok. Runtime=40330s

#DLLk set3: try e=200, frac=0.1, lr=0.0001, noise=100

#DLLk set4: longer run e=500, f=0.5, lr=0.0001, noise=150, batch size=64.
#Runtime=411462.4. Didn't work particularly well

#DLLe set1: batch_size=128, epochs=100, lr=0.0001, b1=0.5, frac=0.05, noise=100 works ok
#Worked ok. Runtime 4946.165291547775

#DLLe set1: same as DLLk set1 (oops) i.e. batch_size=128, epochs=100, lr=0.0001, b1=0.5, frac=0.05, noise=100. But remove batch_norm and dropout
#Worked ok. Runtime=2821.4

#DLLk set5: same as DLLk set1 i.e. batch_size=256, epochs=100, lr=0.0001, b1=0.5, frac=0.05, noise=100. But remove batch_norm and dropout. 
#Runtime=1864.2. Looked weird but maybe improving. Trying again with epochs = 250. Time = 4995.1. Still double peak

#DLLk set6: same as DLLk set1 i.e. batch_size=256, epochs=250, lr=0.0001, b1=0.5, frac=0.05, noise=100. But remove dropout (batchnorm back in).
#Runtime=7962.4. Worked ok ish?

#DLLk set7: same as DLLk set1 i.e. batch_size=256, epochs=250, lr=0.0001, b1=0.5, frac=0.05, noise=100. But remove batchnorm (dropout back in).
#Runtime=5652.1. Doesn't really work (bimodal dist)

#DLLk set8: set1 but with gen layers added (6).
#Runtime=9969.6. Worked vaguely 

#DLLk set9: set1 but with gen and discrim layers added (6,6).
#Runtime=10845.8. Worked vaguely

#DLLk set10: set1 but with gen and discrim layers added (6,5).
#Runtime=10506.6. Worked vaguely

#DLLk set11: set1 but with gen and discrim layers added (6,5).
#Runtime=106634.2. set10 but epochs = 500, frac = 0.25. Worked ok

####################################

#GAN_7DLL set1: batch_size = 256, epochs = 500,  lr=0.0001, noise=100, frac=0.25, gen and discrim layers added (6,5), giving P, Pt
#(128, 256, 512, 1024, 2048, data_dim) and (2048, 1024, 512, 256, 1)
#Runtime=107595.3

#set2: Quicker version of set1. batch_size = 128, epochs = 250,  lr=0.0001, noise=100, frac=0.025, gen and discrim layers added (6,5), giving P, Pt
#(128, 256, 512, 1024, 2048, data_dim) and (2048, 1024, 512, 256, 1)
#Runtime=8430.7.

#set3: set2, but with gen and discrim layers (7,5)
#(256, 256, 512, 512, 1024, 1024, data_dim) and (2048, 1024, 512, 256, 1)
#Runtime=9201.5.

#set4: set2, but with gen and discrim layers (7,6)
#(256, 256, 512, 512, 1024, 1024, data_dim) and (2048, 1024, 512, 256, 128, 1)
#Runtime=9404.0

#set5: set2, but with gen and discrim layers (9,6) 
#(128, 128, 256, 256, 512, 512, 1024, 1024, data_dim) and (2048, 1024, 512, 256, 128, 1)
#Runtime=11032.0

#set6: set2, but with gen and discrim layers (15,8) 
#(128, 128, 256, 256, 512, 512, 1024, 1024, 512, 512, 256, 256, 128, 128, data_dim) and (128, 256, 512, 1024, 512, 256, 128, 1)
#Runtime=15725.8

#set7: set2, but with gen and discrim layers (11,6) 
#(256, 256, 512, 512, 1024, 1024, 512, 512, 256, 256, data_dim) and (256, 512, 1024, 512, 256, 1)
#Runtime=12426.3

#set8: set2, but with gen and discrim layers (11,11) 
#(256, 256, 512, 512, 1024, 1024, 512, 512, 256, 256, data_dim) and (256, 256, 512, 512, 1024, 1024, 512, 512, 256 256, 1)
#Runtime=13336.9

#set9: set2, but with gen and discrim layers (7,6). Generator given the P, Pt data too.
#(256, 256, 512, 512, 1024, 1024, data_dim) and (2048, 1024, 512, 256, 128, 1)
#Runtime=9103.3

#set10: set2, but with gen and discrim layers (7,6). Generator given the P, Pt data too. Fixed gen tests.
#(256, 256, 512, 512, 1024, 1024, data_dim) and (2048, 1024, 512, 256, 128, 1)
#Runtime=9064.2

#set11: set10, but epochs = 500, frac = 0.5

#set12: set10, but fixed generator training
#Runtime=9562.9. Didn't really work.

#set13: set12, but fixed generator training and layers (256x9, data_dim) and (256*9, 1)
#Runtime=12599.4. Worked well ish. 13.1: Same but with generator saved hopefully... (12812.9)

#set14: set13, but with frac = 0.1, epochs = 500
#Runtime=98277.9. Worked ok. Weird spike

#set15: set13, but with particle_source = 'PION'. Also added 6th DLL...
#Runtime=12634.4. 

#set16: set13, but with new gen/discrim training structure
#Runtime=3530.8.  Didn't really work

#set17: set 15 again but KAON again i.e. set13(.1) with all DLLs
#Runtime=12380.4

#set18: set17 but epochs = 500, frac=0.1. KAONS
#Runtime=63672.8

#set19: set17 but epochs = 500, frac=0.1. PIONS
#Runtime=63473.9

#set20: Aiming to change physics - input to both networks but don't generate. Epochs=100. Longer run see set22
#Runtime=5054.7

#set21: set20 but alt structure and epochs = 250
#Runtime=Aborted as wasn't working

#set22: set20 but epochs=500, frac=0.1. KAONS
#Runtime=98997.0

#set23: set22 but PIONS
#Runtime=97557.5

#set24: set23 (PIONS, 500, 0.1, 1024)
#Runtime=36778.9

#set25: set23 (PIONS, 500, 0.1, 32)
#Runtime=

#set26: set23 (PIONS, 500, 0.1, 4096)
#Runtime=29720.6

#set27: set23 (PIONS, 500, 0.1, 128) added NumPV and NumLongTracks to input data
#Runtime=98818.9

#set28: (KAONs, 500, 0.1, 128) with NumPV and NumLongTracks
#Runtime=98905.5

#set29: (KAONs, 250, 0.025, 128) with NumPV, NumLongTracks and new data files
#Runtime=12549.2

#set30: set23 (PIONS, 500, 0.1, 128), NumPV, NumLongTracks
#Runtime=98680.3

#set31: (KAONs, 500, 0.1, 128) with NumPV, NumLongTracks
#Runtime=96957.3

#set32: (KAONs, 500, 1, 128) with NumPV, NumLongTracks 
#Runtime=Aborted. Too long for now

#set33: (PIONS, 500, 1, 128) with NumPV, NumLongTracks
#Runtime=Aborted. Too long for now

#set34: (KAONs, 500, 0.1, 128) with NumPV, NumLongTracks and all pos data
#Runtime=96736.4

#set35: (PIONS, 500, 0.1, 128) with NumPV, NumLongTracks all pos data
#Runtime=96449.1

#set36: (KAONS, 100, 0.025, 128) with all data. Alt model
#Runtime = 4708.1

#set37: (KAONS, 500, 0.1, 128) with all data. Alt model
#Runtime = 

#set38: (KAONS, 100, 0.025, 128) with all data. Alt model
#Runtime = 4772.3

#set39: (KAONS, 100, 0.025, 128) with all data. Alt model_2 (Wasserstein)
#Runtime = 2762.3

#set40: (KAONS, 500, 0.1, 128) with all data. Alt model_2 (Wasserstein)
#Runtime = 



#Consider learning rate decay?

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

from keras.layers import Input, BatchNormalization, concatenate
from keras.models import Model, Sequential
from keras.layers.core import Dense, Dropout
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam
from keras import initializers

from keras.utils import multi_gpu_model
import keras.backend as K
from keras.layers.merge import _Merge
from functools import partial

# The training ratio is the number of discriminator updates per generator update. The paper uses 5.
training_ratio = 5
grad_penalty_weight = 10  # As per the paper

#from matplotlib.ticker import AutoMinorLocator
#from scipy.stats import gaussian_kde
#import math
#from sklearn.preprocessing import QuantileTransformer

#Time total run
t_init = time.time()

#Choose GPU to use
os.environ["CUDA_VISIBLE_DEVICES"]="2" 

#Using tensorflow backend
os.environ["KERAS_BACKEND"] = "tensorflow"

plt.rcParams['agg.path.chunksize'] = 10000 #Needed for plotting lots of data?

#Some tunable variables/parameters...
#Not really passed properly

#Training variables
batch_size = 128
epochs = 500

#Parameters for Adam optimiser
learning_rate = 0.0001
beta_1=0.5
beta_2=0.9

gen_input_dim = 100 #Dimension of random noise vector.

frac = 0.1
train_frac = 0.7

#DLL(DLL[i] - ref_particle) from particle_source data
DLLs = ['e', 'mu', 'k', 'p', 'd', 'bt']
physical_vars = ['TrackP', 'TrackPt', 'NumLongTracks', 'NumPVs', 'TrackVertexX', 'TrackVertexY', 'TrackVertexZ', 
                 'TrackRich1EntryX', 'TrackRich1EntryY', 'TrackRich1EntryZ', 'TrackRich1ExitX', 'TrackRich1ExitY', 
                 'TrackRich1ExitZ']

ref_particle = 'pi'
particle_source = 'KAON'

phys_dim = len(physical_vars)
DLLs_dim = len(DLLs)
data_dim = DLLs_dim + phys_dim
noise_dim = gen_input_dim - phys_dim

#Internal layers of generator and discriminator
gen_layers = 8
discrim_layers = 8

plot_freq = 10 #epochs//10 #Plot data for after this number of epochs

#So reproducable
np.random.seed(10)


#Import data via pandas from data files
def import_single_var(var_type, particle_source):
    
    if(particle_source == 'KAON'):
    
        datafile_kaon = '../../data/PID-train-data-KAONS.hdf'
        data_kaon = pd.read_hdf(datafile_kaon, 'KAONS')
        data_loc = data_kaon
        
    elif(particle_source == 'PION'):
    
        datafile_pion = '../../data/PID-train-data-PIONS.hdf' 
        data_pion = pd.read_hdf(datafile_pion, 'PIONS') 
        data_loc = data_pion

    else:
        print("Please select either kaon or pion as particle source")

    data = data_loc.loc[:, var_type]

    return data


def import_all_var(particle_source):
    
    #Import data from kaons and pions    
    if(particle_source == 'KAON'):        
        datafile = '../../data/PID-train-data-KAONS.hdf' 
    elif(particle_source == 'PION'):    
        datafile = '../../data/PID-train-data-PIONS.hdf' 
    else:
        print("Please select either kaon or pion as particle source")

    data = pd.read_hdf(datafile, particle_source + 'S') 

    return data


#Change DLLs e.g. from K-pi to p-K
def change_DLL(DLL1, DLL2):
    
    if(not np.array_equal(DLL1, DLL2)):
        DLL3 = np.subtract(DLL1, DLL2)
    else:
        print("DLLs are the same!")
        DLL3 = DLL1
    
    return DLL3


#Normalise data via dividing centre on zero and divide by max s.t. range=[-1,1]
def norm(x):
    
    shift = np.zeros(x.shape[1])
    div_num = np.zeros(x.shape[1])
    
    for i in range(x.shape[1]):
        
        x_max = np.max(x[:,i])
        x_min = np.min(x[:,i])
    
        shift[i] = (x_max + x_min)/2
        x[:,i] = np.subtract(x[:,i], shift[i])
        
        if x_max == x_min:
            div_num[i] = 1
        else:
                div_num[i] = x_max - shift[i]
                x[:,i] = np.divide(x[:,i], div_num[i])
    
    return x, shift, div_num


#Get training/test data and normalise
def get_x_data(DLLs, ref_particle, physical_vars, particle_source):
    
    all_data = import_all_var(particle_source)
        
    #Get first set of DLL data
    DLL_data_1 = np.array(all_data.loc[:, 'RichDLL' + DLLs[0]])
                
    x_data_dim = (DLL_data_1.shape[0], DLLs_dim + phys_dim) 
    x_data = np.zeros((x_data_dim))
    x_data[:,0] = DLL_data_1
    
    #Get other DLL data
    for i in range(1, DLLs_dim):    
        x_data[:,i] = np.array(all_data.loc[:, 'RichDLL' + DLLs[i]])
    
    #Get physics data
    for i in range(DLLs_dim, DLLs_dim + phys_dim):
        phys_vars_index = i - DLLs_dim
        x_data[:,i] = np.array(all_data.loc[:, physical_vars[phys_vars_index]])
    
    #(Shuffle) and normalise data by shifting and dividing s.t. lies between -1 and 1
#    np.random.shuffle(x_data)
    x_data, shift, div_num = norm(x_data)
    
    #Use subset of data
    tot_split = int(frac * x_data.shape[0])
    x_data = x_data[:tot_split]
    
    #Now split into training/test data 70/30?
    split = int(train_frac * len(x_data))
    x_train = x_data[:split]
    x_test = x_data[split:]
    
    return x_train, x_test, shift, div_num 


#Get (Adam) optimiser
def get_optimizer():
    
    return Adam(lr = learning_rate, beta_1 = beta_1, beta_2 = beta_2) 


def wasserstein_loss(y_true, y_pred):
    
    """Calculates the Wasserstein loss for a sample batch.
    The Wasserstein loss function is very simple to calculate. In a standard GAN, the discriminator
    has a sigmoid output, representing the probability that samples are real or generated. In Wasserstein
    GANs, however, the output is linear with no activation function! Instead of being constrained to [0, 1],
    the discriminator wants to make the distance between its output for real and generated samples as large as possible.
    The most natural way to achieve this is to label generated samples -1 and real samples 1, instead of the
    0 and 1 used in normal GANs, so that multiplying the outputs by the labels will give you the loss immediately.
    Note that the nature of this loss means that it can be (and frequently will be) less than 0."""
    
    return K.mean(y_true * y_pred)


#Get loss function
def get_loss_function():
    
    return 'binary_crossentropy'


def gradient_penalty_loss(y_true, y_pred, averaged_samples, gradient_penalty_weight):
    """Calculates the gradient penalty loss for a batch of "averaged" samples.
    In Improved WGANs, the 1-Lipschitz constraint is enforced by adding a term to the
    loss function that penalizes the network if the gradient norm moves away from 1.
    However, it is impossible to evaluate this function at all points in the input
    space. The compromise used in the paper is to choose random points on the lines
    between real and generated samples, and check the gradients at these points. Note
    that it is the gradient w.r.t. the input averaged samples, not the weights of the
    discriminator, that we're penalizing!
    In order to evaluate the gradients, we must first run samples through the generator
    and evaluate the loss. Then we get the gradients of the discriminator w.r.t. the
    input averaged samples. The l2 norm and penalty can then be calculated for this
    gradient.
    Note that this loss function requires the original averaged samples as input, but
    Keras only supports passing y_true and y_pred to loss functions. To get around this,
    we make a partial() of the function with the averaged_samples argument, and use that
    for model training."""
    # first get the gradients:
    #   assuming: - that y_pred has dimensions (batch_size, 1)
    #             - averaged_samples has dimensions (batch_size, nbr_features)
    # gradients afterwards has dimension (batch_size, nbr_features), basically
    # a list of nbr_features-dimensional gradient vectors
    
    gradients = K.gradients(y_pred, averaged_samples)[0]
    
    # compute the euclidean norm by squaring ...
    gradients_sqr = K.square(gradients)
    
    #   ... summing over the rows ...
    gradients_sqr_sum = K.sum(gradients_sqr, axis=np.arange(1, len(gradients_sqr.shape)))
    
    #   ... and sqrt
    gradient_l2_norm = K.sqrt(gradients_sqr_sum)
    
    # compute lambda * (1 - ||grad||)^2 still for each single sample
    gradient_penalty = gradient_penalty_weight * K.square(1 - gradient_l2_norm)
    
    # return the mean as loss over all the batch samples
    return K.mean(gradient_penalty)

#    gradients = K.gradients(y_pred, averaged_samples)
#    gradients = K.concatenate([K.flatten(tensor) for tensor in gradients])
#    gradient_l2_norm = K.sqrt(K.sum(K.square(gradients)))
#    gradient_penalty = grad_penalty_weight * K.square(1 - gradient_l2_norm)
#    return gradient_penalty


class RandomWeightedAverage(_Merge):
    """Takes a randomly-weighted average of two tensors. In geometric terms, this
    outputs a random point on the line between each pair of input points.
    Inheriting from _Merge is a little messy but it was the quickest solution I could
    think of. Improvements appreciated."""

#    def _merge_function(self, inputs):
#        weights = K.random_uniform((batch_size, 1, 1, 1))
#        return (weights * inputs[0]) + ((1 - weights) * inputs[1])

    def _merge_function(self, inputs):
        weights = K.random_uniform((batch_size, 1))
        return (weights * inputs[0]) + ((1 - weights) * inputs[1])


#Build generator network
#Changed 'standard' generator final layer to 1 node rather than 28^2 as single number = image
def build_generator(optimizer):
    
    #Input layer
    gen_input_noise = Input(shape=(noise_dim,), name='Input_noise')
    gen_input_phys = Input(shape=(phys_dim,), name='Input_physics')
    gen_input = concatenate([gen_input_noise, gen_input_phys], axis=-1)
    
    layer = Dense(256)(gen_input)
    layer = LeakyReLU(0.2)(layer)
    layer = BatchNormalization(momentum=0.8)(layer)

    #Internal layers
    for i in range(gen_layers):
        layer = Dense(256)(layer)
        layer = LeakyReLU(0.2)(layer)
        layer = BatchNormalization(momentum=0.8)(layer)
        
    #Output layer
    gen_outputs = Dense(DLLs_dim, activation='tanh')(layer)
    
    generator = Model(inputs=[gen_input_noise, gen_input_phys], outputs=gen_outputs)
    
#   generator = multi_gpu_model(generator, gpus=3)

    return generator


#Build discriminator layers network
#Changed input_dim to 1 (see above)
def build_discriminator(optimizer):
    
    #Input layer
    discrim_input_DLLs = Input(shape=(DLLs_dim,), name='Input_DLLs')
    discrim_input_phys = Input(shape=(phys_dim,), name='Input_physics')
    discrim_input = concatenate([discrim_input_DLLs, discrim_input_phys], axis=-1)
    
    layer = Dense(256)(discrim_input)
    layer = LeakyReLU(0.2)(layer)
    layer = Dropout(0.3)(layer)

    #Internal layers
    for i in range(discrim_layers):
        layer = Dense(256)(layer)
        layer = LeakyReLU(0.2)(layer)
        layer = Dropout(0.3)(layer)
        
    #Output layer
    discrim_outputs = Dense(1, activation='linear')(layer)
    
    discriminator = Model(inputs=[discrim_input_DLLs, discrim_input_phys], outputs=discrim_outputs)
    
#    discriminator = multi_gpu_model(discriminator, gpus=3)
    
    return discriminator


#Build/compile overall GAN network
def build_gan_network(discriminator, generator, optimizer):    
        
    #Generator_model:
    
    #Initially discriminator false as only want to train one at a time
    for layer in discriminator.layers:
        layer.trainable = False
    for layer in generator.layers:
        layer.trainable = True
    
    discriminator.trainable = False
    generator.trainable = True        

    gen_input_noise = Input(shape=(noise_dim,))
    gen_input_phys = Input(shape=(phys_dim,))
    
    gen_output = generator([gen_input_noise, gen_input_phys])
    
    discrim_out_for_gen = discriminator([gen_output, gen_input_phys]) 

    generator_model = Model(inputs=[gen_input_noise, gen_input_phys], outputs=discrim_out_for_gen)
    generator_model.compile(optimizer=optimizer, loss=wasserstein_loss)

    #Discriminator_model:

    #Now allow discriminator to be trained but not generator
    for layer in discriminator.layers:
        layer.trainable = True
    for layer in generator.layers:
        layer.trainable = False
    
    discriminator.trainable = True
    generator.trainable = False

    real_DLLs = Input(shape=(DLLs_dim,))
    gen_noise_in_for_discrim = Input(shape=(noise_dim,))
    real_phys = Input(shape=(phys_dim,))
    
    generated_DLLs_for_discrim = generator([gen_noise_in_for_discrim, real_phys])
    
    discrim_out_from_gen = discriminator([generated_DLLs_for_discrim, real_phys])
    discrim_out_from_real_DLLs = discriminator([real_DLLs, real_phys])
    
    # We also need to generate weighted-averages of real and generated samples,
    # to use for the gradient norm penalty.
    averaged_samples = RandomWeightedAverage()([real_DLLs, generated_DLLs_for_discrim])
    # We then run these samples through the discriminator as well. Note that we never
    # really use the discriminator output for these samples - we're only running them to
    # get the gradient norm for the gradient penalty loss.
    averaged_samples_out = discriminator([averaged_samples, real_phys])

    # The gradient penalty loss function requires the input averaged samples to get
    # gradients. However, Keras loss functions can only have two arguments, y_true and
    # y_pred. We get around this by making a partial() of the function with the averaged
    # samples here.
    partial_gp_loss = partial(gradient_penalty_loss, averaged_samples=averaged_samples, gradient_penalty_weight=grad_penalty_weight)
    # Functions need names or Keras will throw an error
    partial_gp_loss.__name__ = 'gradient_penalty'


    discriminator_model = Model(inputs=[real_DLLs, gen_noise_in_for_discrim, real_phys], 
                                outputs=[discrim_out_from_real_DLLs, discrim_out_from_gen, averaged_samples_out])

    discriminator_model.compile(optimizer=optimizer, loss=[wasserstein_loss, wasserstein_loss, partial_gp_loss])
    
    return discriminator_model, generator_model


def plot_examples(generated_vars, var_name, epoch, bin_no=400, x_range = None, y_range = None):
    
    fig1, ax1 = plt.subplots()
    ax1.cla()
    
    title = 'GAN6_generated_' + var_name + '_epoch_%d.eps'
    
    if y_range is not None:
        ax1.set_ylim(bottom = 0, top = y_range)
    
    if x_range is not None:
        ax1.set_xlim(x_range)
    
    ax1.set_xlabel(var_name)
    ax1.set_ylabel("Number of events")
    ax1.hist(generated_vars, bins=bin_no, range=x_range)
    
    fig1.savefig(title % epoch, format='eps', dpi=2500)


#Plot histogram via 'examples' number of numbers generated
def gen_examples(x_test, epoch, generator, shift, div_num, examples=250000):
     
    #Get data to input to generator
    data_batch = x_test[np.random.randint(0, x_test.shape[0], size=examples)]
    phys_data = data_batch[:, DLLs_dim:]
    noise = np.random.normal(0, 1, size=[examples, noise_dim])

    generated_data = generator.predict([noise, phys_data])
    
    #Shift back to proper distribution?
    for i in range(generated_data.shape[1]):
        
        generated_data[:,i] = np.multiply(generated_data[:,i], div_num[i])
        generated_data[:,i] = np.add(generated_data[:,i], shift[i])    
        
        if i<DLLs_dim:
            plot_examples(generated_data[:,i], 'DLL'+ DLLs[i], epoch)
#        else:
#            plot_examples(generated_vars[:,i], physical_vars[i-len(DLLs)], epoch)

#        if i<phys_dim:
#            phys_data[:,i] = np.multiply(phys_data[:,i], div_num[i+DLLs_dim])
#            phys_data[:,i] = np.add(phys_data[:,i], shift[i+DLLs_dim])    


#Training function. Import data, split into batches to train and train data, 
#plotting data every plot_freq epochs 
def train(epochs=1, batch_size=128):

    print("Importing data...")
    #Get the training and testing data
    x_train, x_test, shift, div_num = get_x_data(DLLs, ref_particle, physical_vars, particle_source)
    print("Data imported")
    
    print("Building network...")
    # Build GAN netowrk
    optimizer = get_optimizer()
    generator = build_generator(optimizer)
    discriminator = build_discriminator(optimizer)
    discriminator_model, generator_model = build_gan_network(discriminator, generator, optimizer)
    print("Network built")

    # Split the training data into batches of size 128
    batch_count = x_train.shape[0] // batch_size

    discrim_loss_tot = []
    discrim_loss_real_tot = []
    discrim_loss_gen_tot = []
    discrim_loss_grad_tot = []
    gen_loss_tot = []

    positive_y = np.ones((batch_size, 1), dtype=np.float32)
    negative_y = -positive_y
    dummy_y = np.zeros((batch_size, 1), dtype=np.float32)

    for i in range(1, epochs+1):

        print('-'*15, 'Epoch %d' % i, '-'*15)

        gen_loss = []
        
        num_batches = int(x_train.shape[0] // batch_size)
        
        np.random.shuffle(x_train)
        print("Number of batches: ", num_batches)
    
        minibatches_size = batch_size * training_ratio
        
        gen_batches = int(x_train.shape[0] // (batch_size * training_ratio))
        
        discrim_loss_batch = np.zeros(gen_batches)
        discrim_loss_real_batch = np.zeros(gen_batches)
        discrim_loss_gen_batch = np.zeros(gen_batches)    
        discrim_loss_grad_batch = np.zeros(gen_batches)
        
        for j in tqdm(range(gen_batches)):
            
            discriminator_minibatches = x_train[j * minibatches_size: (j + 1) * minibatches_size]
            
            discrim_loss = np.zeros(training_ratio)
            discrim_loss_real = np.zeros(training_ratio)
            discrim_loss_gen = np.zeros(training_ratio)    
            discrim_loss_grad = np.zeros(training_ratio)    

            for k in tqdm(range(training_ratio)):
                
                data_batch = discriminator_minibatches[k * batch_size: (k + 1) * batch_size]
                
                DLL_data = data_batch[:, :DLLs_dim]
                phys_data = data_batch[:, DLLs_dim:]
                noise = np.random.normal(0, 1, size=[batch_size, noise_dim])                

                discriminator.trainable = True
                generator.trainable = False
                
                discrim_loss[k], discrim_loss_real[k], discrim_loss_gen[k], discrim_loss_grad[k]  = discriminator_model.train_on_batch([DLL_data, noise, phys_data],  [positive_y, negative_y, dummy_y])
            
            data_batch = x_train[np.random.randint(0, x_train.shape[0], size=batch_size)]            
            phys_data = data_batch[:, DLLs_dim:]
            noise = np.random.normal(0, 1, size=[batch_size, noise_dim])

            generator.trainable = True
            discriminator.trainable = False

            gen_loss.append(generator_model.train_on_batch([noise, phys_data], positive_y))
                
            discrim_loss_batch[j] = np.average(discrim_loss)
            discrim_loss_real_batch[j] = np.average(discrim_loss_real)
            discrim_loss_gen_batch[j] = np.average(discrim_loss_gen)
            discrim_loss_grad_batch[j] = np.average(discrim_loss_grad)


        #Generate histogram via generator every plot_freq epochs
        if i == 1 or i % plot_freq == 0:
            gen_examples(x_test, i, generator, shift, div_num)

        gen_loss_batch_av = [np.average(gen_loss)]
        discrim_loss_batch_av = [np.average(discrim_loss_batch)]
        discrim_loss_real_batch_av = [np.average(discrim_loss_real_batch)]
        discrim_loss_gen_batch_av = [np.average(discrim_loss_gen_batch)]
        discrim_loss_grad_batch_av = [np.average(discrim_loss_grad_batch)]

        gen_loss_tot = np.concatenate((gen_loss_tot, gen_loss_batch_av))
        discrim_loss_tot = np.concatenate((discrim_loss_tot, discrim_loss_batch_av))
        discrim_loss_real_tot = np.concatenate((discrim_loss_real_tot, discrim_loss_real_batch_av))
        discrim_loss_gen_tot = np.concatenate((discrim_loss_gen_tot, discrim_loss_gen_batch_av))
        discrim_loss_grad_tot = np.concatenate((discrim_loss_grad_tot, discrim_loss_grad_batch_av))

    epoch_arr = np.linspace(1,epochs,num=epochs)

    #Plot loss functions
    fig1, ax1 = plt.subplots()
    ax1.cla()
    ax1.plot(epoch_arr, discrim_loss_tot)
    ax1.plot(epoch_arr, discrim_loss_real_tot)
    ax1.plot(epoch_arr, discrim_loss_gen_tot)
    ax1.plot(epoch_arr, discrim_loss_grad_tot)
    ax1.plot(epoch_arr, gen_loss_tot)
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')

    generator.save('trained_gan.h5')  # creates a HDF5 file 'trained_gan.h5'

    ax1.legend(["Combined discriminator loss", "Real DLLs discriminator loss", "Gradient loss", "Generated DLLs discriminator loss", "Generator loss"])
    fig1.savefig('GAN6_loss.eps', format='eps', dpi=2500)

#Call training function
if __name__ == '__main__':
    train(epochs, batch_size) #Epochs, batch size e.g. 400,128

#Measure total run time for script
t_final = time.time()
runtime = t_final - t_init
print("Total run time = ", runtime)

#Save runtime as text
with open('GAN6_runtime.txt', 'w') as f:
    print(runtime, file=f)
