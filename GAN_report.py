#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =============================================================================
# Notes
# =============================================================================

#GAN training script.
#GAN generates DLLs with input physics taken from same datafiles as target DLLs
#Saves generator network periodically for further testing among other outputs

#Builds on GAN_corr_RNN.py. Rewritten to allow implementation of WGAN and 
#gradient penalty again

#Note: cannot calculate gradient of recurrent layers currently so cannot 
#implement both simultaneously

#Also note: WGAN/GP/RNN currently show no improvement over default params

# =============================================================================
# Useful code
# =============================================================================

#Basic GAN:
#github.com/eriklindernoren/Keras-GAN/blob/master/gan/gan.py

#Physics conditioning:
#github.com/eriklindernoren/Keras-GAN/blob/master/cgan/cgan.py

#WGAN-GP code:
#github.com/keras-team/keras-contrib/blob/master/examples/improved_wgan.py

#Elements of GAN-RNN implementation taken from:
#github.com/corynguyen19/midi-lstm-gan/blob/master/mlp_gan.py
#github.com/sachinruk/PyData_Keras_Talk/blob/master/cosine_LSTM.ipynb

# =============================================================================
# Expected file structure to run this file:
# =============================================================================

#Data files for kaon and pion tracks (mod refers to additonal variables added):
# '../../data/mod-PID-train-data-KAONS.hdf'
# '../../data/mod-PID-train-data-PIONS.hdf'

#csv files for normalisation of data (shifts and divisors) calc from datafiles
# '../../data/KAON_norm.csv'
# '../../data/PION_norm.csv'

# =============================================================================
# Outputs
# =============================================================================

# csv file containing defining tracks that have not been used in training: 
# 'unused_data_mask.csv'

#Plots of DLL distributions during training:
# GAN_generated_DLLx_epoch_xxx.eps'

#Trained (and half trained, penultimate trained etc.) GAN:
# 'trained_gan.h5'

#Plots of loss functions:
# 'GAN_loss.eps'

#Runtime:
# GAN_runtime.txt'

# =============================================================================
# Import libraries
# =============================================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

from keras.layers import Input, BatchNormalization, concatenate
from keras.models import Model
from keras.layers.core import Dense, Dropout, Reshape
from keras.layers import CuDNNLSTM, Bidirectional
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam
from keras import initializers

import keras.backend as K
import tensorflow as tf
from keras.layers.merge import _Merge
from functools import partial

#Time total run from here to end
t_init = time.time()

# =============================================================================
# Input parameters
# =============================================================================

os.environ["CUDA_VISIBLE_DEVICES"]="1" #Choose GPU to use e.g. "0", "1" or "2"

os.environ["KERAS_BACKEND"] = "tensorflow" #Using tensorflow backend

plt.rcParams['agg.path.chunksize'] = 10000 #Needed for plotting lots of data

#Training variables
batch_size = 128 #Default = 128
epochs = 200 #Default = 100

#Parameters for Adam optimiser
learning_rate = 0.0001 #Default = 0.0001 (Adam default: learning_rate = 0.001)
beta_1=0.5 #Default = 0.5 (Adam default: beta_1 = 0.9)
beta_2=0.9 #Default = 0.9 (Adam default: beta_2 = 0.999)

#Define how much of the total data to use, and how much of that to train with
frac = 0.1 #Default = 0.1
train_frac = 0.7 #Default = 0.7

#The training ratio is the number of discriminator updates per generator update
training_ratio = 1 #Default 5 as per the paper
grad_penalty_weight = 10  #Default 10 as per the paper

grad_loss = False #Use gradient penalty loss
WGAN = False #Use Wasserstein loss function
RNN = True #Use recurrent layers

#DLL(DLL[i] - ref_particle) from particle_source data 
#e.g. DLL(K-pi) from kaon data file
DLLs = ['e', 'mu', 'k', 'p', 'd', 'bt']

ref_particle = 'pi' #Usually pi(on)
particle_source = 'KAON' #'KAON' or 'PION'

#Physics network will be conditioned on:

physical_vars = ['TrackP', 'TrackPt', 'NumLongTracks', 'NumPVs', 
                 'TrackVertexX', 'TrackVertexY', 'TrackVertexZ', 
                 'TrackRich1EntryX', 'TrackRich1EntryY', 'TrackRich1EntryZ', 
                 'TrackRich1ExitX', 'TrackRich1ExitY', 'TrackRich1ExitZ', 
                 'TrackRich2EntryX', 'TrackRich2EntryY', 'TrackRich2EntryZ', 
                 'TrackRich2ExitX', 'TrackRich2ExitY', 'TrackRich2ExitZ']

noise_dim = 100 # Dimension of random noise vector. Default = 100

phys_dim = len(physical_vars) #Size of phyiscs input
DLLs_dim = len(DLLs) #Number of DLLs

#Size of DLLs and physics combined (input to discim)
data_dim = DLLs_dim + phys_dim 

#Size of noise and physics combined (input to gen)
gen_input_row_dim  = noise_dim + phys_dim 

#Internal layers of generator and discriminator
gen_layers = 6 #Default = 8
discrim_layers = 6 #Default = 8

#Number of nodes in input layers and hidden layers ('x_nodes')
gen_input_nodes = 256 #Default = 256
gen_nodes = 256 #Default = 256
discrim_input_nodes = 256 #Default = 256
discrim_nodes = 256 #Default = 256

#Discriminator output dimensions, usually single scalar corresponding to prob 
#the input was real or generated
discrim_output_dim = 1 #Default = 1

#If including recurrent layers and choose how to sort data. If sort by 'None' 
#will still be sorted by Event/Run number

#Still group by events, but also sort by busy-ness
sort_var = ['RunNumber', 'EventNumber', 'RICH1EntryDist0'] 

#If running recurrent layers, must change input dimensions accordingly, 
#define sequence lengths etc.
if RNN:

    gen_layers -= 1 #RNN increases layers before loop by one so subtract this 
    discrim_layers -= 1 #Same for discriminator

    seq_length = batch_size // 32 #Rows, default 32 (//4)
    
    #Making sequences out of batch_size results in [apparent_batch_size] rows 
    #i.e. treat as new batch size:
    apparent_batch_size = batch_size - seq_length + 1 

    gen_output_dim = (seq_length, DLLs_dim) #Output N rows of DLL values
 
    #Input DLLs and physics as sequences:
    discrim_input_dim = (seq_length, data_dim) 
    discrim_noise_input_dim = (seq_length, noise_dim)
    discrim_phys_input_dim = (seq_length, phys_dim)
    discrim_DLL_input_dim = gen_output_dim
    
    gen_noise_input_dim = (seq_length, noise_dim) #Noise input to combined GAN
    gen_phys_input_dim = (seq_length, phys_dim) #Physics input to combined GAN

else:
    
    apparent_batch_size = batch_size #No change to batch size 
    
    gen_output_dim = DLLs_dim #Output single row of DLLs from gen
    
    discrim_input_dim = (data_dim,) #Input DLLs and physics for discrim
    discrim_noise_input_dim = (noise_dim,) #Input noise for discrim
    discrim_phys_input_dim = (phys_dim,) #Physics input for discrim
    discrim_DLL_input_dim = (gen_output_dim,) #Input DLLs for discrim
    
    gen_noise_input_dim = (noise_dim,) #Noise input to generator
    gen_phys_input_dim = (phys_dim,) #Physics input to generator


#WGAN needs linear discriminator activation rather than sigmoid
if WGAN:
    discrim_activation = 'linear'
else:
    discrim_activation = 'sigmoid'

plot_freq = 20 #epochs//10 #Plot data for after this number of epochs

if RNN and grad_loss:
    print("Error: Cannot use RNN and gradient penalty together as gradients \
          cannot be calculated")

###############################################################################

#Import data of a single variable (e.g. momentum) via pandas from data files
#Inputs: variable type and particle source, corresponding to the single 
#        variable to be imported from the source datafile
#Returns: pandas column containing the single variable data only
def import_single_var(var_type, particle_source):
    
    #Define data file as KAONs/PIONs. 'mod' datafile has newly calc. vars
    if(particle_source == 'KAON'):

        #Define datafile location, read file and select variable of interest
        datafile_kaon = '../../data/mod-PID-train-data-KAONS.hdf'
        data_kaon = pd.read_hdf(datafile_kaon, 'KAONS')
        data = data_kaon.loc[:, var_type]

        
    elif(particle_source == 'PION'):
    
        #Define datafile location, read file and select variable of interest
        datafile_pion = '../../data/mod-PID-train-data-PIONS.hdf'
        data_pion = pd.read_hdf(datafile_pion, 'PIONS')
        data = data_pion.loc[:, var_type]

    else:
        print("Please select either kaon or pion as particle source")

    return data


#Import all data via pandas from data files
#Inputs: Particle source e.g. KAONS corresponding to the datafile from which 
#        data will be imported
#Returns: pandas structure containing all variables from the source
def import_all_var(particle_source):
    
    #Import data from kaons or pions    
    if(particle_source == 'KAON'):        
        datafile = '../../data/mod-PID-train-data-KAONS.hdf' 
    elif(particle_source == 'PION'):    
        datafile = '../../data/mod-PID-train-data-PIONS.hdf' 
    else:
        print("Please select either kaon or pion as particle source")

    data = pd.read_hdf(datafile, particle_source + 'S') #Read in all data

    return data


#Change DLLs e.g. from (K-pi) and (p-pi) to p-K
#Input: Two DLL arrays w.r.t. pi, to be changed s.t. the new DLL is w.r.t. 
#       the first particle in each DLL
#Returns: New DLL array e.g. DLL(p-K)
def change_DLL(DLL1, DLL2):
    
    if(not np.array_equal(DLL1, DLL2)):
        DLL3 = np.subtract(DLL1, DLL2)
    else:
        print("DLLs are the same!")
        DLL3 = DLL1
    
    return DLL3


#Import information needed to normalise all data to between -1 and 1
#Input: particle source i.e. either KAON or PION so read in respective csv 
#       file with vales
#Returns: div_num and shift needed to normalise all relevant data (DLLs and 
#         input physics)
def import_norm_info(particle_source):

    #Read in csv datafile
    data_norm = np.array(pd.read_csv('../../data/' + particle_source + \
                                     '_norm.csv'))
    
    #shift = [0,x], div_num = [1,x], where x starts at 1 for meaningful data

    #Order of variables:
    columns = ['RunNumber', 'EventNumber', 'MCPDGCode', 'NumPVs',
               'NumLongTracks', 'NumRich1Hits',  'NumRich2Hits', 'TrackP',
               'TrackPt', 'TrackChi2PerDof', 'TrackNumDof', 'TrackVertexX',
               'TrackVertexY', 'TrackVertexZ', 'TrackRich1EntryX', 
               'TrackRich1EntryY', 'TrackRich1EntryZ', 'TrackRich1ExitX', 
               'TrackRich1ExitY', 'TrackRich1ExitZ', 'TrackRich2EntryX',
               'TrackRich2EntryY',  'TrackRich2EntryZ','TrackRich2ExitX', 
               'TrackRich2ExitY', 'TrackRich2ExitZ', 'RichDLLe', 'RichDLLmu',
               'RichDLLk', 'RichDLLp', 'RichDLLd', 'RichDLLbt',
               'RICH1EntryDist0', 'RICH1ExitDist0',  'RICH2EntryDist0',
               'RICH2ExitDist0', 'RICH1EntryDist1', 'RICH1ExitDist1',
               'RICH2EntryDist1',  'RICH2ExitDist1', 'RICH1EntryDist2',
               'RICH1ExitDist2', 'RICH2EntryDist2', 'RICH2ExitDist2',
               'RICH1ConeNum', 'RICH2ConeNum']
    
    #Create arrays for shift and div_num to be stored in. Only need to save 
    #DLLs and physics input values (data_dim)
    shift = np.zeros(data_dim)
    div_num = np.zeros(data_dim)
    
    #Loop over all DLLs and input physics
    for i in range(data_dim):
        #First values correspond to DLLs
        if i < DLLs_dim:
            for j in range(len(columns)):
                if columns[j] == 'RichDLL' + DLLs[i]:
                    shift[i] = data_norm[0,j+1]
                    div_num[i] = data_norm[1,j+1]
                    break
        #Next set correspond to physics inputs
        else:
            for k in range(len(columns)):
                if columns[k] == physical_vars[i-DLLs_dim]:
                    shift[i] = data_norm[0,k+1]
                    div_num[i] = data_norm[1,k+1]
                    break

    return shift, div_num


#Normalise data via dividing centre on zero and divide by max s.t. range=[-1,1]
#Input: Data array to be normalised (x) and particle source, so know which 
#       set of normalisation values to use
#Returns: Normalised data array (x) and shift/div_num used to do so (so can 
#         unnormalise later)
def norm(x, particle_source):

    #Import normalistion arrays (shift and div_number) from csv file        
    shift, div_num, = import_norm_info(particle_source)
    
    #For each column in input data array, normalise by shifting and dividing
    for i in range(x.shape[1]):
        
        x[:,i] = np.subtract(x[:,i], shift[i])
        x[:,i] = np.divide(x[:,i], div_num[i])
    
    return x, shift, div_num


#Create sequences of data
#Inputs: data array to be made into sequences, length of sequences (=look_back)
#Returns: arrays of sequenced data, array containing the final row of each seq
#Note: Number of sequences = original number of rows - look_back + 1 
#      e.g. input 10 rows, lookback = 4 -> 7 output
def create_dataset(data, look_back=1):
  
    dataX, dataY = [], []

    #Loop from 0 to len(data)-look_back+1
    for i in range(len(data)-look_back+1):
    
        #Extract [look_back] data rows starting from the ith row and final rows
        a = data[i:(i+look_back), :] 
        dataX.append(a)
        dataY.append(data[i + look_back - 1, :])
    
    return np.array(dataX), np.array(dataY)


#Get all relevant training (and test) data
#Inputs: List of DLLs of interest, list of physical vars of interest, particle 
#        source for data e.g. KAONS
#Returns: training and test data, as well as values used to normalise the data
def get_x_data(DLLs, physical_vars, particle_source):
    
    #Inport all data from particle source datafile
    all_data = import_all_var(particle_source)
    
    #If recurrent layers, can sort by a variable e.g. TrackP. 
    #If 'None' will still be grouped by event/run number
    if RNN and sort_var is not None:
        all_data = all_data.sort_values(by=sort_var,ascending=True)
        
    #Number of data rows (10,000,000 usually)
    data_length = all_data.shape[0]
    
    #Create array for all data to be sorted 
    #(size: number of rows x number of DLLs and physics inputs)      
    x_data_dim = (data_length, data_dim) 
    x_data = np.zeros((x_data_dim))
    
    #Get DLL data
    for i in range(0, DLLs_dim):    
        x_data[:,i] = np.array(all_data.loc[:, 'RichDLL' + DLLs[i]])
    
    #Get physics data
    for i in range(DLLs_dim, DLLs_dim + phys_dim):
        phys_vars_index = i - DLLs_dim
        x_data[:,i] = np.array(all_data.loc[:, physical_vars[phys_vars_index]])

    #Use subset of data for training/testing
    tot_split = int(frac * data_length)

    #Create and array with fraction of 1s and 0s randomly mixed and apply as 
    #boolean mask to use fraction of data only
    zero_arr =np.zeros(data_length - tot_split, dtype=bool)
    ones_arr = np.ones(tot_split, dtype=bool)
    frac_mask = np.concatenate((zero_arr,ones_arr))
    np.random.shuffle(frac_mask)

    #Apply mask to get fraction of data. 
    #Could also shuffle here should not be necessary. Data selected randomly 
    #(but in order here), then batches take a random ordering of this later
    x_data = x_data[frac_mask]

    #Normalise data by shifting and dividing s.t. lies between -1 and 1
    x_data, shift, div_num = norm(x_data, particle_source)    

    #Split into training/test data e.g. 70/30
    split = int(train_frac * tot_split)

    #Create and array with fraction of 1s and 0s randomly mixed and apply as 
    #boolean mask for training/test split
    zero_arr_2 =np.zeros(tot_split - split, dtype=bool)
    ones_arr_2 = np.ones(split, dtype=bool)
    train_mask = np.concatenate((zero_arr_2,ones_arr_2))
    np.random.shuffle(train_mask)
    test_mask = np.logical_not(train_mask)
    
    #Apply masks to get training and test sets of data
    x_train = x_data[train_mask]
    x_test = x_data[test_mask]
    
    #Apply the train mask to the fraction mask, to give a new mask where 0 
    #represents either not in the fraction, or if it was in the frac it was 
    #assigned as test data
    frac_mask[frac_mask==1] = train_mask

    #Take inverse of frac so can extract these 0s
    not_frac_mask = np.logical_not(frac_mask)
    
    #Save mask info so when testing, can use different data to training data
    pd.DataFrame(not_frac_mask).to_csv('unused_data_mask.csv')

    return x_train, x_test, shift, div_num


#Takes randomly-weighted average of two tensors - outputs a random point on 
#the line between each pair of points
class RandomWeightedAverage(_Merge):
    
    def _merge_function(self, inputs):

        # Note: would be (batch_size, 1, 1, 1) for images
        if RNN:
            weights = K.random_uniform((batch_size, 1, 1)) 
        else:
            weights = K.random_uniform((batch_size, 1)) 
        
        return (weights * inputs[0]) + ((1 - weights) * inputs[1])


#Calculate the gradient penalty loss for a batch of "averaged" samples
#Penalise the network if the grad norm moves away from 1
#Random points on the lines between real and generated samples are chosen, 
#and grad checked at these points
#To eval grads, must run samples through the generator and evaluate loss, 
#then get grad of the discrim w.r.t. the input averaged samples
#Loss requires the original averaged samples as input, but Keras only supports
# passing y_true and y_pred to 
#loss functions, so make a partial() of the function with averaged_samlpes 
#argument, and use for training
def gradient_penalty_loss(y_true, y_pred, averaged_samples, 
                          gradient_penalty_weight):

#    First get the gradients:
#      assuming: - that y_pred has dimensions (batch_size, 1)
#                - averaged_samples has dimensions (batch_size, nbr_features)
#    Gradients afterwards has dimension (batch_size, nbr_features), basically 
#    a list of nbr_features-dimensional gradient vectors
        
    gradients = K.gradients(y_pred, averaged_samples)[0]
        
    #Compute the euclidean norm by squaring
    gradients_sqr = K.square(gradients)
    
    #Sum over the rows
    gradients_sqr_sum = K.sum(gradients_sqr, 
                              axis=np.arange(1, len(gradients_sqr.shape)))
    
    #Sqrt
    gradient_l2_norm = K.sqrt(gradients_sqr_sum)
    
    #Compute lambda * (1 - ||grad||)^2 still for each single sample
    gradient_penalty = gradient_penalty_weight * K.square(1 - gradient_l2_norm)
    
    #Return the mean as loss over all the batch samples
    return K.mean(gradient_penalty)


#Get (Adam) optimiser. Can replace Adam with others if required
#Input: None 
#Returns: Adam optimiser with learning rate and beta set at start of code.
def get_optimizer():
    
    return Adam(lr=learning_rate, beta_1=beta_1, beta_2=beta_2) 


#Calculate Wasserstein loss functiont for a sample batch. 
#Note: if used discriminator output must be linear, and samples labelled -1 if 
#      gen and 1 if real (instead of 0/1)
#Inputs: True and predicted y values
#Returns: Wasserstein loss
def wasserstein_loss(y_true, y_pred):
    
    return K.mean(y_true * y_pred)


#Calculate cramer loss function
#Inputs: Generated and real inputs for discriminator, as well as the 
#        discriminator network itself
#Returns: Cramer loss function
def cramer_critic(x, y, discriminator):
    
    discriminated_x = discriminator(x)
    
    return tf.norm(discriminated_x - discriminator(y),
                   axis=1) - tf.norm(discriminated_x, axis=1)


#Get name loss function to be used. If WGAN/Cramer etc must use full functions
#rather than calling this function (see above)
#Inputs: None 
#Returns: 'binary_crossentropy' loss function currently, 
#Note: can replace with other built in Keras loss functions
def get_loss_function():
    
    return 'binary_crossentropy'


#Build generator network, including RNN layers is required. 
#Input/output dimensions, number of layers etc defined at start of code
#Inputs: Optimizer and loss function used to complile network
#Returns: compiled generator network
def build_generator(optimizer, loss_func):

    #Input layer
    gen_input_noise = Input(shape=(gen_noise_input_dim), name='Input_noise')
    gen_input_phys = Input(shape=(gen_phys_input_dim), name='Input_physics')
    gen_input = concatenate([gen_input_noise, gen_input_phys], axis=-1)
    
    #First layer connected to input layer  
    if RNN:
        
        #return_sequences = True to connect two RNN layers. 
        #Would need to flatten after CuDNNLSTM if no Bidirectional layer
        layer = CuDNNLSTM(gen_input_nodes, kernel_initializer = \
                          initializers.RandomNormal(stddev=0.02), 
                          return_sequences=True)(gen_input)
        
        layer = Bidirectional(CuDNNLSTM(gen_input_nodes))(layer)
        
    else:
        layer = Dense(gen_input_nodes, kernel_initializer = \
                      initializers.RandomNormal(stddev=0.02))(gen_input)

        layer = LeakyReLU(0.2)(layer)
        layer = BatchNormalization(momentum=0.8)(layer)

    #Internal layers
    for i in range(gen_layers):
        layer = Dense(256)(layer)
        layer = LeakyReLU(0.2)(layer)
        layer = BatchNormalization(momentum=0.8)(layer)

    #Output layer. Tanh activation so lies between -1 and 1
    if RNN:
        layer = Dense(np.prod(gen_output_dim), activation='tanh')(layer)
        gen_outputs = Reshape(gen_output_dim)(layer)
    else:
        gen_outputs = Dense(gen_output_dim, activation='tanh')(layer)
    
    generator = Model(inputs=[gen_input_noise, gen_input_phys], 
                      outputs=gen_outputs)
    
#   generator = multi_gpu_model(generator, gpus=3)

    if WGAN:
        generator.compile(loss=wasserstein_loss, optimizer=optimizer)
    else:
        generator.compile(loss=loss_func, optimizer=optimizer)
        
    return generator


#Build discriminator network, including RNN layers is required. 
#Input/output dimensions, number of layers etc defined at start of code
#Inputs: Optimizer and loss function used to complile network
#Returns: compiled discriminator network
def build_discriminator(optimizer, loss_func):

    #Input layer
    discrim_input_DLLs = Input(shape=(discrim_DLL_input_dim),
                               name='Input_DLLs')

    discrim_input_phys = Input(shape=(discrim_phys_input_dim),
                               name='Input_physics')

    discrim_input = concatenate([discrim_input_DLLs, discrim_input_phys],
                                axis=-1)
    
    #Input layer    
    if RNN:
        #return_sequences to connext two RNN layers. Would need to flatten 
        #after CuDNNLSTM if no Bidirectional layer
        layer = CuDNNLSTM(discrim_input_nodes, kernel_initializer = \
                          initializers.RandomNormal(stddev=0.02), 
                          return_sequences=True)(discrim_input)

        layer = Bidirectional(CuDNNLSTM(discrim_input_nodes))(layer)

    else:

        layer = Dense(discrim_input_nodes, kernel_initializer = \
                      initializers.RandomNormal(stddev=0.02))(discrim_input)

        layer = LeakyReLU(0.2)(layer)
        layer = Dropout(0.3)(layer)
    
    #Internal layers
    for i in range(discrim_layers):
        layer = Dense(256)(layer)
        layer = LeakyReLU(0.2)(layer)
        if not RNN:
            layer = Dropout(0.3)(layer)
      
    #Output layer. Usually sigmoid activation so outputs 0-1 (false-true) 
    #(if WGAN: linear activation)
    discrim_outputs = Dense(discrim_output_dim, 
                            activation=discrim_activation)(layer)
        
    discriminator = Model(inputs=[discrim_input_DLLs, discrim_input_phys], 
                          outputs=discrim_outputs)    

    if WGAN:
        discriminator.compile(loss=wasserstein_loss, optimizer=optimizer)
    else:
        discriminator.compile(loss=loss_func, optimizer=optimizer)
        
    return discriminator


#Build/compile overall GAN network to train generator - output of which is 
#tested via the discriminator
#Input/output dimensions, number of layers etc defined at start of code
#Inputs: Discriminator and generator networks, as well as optimizer and 
#        loss function used to complile network
#Returns: compiled GAN network
def build_gan_network(discriminator, generator, optimizer, loss_func):
    
# =============================================================================
#     Generator model
# =============================================================================
    
    #Initially discriminator false as only want to train one at a time
    for layer in discriminator.layers:
        layer.trainable = False
    for layer in generator.layers:
        layer.trainable = True
    
    discriminator.trainable = False
    generator.trainable = True

    #GAN input will be n-dimensional vectors (n = noise_dim + phys_dim)
    gen_noise_input = Input(shape=(gen_noise_input_dim))
    gen_phys_input = Input(shape=(gen_phys_input_dim))

    #Output of the generator i.e. DLLs
    gen_output = generator([gen_noise_input, gen_phys_input])
    
    #Get output of discriminator 
    #(Usually probability if the image is real or generated for normal GAN)
    gan_output = discriminator([gen_output, gen_phys_input])

    #Combined model inputs/outputs
    gen_model = Model(inputs=[gen_noise_input, gen_phys_input], 
                      outputs=gan_output)
    
    if WGAN:
        gen_model.compile(loss=wasserstein_loss, optimizer=optimizer)
    else:
        gen_model.compile(loss=loss_func, optimizer=optimizer)


# =============================================================================
#     Discriminator model
# =============================================================================
            
    #Now allow discriminator to be trained but not generator
    for layer in discriminator.layers:
        layer.trainable = True
    for layer in generator.layers:
        layer.trainable = False
    
    discriminator.trainable = True
    generator.trainable = False

    real_DLLs = Input(shape=(discrim_DLL_input_dim))
    discrim_noise_input = Input(shape=(discrim_noise_input_dim))
    discrim_phys_input = Input(shape=(discrim_phys_input_dim))

    generated_DLLs_for_discrim = generator([discrim_noise_input, 
                                            discrim_phys_input])
    discrim_output_from_gen = discriminator([generated_DLLs_for_discrim, 
                                             discrim_phys_input])
    discrim_output_from_real_DLLs = discriminator([real_DLLs, 
                                                   discrim_phys_input])

    if grad_loss:
                
        #Need to generate weighted averages of real and generated samples, 
        #to use for the gradient norm penalty
        averaged_samples = RandomWeightedAverage()([real_DLLs, 
                                                generated_DLLs_for_discrim])
        
        #Run these samples through the discriminator as well. Note, never 
        #really use the discriminator output. For these samples, only running 
        #them to get the gradient norm for the gradient penalty loss.
        averaged_samples_out = discriminator([averaged_samples, 
                                              discrim_phys_input])

        #Gradient penalty loss function requires the input averaged samples to 
        #get gradients. However, Keras loss functions can only have two 
        #arguments, y_true and y_pred. Get around this by making a  partial() 
        #of the function with the averaged samples here.
        partial_gp_loss = partial(gradient_penalty_loss, 
                                  averaged_samples=averaged_samples, 
                                  gradient_penalty_weight=grad_penalty_weight)

        # Functions need names or Keras will throw an error
        partial_gp_loss.__name__ = 'gradient_penalty'

        discrim_model = Model(inputs=[real_DLLs, discrim_noise_input, 
                                      discrim_phys_input],
                              outputs=[discrim_output_from_real_DLLs, 
                                       discrim_output_from_gen, 
                                       averaged_samples_out])

        if WGAN:
            discrim_model.compile(optimizer=optimizer, 
                                  loss=[wasserstein_loss, wasserstein_loss, 
                                        partial_gp_loss])
        else:
            discrim_model.compile(optimizer=optimizer, 
                                  loss=[loss_func, loss_func, partial_gp_loss])

    else:
        discrim_model = Model(inputs=[real_DLLs, discrim_noise_input, 
                                      discrim_phys_input], 
                              outputs=[discrim_output_from_real_DLLs, 
                                       discrim_output_from_gen])

        if WGAN:
            discrim_model.compile(optimizer=optimizer,
                                  loss=[wasserstein_loss, wasserstein_loss])
        else:
            discrim_model.compile(optimizer=optimizer, 
                                  loss=[loss_func, loss_func])

    return gen_model, discrim_model


#Plot examples of generated DLL vales
#Inputs: column of one of the generated DLLs, (DLL) name, epoch of the 
#        training, number of bins, x and y range for histogram
#Returns: 0. DLL distributions plotted and saved.
def plot_examples(generated_vars, var_name, epoch, bin_no=400, x_range = None, 
                  y_range = None):

    fig1, ax1 = plt.subplots()
    ax1.cla()

    title = 'GAN_generated_' + var_name + '_epoch_%d.eps'

    if y_range is not None:
        ax1.set_ylim(bottom = 0, top = y_range)

    if x_range is not None:
        ax1.set_xlim(x_range)

    ax1.set_xlabel(var_name)
    ax1.set_ylabel("Number of events")
    ax1.hist(generated_vars, bins=bin_no, range=x_range)

    fig1.savefig(title % epoch, format='eps', dpi=2500)

    return 0

#Use (trained) generator network to predict DLL values and call plotting 
#function to plot histograms
#Inputs: test data to generate predictions; epoch to save graphs periodically; 
#        generator to make predictions; shift/div_num to unnormalise generated 
#        data; examples to define number of predictions to be made
#Returns: 0. Graphs plotted and saved via plot_examples function
def gen_examples(x_test, epoch, generator, shift, div_num, examples=250000):

    #Define random set integers, to be used to extract random rows from x_test
    batch_ints = np.random.randint(0, x_test.shape[0], size=examples)
            
    #Can use random indicies, but must still be sorted for RNN. 
    #Otherwise use indicies from a specified starting point
    if RNN:                
        batch_ints = np.sort(batch_ints)
 
        #Or have if want all in order:                      
#        start_index = np.random.randint(0, x_test.shape[0] - examples)
#        end_index = start_index + examples
#        batch_ints = np.arange(start_index, end_index)

    #Use batch_ints to form generator input 
    data_batch = x_test[batch_ints]
    phys_data = data_batch[:, DLLs_dim:]
    noise = np.random.normal(0, 1, size=[examples, noise_dim])
    
    gen_input = [noise, phys_data]

    #If RNN need to reshape generator inputs into sequences first              
    if RNN:
        #Create sequences to input to generator
        noise_RNN, _ = create_dataset(noise, seq_length)
        phys_data_RNN, _ = create_dataset(phys_data, seq_length)
        gen_input_RNN = [noise_RNN, phys_data_RNN]

        #Generate fake DLLs and reshape to extract just DLLs from seq form
        generated_data = generator.predict(gen_input_RNN, 
                                           batch_size=apparent_batch_size)

        generated_data = np.concatenate((generated_data[0,:-1,:], 
                                         generated_data[:,-1,:]))

    else:
        #Generate fake data DLLs
        generated_data = generator.predict(gen_input)

    #Shift generated data back to proper distribution and plot data
    for i in range(generated_data.shape[1]):
        
        generated_data[:,i] = np.multiply(generated_data[:,i], div_num[i])
        generated_data[:,i] = np.add(generated_data[:,i], shift[i])    
        plot_examples(generated_data[:,i], 'DLL'+ DLLs[i], epoch)
        
    return 0


#Overall network training function. Imports data, split into batches to train, 
#and train networks
#Inputs: Number of epochs to train over, size of batches to train at a time
#Returns: 0. Calls functions to generate and plot examples periodically. 
#         Generator is also saved periodically, loss functions plotted/saved
def train(epochs=20, batch_size=128):

    print("Importing data...")
    #Get the training/testing data and norm values
    x_train, x_test, shift, div_num = get_x_data(DLLs, physical_vars,
                                                 particle_source)
    print("Data imported")

    # Build GAN netowrk
    print("Building network...")
    optimizer = get_optimizer()
    loss_func = get_loss_function()
    generator = build_generator(optimizer, loss_func)
    discriminator = build_discriminator(optimizer, loss_func)
    gen_model, discrim_model = build_gan_network(discriminator, generator, 
                                                 optimizer, loss_func)
    print("Network built")

    #Places holders for losses to be plotted
    discrim_loss_tot = []
    discrim_loss_real_tot = []
    discrim_loss_gen_tot = []
    discrim_loss_grad_tot = []
    gen_loss_tot = []

    real_discrim_y = np.ones((apparent_batch_size, 1), dtype=np.float32)
    gen_y = real_discrim_y

    if WGAN:    
        fake_discrim_y = -real_discrim_y
    else:
        real_discrim_y[:] = 0.9 #One-sided label smoothing        
        fake_discrim_y = np.zeros((apparent_batch_size, 1), dtype=np.float32)

    if grad_loss:
        dummy_discrim_y = np.zeros((apparent_batch_size, 1), dtype=np.float32)
        
    #Loop over number of epochs. Starting from epoch 1
    for i in range(1, epochs+1):

        print('-'*15, 'Epoch %d' % i, '-'*15)

        #If not RNN, shuffle as rows to be taken sequentially 
        #(if RNN must be sorted so do not shuffle)
        if not RNN:
            np.random.shuffle(x_train)

        #Size of minibatch (data discriminator trained on for each batch 
        #generator is trained on)
        minibatch_size = batch_size * training_ratio 

        #Number of batches generator is trained on
        gen_batches = int(x_train.shape[0] // (batch_size * training_ratio))
        
        #Initialise osses for each batch
        gen_loss = []
        discrim_loss_batch = np.zeros(gen_batches)
        discrim_loss_real_batch = np.zeros(gen_batches)
        discrim_loss_gen_batch = np.zeros(gen_batches)    
        discrim_loss_grad_batch = np.zeros(gen_batches)

        #Loop for each batch. tqdm to show progress bars
        for j in tqdm(range(gen_batches)):
            
            if RNN:
                start_index = np.random.randint(0, (x_train.shape[0] - 
                                                    minibatch_size))
            else:
                start_index = j * minibatch_size
                
            end_index = start_index + minibatch_size
            batch_ints = np.arange(start_index, end_index)
            discrim_minibatches = x_train[batch_ints]
            
            discrim_loss = np.zeros(training_ratio)
            discrim_loss_real = np.zeros(training_ratio)
            discrim_loss_gen = np.zeros(training_ratio)    
            discrim_loss_grad = np.zeros(training_ratio)    

            #Train discriminator [training_ratio] more times than generator
            for k in range(training_ratio):
                
# =============================================================================
#                Training the discriminator
# =============================================================================

                #Use batch_ints get data batch for discriminator input
                if RNN:
                    if training_ratio==1:
                        start_index=0
                    else:
                        start_index = \
                        np.random.randint(0, (discrim_minibatches.shape[0]  - 
                                              batch_size))
                else:
                    start_index = k * batch_size

                end_index = start_index + batch_size
                batch_ints = np.arange(start_index, end_index)

                #Use batch_ints to form discriminator input 
                data_batch = discrim_minibatches[batch_ints]
                phys_data = data_batch[:, DLLs_dim:]
                DLL_data = data_batch[:, :DLLs_dim]
                noise = np.random.normal(0, 1, size=[batch_size, noise_dim])

                #If RNN data must be made into sequences
                if RNN:
                    noise, _ = create_dataset(noise, seq_length)
                    phys_data, _ = create_dataset(phys_data, seq_length)
                    DLL_data, _ = create_dataset(DLL_data, seq_length)
                
                #Train only discriminator, not generator
                discriminator.trainable = True
                generator.trainable = False
                
                #Train discriminator network
                if grad_loss:
                    discrim_loss[k], discrim_loss_real[k], 
                    discrim_loss_gen[k], discrim_loss_grad[k]  = \
                    discrim_model.train_on_batch([DLL_data, noise, phys_data],
                                                 [real_discrim_y, 
                                                  fake_discrim_y, 
                                                  dummy_discrim_y])
                else:
                    discrim_loss[k], discrim_loss_real[k], \
                    discrim_loss_gen[k] = \
                    discrim_model.train_on_batch([DLL_data, noise, phys_data], 
                                                 [real_discrim_y, 
                                                  fake_discrim_y])


            
# =============================================================================
#             Training the generator (combined network)
# =============================================================================

            #Def random set integers, used to extract random rows from x_train
            batch_ints = np.random.randint(0, x_train.shape[0], 
                                           size=batch_size)
            
            #Get batch index numbers. Either sort random ints, or start from 
            #random int and use remainder in order
            if RNN:
#                batch_ints = np.sort(batch_ints)
                start_index = np.random.randint(0,
                                                x_train.shape[0] - batch_size)
                end_index = start_index + batch_size
                batch_ints = np.arange(start_index, end_index)

            #Use batch_ints to form generator input 
            data_batch = x_train[batch_ints]
            phys_data = data_batch[:, DLLs_dim:]
            noise = np.random.normal(0, 1, size=[batch_size, noise_dim])

            #If RNN need to create sequences for noise/physics inputs
            if RNN:
                noise, _ = create_dataset(noise, seq_length)
                phys_data, _ = create_dataset(phys_data, seq_length)

            #Train only generator, not discriminator
            generator.trainable = True
            discriminator.trainable = False

            #Train generator network
            gen_loss.append(gen_model.train_on_batch([noise, phys_data], 
                                                     gen_y))
                
            discrim_loss_batch[j] = np.average(discrim_loss)
            discrim_loss_real_batch[j] = np.average(discrim_loss_real)
            discrim_loss_gen_batch[j] = np.average(discrim_loss_gen)
            discrim_loss_grad_batch[j] = np.average(discrim_loss_grad)

        #Generate histogram via generator every plot_freq epochs
        if i == 1 or i % plot_freq == 0:
            gen_examples(x_test, i, generator, shift, div_num)

        #Save generator model when fully trained, half trained 
        #and the previous two versions before each
        if WGAN:
            network = 'wgan'
        else:
            network = 'gan'
            
        #Create HDF5 files:   
        if i == (epochs//2):
            generator.save('half_trained_' + network + '.h5')  
            
        if i == (epochs//2 -2):
            generator.save('antepenult_half_trained_' + network + '.h5')

        if i == (epochs//2 - 1):
            generator.save('penult_half_trained_' + network + '.h5')
            
        if i == (epochs - 2):
            generator.save('antepenult_trained_' + network + '.h5')
            
        if i == (epochs - 1):
            generator.save('penult_trained_' + network + '.h5')
        
        #Average loss functions over batch to give average for this epoch
        gen_loss_batch_av = [np.average(gen_loss)]
        discrim_loss_batch_av = [np.average(discrim_loss_batch)]
        discrim_loss_real_batch_av = [np.average(discrim_loss_real_batch)]
        discrim_loss_gen_batch_av = [np.average(discrim_loss_gen_batch)]
        discrim_loss_grad_batch_av = [np.average(discrim_loss_grad_batch)]

        gen_loss_tot = np.concatenate((gen_loss_tot, gen_loss_batch_av))

        discrim_loss_tot = np.concatenate((discrim_loss_tot, 
                                           discrim_loss_batch_av))

        discrim_loss_real_tot = np.concatenate((discrim_loss_real_tot, 
                                                discrim_loss_real_batch_av))

        discrim_loss_gen_tot = np.concatenate((discrim_loss_gen_tot, 
                                               discrim_loss_gen_batch_av))

        discrim_loss_grad_tot = np.concatenate((discrim_loss_grad_tot, 
                                                discrim_loss_grad_batch_av))
        
    
    #Array of numbers corresponding to each epoch of training
    epoch_arr = np.linspace(1, epochs, num=epochs)

    #Plot loss functions
    fig1, ax1 = plt.subplots()
    ax1.cla()
    ax1.plot(epoch_arr, discrim_loss_tot) #Plot discriminator loss
    ax1.plot(epoch_arr, discrim_loss_real_tot) #Plot real DLL discrim loss 
    ax1.plot(epoch_arr, discrim_loss_gen_tot) #Plot generated DLL discrim loss
    ax1.plot(epoch_arr, gen_loss_tot) #Plot generator loss
   
    if grad_loss:
        ax1.plot(epoch_arr, discrim_loss_grad_tot) #Plot gradient loss function
        ax1.legend(["Combined discriminator loss", 
                    "Real DLLs discriminator loss", "Gradient loss", 
                    "Generated DLLs discriminator loss", "Generator loss"])
    else:
        ax1.legend(["Combined discriminator loss", 
                    "Real DLLs discriminator loss", 
                    "Generated DLLs discriminator loss", "Generator loss"])
    
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    fig1.savefig(network + '_loss.eps', format='eps', dpi=2500)

    if WGAN:
        generator.save('trained_wgan.h5')  #Create HDF5 file 'trained_gan.h5'
    else: 
        generator.save('trained_gan.h5') #Save gen as 'trained_gan.h5'
        
    return 0


#Call training function
if __name__ == '__main__':
    train(epochs, batch_size) #Epochs, batch size e.g. 400,128
    
#Measure total run time for script
t_final = time.time()
runtime = t_final - t_init
print("Total run time = ", runtime)

#Save runtime as text
with open('GAN_runtime.txt', 'w') as f:
    print(runtime, file=f)
