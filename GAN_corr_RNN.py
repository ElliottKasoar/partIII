#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 17:28:09 2019

@author: Elliott
"""

#This is building on the code which allows correlations to be included.
#Now also include recurrent layers
#To do so, must order data that is fed in e.g. first order by TrackP
#Then try ordering by busy-ness e.g. RICH1ConeNum

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

from keras.layers import Input, BatchNormalization, concatenate, Flatten
from keras.models import Model, Sequential
from keras.layers.core import Dense, Dropout, Reshape
from keras.layers import CuDNNLSTM, Bidirectional
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam
from keras import initializers

from keras.utils import multi_gpu_model
import keras.backend as K
import tensorflow as tf

#from matplotlib.ticker import AutoMinorLocator
#from scipy.stats import gaussian_kde
#import math
#from sklearn.preprocessing import QuantileTransformer

#Time total run
t_init = time.time()

#Input parameters
##############################################################################################################

#Choose GPU to use
os.environ["CUDA_VISIBLE_DEVICES"]="2"

#Using tensorflow backend
os.environ["KERAS_BACKEND"] = "tensorflow"

plt.rcParams['agg.path.chunksize'] = 10000 #Needed for plotting lots of data?

#Some tunable variables/parameters...
#Not really passed properly

#Training variables
batch_size = 128 #Default = 128
epochs = 100 #Default =  500

#Parameters for Adam optimiser
learning_rate = 0.0001 #Default = 0.0001
beta_1=0.5 #Default = 0.5

frac = 0.05 #Default = 0.1
train_frac = 0.7 #Default = 0.7

#DLL(DLL[i] - ref_particle) from particle_source data
DLLs = ['e', 'mu', 'k', 'p', 'd', 'bt']

#physical_vars = ['TrackP', 'TrackPt', 'NumLongTracks', 'NumPVs', 'TrackVertexX', 'TrackVertexY', 'TrackVertexZ', 
#                 'TrackRich1EntryX', 'TrackRich1EntryY', 'TrackRich1EntryZ', 'TrackRich1ExitX', 'TrackRich1ExitY', 
#                 'TrackRich1ExitZ', 'TrackRich2EntryX', 'TrackRich2EntryY', 'TrackRich2EntryZ', 'TrackRich2ExitX', 
#                 'TrackRich2ExitY', 'TrackRich2ExitZ', 'RICH1EntryDist0', 'RICH1ExitDist0', 'RICH2EntryDist0',
#                 'RICH2ExitDist0', 'RICH1EntryDist1', 'RICH1ExitDist1', 'RICH2EntryDist1', 'RICH2ExitDist1', 
#                 'RICH1EntryDist2', 'RICH1ExitDist2', 'RICH2EntryDist2', 'RICH2ExitDist2', 'RICH1ConeNum',
#                 'RICH2ConeNum']

physical_vars = ['TrackP', 'TrackPt', 'NumLongTracks', 'NumPVs', 'TrackVertexX', 'TrackVertexY', 'TrackVertexZ', 
                 'TrackRich1EntryX', 'TrackRich1EntryY', 'TrackRich1EntryZ', 'TrackRich1ExitX', 'TrackRich1ExitY', 
                 'TrackRich1ExitZ', 'TrackRich2EntryX', 'TrackRich2EntryY', 'TrackRich2EntryZ', 'TrackRich2ExitX', 
                 'TrackRich2ExitY', 'TrackRich2ExitZ']


#physical_vars = ['TrackP', 'TrackPt', 'NumLongTracks', 'NumPVs', 'RICH1EntryDist0', 'RICH1ExitDist0', 
#                 'RICH2EntryDist0', 'RICH2ExitDist0', 'RICH1EntryDist1', 'RICH1ExitDist1', 
#                 'RICH2EntryDist1', 'RICH2ExitDist1', 'RICH1EntryDist2', 'RICH1ExitDist2', 
#                 'RICH2EntryDist2', 'RICH2ExitDist2', 'RICH1ConeNum','RICH2ConeNum']

ref_particle = 'pi'
particle_source = 'KAON'

noise_dim = 100 #Dimension of random noise vector. Default = 100
#gen_input_dim = 100
#noise_dim = gen_input_dim - phys_dim

phys_dim = len(physical_vars)
DLLs_dim = len(DLLs)
data_dim = DLLs_dim + phys_dim
gen_input_row_dim  = noise_dim + phys_dim

#Internal layers of generator and discriminator
gen_layers = 8 #Default 8
discrim_layers = 8 #Default 8
gen_nodes = 256 #Default 256
discrim_nodes = 256 #Default 256

discrim_output_dim = 1

RNN = True
sort_var = 'RICH1EntryDist0'

if RNN:

    gen_layers -= 1 #Currently have two LSTM layers before loop, rather than one input layer if not gen_RNN
    discrim_layers -= 1

    seq_length = batch_size // 4 #Rows, default 32

    gen_input_dim = (seq_length, gen_input_row_dim) #Input N rows of noise and physics    
    gen_output_dim = (seq_length, DLLs_dim) #Output N rows of DLL values. *****************Needs fixing
 
    discrim_input_dim = (seq_length, data_dim)
    discrim_phys_input_dim = (seq_length, phys_dim)
    
    gan_noise_input_dim = (seq_length, noise_dim,)
    gan_phys_input_dim = (seq_length, phys_dim,)

    output_batch_size = batch_size - seq_length + 1
    
else:
    
    gen_input_dim = gen_input_row_dim #Input single row of noise and physics    
    gen_output_dim = DLLs_dim #Output single row of DLLs
    
    discrim_input_dim = data_dim
    discrim_phys_input_dim = (phys_dim,)
    
    gan_noise_input_dim = (noise_dim,)
    gan_phys_input_dim = (phys_dim,)
    
    output_batch_size = batch_size


plot_freq = 20 #epochs//10 #Plot data for after this number of epochs

#So reproducable
np.random.seed(10)

##############################################################################################################

#Import data via pandas from data files
def import_single_var(var_type, particle_source):
    
    if(particle_source == 'KAON'):
    
        datafile_kaon = '../../data/mod-PID-train-data-KAONS.hdf'
        data_kaon = pd.read_hdf(datafile_kaon, 'KAONS')
        data_loc = data_kaon
        
    elif(particle_source == 'PION'):
    
        datafile_pion = '../../data/mod-PID-train-data-PIONS.hdf' 
        data_pion = pd.read_hdf(datafile_pion, 'PIONS') 
        data_loc = data_pion

    else:
        print("Please select either kaon or pion as particle source")

    data = data_loc.loc[:, var_type]

    return data


def import_all_var(particle_source):
    
    #Import data from kaons and pions    
    if(particle_source == 'KAON'):        
        datafile = '../../data/mod-PID-train-data-KAONS.hdf' 
    elif(particle_source == 'PION'):    
        datafile = '../../data/mod-PID-train-data-PIONS.hdf' 
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


def norm_info(particle_source):

    data_norm = np.array(pd.read_csv('../../data/' + particle_source + '_norm.csv'))
#    KAON_norm = np.array(pd.read_csv('KAON_norm.csv'))
    
    #shift = [0,x], div_num = [1,x], x starts at 1

    #Order of variables:
    columns = ['RunNumber', 'EventNumber', 'MCPDGCode', 'NumPVs', 'NumLongTracks',
           'NumRich1Hits', 'NumRich2Hits', 'TrackP', 'TrackPt', 'TrackChi2PerDof',
           'TrackNumDof', 'TrackVertexX', 'TrackVertexY', 'TrackVertexZ',
           'TrackRich1EntryX', 'TrackRich1EntryY', 'TrackRich1EntryZ',
           'TrackRich1ExitX', 'TrackRich1ExitY', 'TrackRich1ExitZ',
           'TrackRich2EntryX', 'TrackRich2EntryY', 'TrackRich2EntryZ',
           'TrackRich2ExitX', 'TrackRich2ExitY', 'TrackRich2ExitZ', 'RichDLLe',
           'RichDLLmu', 'RichDLLk', 'RichDLLp', 'RichDLLd', 'RichDLLbt',
           'RICH1EntryDist0', 'RICH1ExitDist0', 'RICH2EntryDist0',
           'RICH2ExitDist0', 'RICH1EntryDist1', 'RICH1ExitDist1',
           'RICH2EntryDist1', 'RICH2ExitDist1', 'RICH1EntryDist2',
           'RICH1ExitDist2', 'RICH2EntryDist2', 'RICH2ExitDist2', 'RICH1ConeNum',
           'RICH2ConeNum']
    
    shift = np.zeros(data_dim)
    div_num = np.zeros(data_dim)
    
    for i in range(data_dim):
        if i < DLLs_dim:
            for j in range(len(columns)):
                if columns[j] == 'RichDLL' + DLLs[i]:
                    shift[i] = data_norm[0,j+1]
                    div_num[i] = data_norm[1,j+1]
                    break
        else:
            for k in range(len(columns)):
                if columns[k] == physical_vars[i-DLLs_dim]:
                    shift[i] = data_norm[0,k+1]
                    div_num[i] = data_norm[1,k+1]
                    break

    return shift, div_num


#Normalise data via dividing centre on zero and divide by max s.t. range=[-1,1]
def norm(x, particle_source):
    
#    shift = np.zeros(x.shape[1])
#    div_num = np.zeros(x.shape[1])
    
    shift, div_num, = norm_info(particle_source)
    
    for i in range(x.shape[1]):
        
#        x_max = np.max(x[:,i])
#        x_min = np.min(x[:,i])
    
#        shift[i] = (x_max + x_min)/2
        x[:,i] = np.subtract(x[:,i], shift[i])
        
#        if x_max == x_min:
#            div_num[i] = 1
#        else:
#                div_num[i] = x_max - shift[i]
        x[:,i] = np.divide(x[:,i], div_num[i])
    
    return x, shift, div_num

def create_dataset(dataset, look_back=1):
  
    dataX, dataY = [], []
    
    for i in range(len(dataset)-look_back+1):
    
        #Extract [look_back] data rows starting from the ith row
        a = dataset[i:(i+look_back), :]
        
        
        dataX.append(a)
        dataY.append(dataset[i + look_back - 1, :])
    
    return np.array(dataX), np.array(dataY)


#Get training/test data and normalise
def get_x_data(DLLs, ref_particle, physical_vars, particle_source):
    
    all_data = import_all_var(particle_source)
    
    if RNN:
        all_data = all_data.sort_values(by=sort_var,ascending=True)
        
    data_length = all_data.shape[0]
    
    #Get first set of DLL data
    DLL_data_1 = np.array(all_data.loc[:, 'RichDLL' + DLLs[0]])
                
    x_data_dim = (data_length, DLLs_dim + phys_dim) 
    x_data = np.zeros((x_data_dim))
    x_data[:,0] = DLL_data_1
    
    #Get other DLL data
    for i in range(1, DLLs_dim):    
        x_data[:,i] = np.array(all_data.loc[:, 'RichDLL' + DLLs[i]])
    
    #Get physics data
    for i in range(DLLs_dim, DLLs_dim + phys_dim):
        phys_vars_index = i - DLLs_dim
        x_data[:,i] = np.array(all_data.loc[:, physical_vars[phys_vars_index]])

    #Use subset of data
    tot_split = int(frac * data_length)

    #Create and array with fraction of 1s and 0s randomly mixed and apply as boolean mask to use fraction of data only
    zero_arr =np.zeros(data_length - tot_split, dtype=bool)
    ones_arr = np.ones(tot_split, dtype=bool)
    frac_mask = np.concatenate((zero_arr,ones_arr))
    np.random.shuffle(frac_mask)

    x_data = x_data[frac_mask]

    #(Shuffle) and normalise data by shifting and dividing s.t. lies between -1 and 1
    #Shuffle should not be necessary, as data is selected randomly (but in order), then batches take a random ordering of this
#    if not(RNN):
#        np.random.shuffle(x_data)

    x_data, shift, div_num = norm(x_data, particle_source)    

    #Split into training/test data e.g. 70/30
    split = int(train_frac * tot_split)

    #Create and array with fraction of 1s and 0s randomly mixed and apply as boolean mask for training/test split
    zero_arr_2 =np.zeros(tot_split - split, dtype=bool)
    ones_arr_2 = np.ones(split, dtype=bool)
    train_mask = np.concatenate((zero_arr_2,ones_arr_2))
    np.random.shuffle(train_mask)
    test_mask = np.logical_not(train_mask)
    
    x_train = x_data[train_mask]
    x_test = x_data[test_mask]
    
    #Apply the train mask to the fraction mask, to give a new mask where 0 represents either not in the fraction, or if it was in the frac it was assigned as test data
    frac_mask[frac_mask==1] = train_mask

    #Take inverse of frac so can extract these 0s
    not_frac_mask = np.logical_not(frac_mask)
    
    #Save mask info
    pd.DataFrame(not_frac_mask).to_csv('unused_data_mask.csv')

    return x_train, x_test, shift, div_num


#Get (Adam) optimiser
def get_optimizer():
    
    return Adam(lr = learning_rate, beta_1 = beta_1) 


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


def cramer_critic(x, y, discriminator):
    
    discriminated_x = discriminator(x)
    
    return tf.norm(discriminated_x - discriminator(y), axis=1) - tf.norm(discriminated_x, axis=1)


#Get loss function
def get_loss_function():
    
    return 'binary_crossentropy'


#Build generator network
#Changed 'standard' generator final layer to 1 node rather than 28^2 as single number = image
def build_generator(optimizer, loss_func):

    #Input layer    
    generator = Sequential()

    if RNN:
        generator.add(CuDNNLSTM(gen_nodes, input_shape=gen_input_dim, kernel_initializer=initializers.RandomNormal(stddev=0.02), return_sequences=True))
        generator.add(Bidirectional(CuDNNLSTM(gen_nodes)))
#        generator.add(Flatten()) #Would need if didn't have Bidirectional layer?
    else:
        generator.add(Dense(gen_nodes, input_dim=gen_input_dim, kernel_initializer=initializers.RandomNormal(stddev=0.02)))
        generator.add(LeakyReLU(0.2))
        generator.add(BatchNormalization(momentum=0.8))

    #Internal layers
    for i in range(gen_layers):
        generator.add(Dense(gen_nodes))
        generator.add(LeakyReLU(0.2))
        generator.add(BatchNormalization(momentum=0.8))

    #Output layer
    if RNN:
        generator.add(Dense(np.prod(gen_output_dim), activation='tanh'))
        generator.add(Reshape(gen_output_dim))

    else:
        generator.add(Dense(gen_output_dim, activation='tanh'))
    
#    generator = multi_gpu_model(generator, gpus=2)

    generator.compile(loss=loss_func, optimizer=optimizer)
    
    return generator

#Build discriminator layers network
#Changed input_dim to 1 (see above)
def build_discriminator(optimizer, loss_func):
    
    discriminator = Sequential()

    #Input layer    
    if RNN:
        discriminator.add(CuDNNLSTM(discrim_nodes, input_shape=discrim_input_dim, kernel_initializer=initializers.RandomNormal(stddev=0.02), return_sequences=True))
        discriminator.add(Bidirectional(CuDNNLSTM(discrim_nodes)))
#        discriminator.add(Flatten()) Would need if didn't have Bidirectional layer?

    else:
        discriminator.add(Dense(discrim_nodes, input_dim=discrim_input_dim, kernel_initializer=initializers.RandomNormal(stddev=0.02)))
        discriminator.add(LeakyReLU(0.2))
        discriminator.add(Dropout(0.3))
    
    #Internal layers
    for i in range(discrim_layers):
        discriminator.add(Dense(discrim_nodes))
        discriminator.add(LeakyReLU(0.2))
        discriminator.add(Dropout(0.3))
    
    #Output layer
    discriminator.add(Dense(discrim_output_dim, activation='sigmoid'))
    
#    discriminator = multi_gpu_model(discriminator, gpus=2)
    
    discriminator.compile(loss=loss_func, optimizer=optimizer)
    
    return discriminator


#Build/compile overall GAN network
def build_gan_network(discriminator, generator, optimizer, loss_func, batch_size):

    #Initially set trainable to False since only want to train generator or discriminator at a time
    discriminator.trainable = False

    #GAN input will be n-dimensional vectors (n = noise_dim + phys_dim)
    gan_noise_input = Input(shape=(gan_noise_input_dim))
    gan_phys_input = Input(shape=(gan_phys_input_dim))

    gan_input = concatenate([gan_noise_input, gan_phys_input], axis=-1)

    #Output of the generator i.e. DLLs
    gen_output = generator(gan_input)
    
    #Generator output + real physics is input for discriminator
    discrim_input = concatenate([gen_output, gan_phys_input], axis=-1)

    #Get output of discriminator (probability if the image is real or not)
    gan_output = discriminator(discrim_input)

    gan = Model(inputs=[gan_noise_input, gan_phys_input], outputs=gan_output)

#    gan = multi_gpu_model(gan, gpus=2)

    gan.compile(loss=loss_func, optimizer=optimizer)

    return gan


def plot_examples(generated_vars, var_name, epoch, bin_no=400, x_range = None, y_range = None):

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


#Plot histogram via 'examples' number of numbers generated
def gen_examples(x_test, epoch, generator, shift, div_num, examples=250000):

    #Get data to input to generator:

    batch_ints = np.random.randint(0, x_test.shape[0], size=examples)
            
    #Have taken random sample, but still want to be sorted as before. Consider taking sample sequentially rather than in order. RNN for either model?
    if RNN:
        batch_ints = np.sort(batch_ints) 

    data_batch = x_test[batch_ints]
    phys_data = data_batch[:, DLLs_dim:]
    noise = np.random.normal(0, 1, size=[examples, noise_dim])

    gen_input = np.concatenate((noise, phys_data), axis=1)
              
    #Generate fake data (DLLs only)
    if RNN:
       
        noise_RNN, _ = create_dataset(noise, seq_length)
        phys_data_RNN, _ = create_dataset(phys_data, seq_length)
        gen_input_RNN = np.concatenate((noise_RNN, phys_data_RNN), axis=2)

        #Generate fake data (DLLs only)
        generated_data = generator.predict(gen_input_RNN)
        generated_data = np.concatenate((generated_data[0,:-1,:],generated_data[:,-1,:]))
#        generated_data = np.reshape(generated_data,(249969*32,6))

    else:
        #Generate fake data (DLLs only)
        generated_data = generator.predict(gen_input)

    #Shift back to proper distribution?
    for i in range(generated_data.shape[1]):
        
        generated_data[:,i] = np.multiply(generated_data[:,i], div_num[i])
        generated_data[:,i] = np.add(generated_data[:,i], shift[i])    
        plot_examples(generated_data[:,i], 'DLL'+ DLLs[i], epoch)


#Training function. Import data, split into batches to train and train data, plotting data every plot_freq epochs 
def train(epochs=20, batch_size=128):

    print("Importing data...")
    #Get the training and testing data
    x_train, x_test, shift, div_num = get_x_data(DLLs, ref_particle, physical_vars, particle_source)
    print("Data imported")

    # Split the training data into batches of size 128
    batch_count = x_train.shape[0] // batch_size

    # Build GAN netowrk
    optimizer = get_optimizer()
    loss_func = get_loss_function()
    generator = build_generator(optimizer, loss_func)
    discriminator = build_discriminator(optimizer, loss_func)
    gan = build_gan_network(discriminator, generator, optimizer, loss_func, batch_size)

    discrim_loss_tot = []
    gen_loss_tot = []

    for i in range(1, epochs+1):

        print('-'*15, 'Epoch %d' % i, '-'*15)

        discrim_loss = []
        gen_loss = []

        for _ in tqdm(range(batch_count)):

            #Get data to train discriminator:
            batch_ints = np.random.randint(0, x_train.shape[0], size=batch_size)
            noise = np.random.normal(0, 1, size=[batch_size, noise_dim])
                
            #Have taken random sample, but still want to be sorted as before if RNN. Consider taking sample sequentially rather than in order? RNN for either model?
            if RNN:
                batch_ints = np.sort(batch_ints) 
            
            data_batch = x_train[batch_ints]
            phys_data = data_batch[:, DLLs_dim:]
            DLL_data = data_batch[:, :DLLs_dim]

            #Generate fake data (DLLs only)
            if RNN:
                noise_RNN, _ = create_dataset(noise, seq_length)
                phys_data_RNN, _ = create_dataset(phys_data, seq_length)
                DLL_data_RNN, _ = create_dataset(DLL_data, seq_length)
                
                gen_input = np.concatenate((noise_RNN, phys_data_RNN), axis=2)
                real_discrim_input = np.concatenate((DLL_data_RNN, phys_data_RNN), axis=2)

            else:
                gen_input = np.concatenate((noise, phys_data), axis=1)
                real_discrim_input = np.concatenate((DLL_data, phys_data), axis=1)

            #Predict data with generator to be fed to discriminator. Output shape (128, 6), or (97, 32, 6) if RNN
            generated_data = generator.predict(gen_input) 

            #Combine generated data and physics data
            if RNN:                        
                generated_discrim_input = np.concatenate((generated_data, phys_data_RNN), axis=2)
            else:
                generated_discrim_input = np.concatenate((generated_data, phys_data), axis=1)                

            #Input real and generated DLL and physics data to discriminator
            discrim_input = np.concatenate([real_discrim_input, generated_discrim_input])

            #Labels for generated and real data
            y_dis = np.zeros(2*output_batch_size) #Length 2* batch_size or 2*(batch_size - seq_length + 1)

            #One-sided label smoothing
            y_dis[:output_batch_size] = 0.9

            #Train discriminator
            discriminator.trainable = True
            discrim_loss.append(discriminator.train_on_batch(discrim_input, y_dis))

            ######################################################################################################################################################
            #Get data to train generator
            
            batch_ints = np.random.randint(0, x_train.shape[0], size=batch_size)
            
            #Have taken random sample, but still want to be sorted as before. Consider taking sample sequentially rather than in order
            if RNN:
                batch_ints = np.sort(batch_ints) 
            
            data_batch = x_train[batch_ints]
            phys_data = data_batch[:, DLLs_dim:]
            noise = np.random.normal(0, 1, size=[batch_size, noise_dim])

            #Train generator
            discriminator.trainable = False
            y_gen = np.ones(output_batch_size) #batch_size-32+1 

            if RNN:
                noise_RNN, _ = create_dataset(noise, seq_length)
                phys_data_RNN, _ = create_dataset(phys_data, seq_length)
                
            if RNN:  
                gen_loss.append(gan.train_on_batch([noise_RNN, phys_data_RNN], y_gen))
            else:
                gen_loss.append(gan.train_on_batch([noise, phys_data], y_gen))


        #Generate histogram via generator every plot_freq epochs
        if i == 1 or i % plot_freq == 0:
            gen_examples(x_test, i, generator, shift, div_num)

        if i == (epochs//2):
            generator.save('half_trained_gan.h5')  # creates a HDF5 file 'trained_gan.h5'
            
        if i == (epochs//2 -2):
            generator.save('antepenult_half_trained_gan.h5')  # creates a HDF5 file 'trained_gan.h5'

        if i == (epochs//2 - 1):
            generator.save('penult_half_trained_gan.h5')  # creates a HDF5 file 'trained_gan.h5'
            
        if i == (epochs - 2):
            generator.save('antepenult_trained_gan.h5')  # creates a HDF5 file 'trained_gan.h5'
            
        if i == (epochs - 1):
            generator.save('penult_trained_gan.h5')  # creates a HDF5 file 'trained_gan.h5'
            
        gen_loss_batch_av = [np.average(gen_loss)]
        discrim_loss_batch_av = [np.average(discrim_loss)]

        gen_loss_tot = np.concatenate((gen_loss_tot, gen_loss_batch_av))
        discrim_loss_tot = np.concatenate((discrim_loss_tot, discrim_loss_batch_av))

    epoch_arr = np.linspace(1, epochs, num=epochs)

    #Plot loss functions
    fig1, ax1 = plt.subplots()
    ax1.cla()
    ax1.plot(epoch_arr, discrim_loss_tot)
    ax1.plot(epoch_arr, gen_loss_tot)
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')

    generator.save('trained_gan.h5')  # creates a HDF5 file 'trained_gan.h5'

    ax1.legend(["Discriminator loss", "Generator loss"])
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
