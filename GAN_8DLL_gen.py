#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 13 15:20:22 2019

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
#Runtime=106634.2. set10 but epochs = 500, frac = 0.25. Worked reasonably well

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
#Runtime=12599.4. Worked well ish


#Consider learning rate decay?

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

from keras.layers import Input, BatchNormalization
from keras.models import Model, Sequential
from keras.layers.core import Dense, Dropout
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam
from keras import initializers

from keras.utils import multi_gpu_model


#from matplotlib.ticker import AutoMinorLocator
#from scipy.stats import gaussian_kde
#import math
#from sklearn.preprocessing import QuantileTransformer

#Time total run
t_init = time.time()

#Choose GPU to use
os.environ["CUDA_VISIBLE_DEVICES"]="1" 

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

noise_dim = 100 #Dimension of random noise vector.

frac = 0.1
train_frac = 0.7

#DLL(DLL[i] - ref_particle) from particle_source data
DLLs = ['e', 'mu', 'k', 'p', 'd', 'bt']
physical_vars = ['TrackP', 'TrackPt']
ref_particle = 'pi'
particle_source = 'KAON'

data_dim = len(DLLs) + len(physical_vars) 

plot_freq = 20 #epochs//10 #Plot data for after this number of epochs

#So reproducable
np.random.seed(10)


#Import data via pandas from data files
def import_data(var_type, particle_source):
    #Import data from kaons and pions
    datafile_kaon = '../../data/PID-train-data-KAONS.hdf'
    data_kaon = pd.read_hdf(datafile_kaon, 'KAONS')
    #print(data_kaon.columns)

    datafile_pion = '../../data/PID-train-data-PIONS.hdf' 
    data_pion = pd.read_hdf(datafile_pion, 'PIONS') 
    #print(data_pion.columns)
    
    if(particle_source == 'KAON'):
        data_loc = data_kaon
    elif(particle_source == 'PION'):
        data_loc = data_pion
    else:
        print("Please select either kaon or pion as particle source")

    data = data_loc.loc[:, var_type]

    return data


#Change DLLs e.g. from K-pi to p-K
def change_DLL(DLL1, DLL2):
    
    if(not np.array_equal(DLL1, DLL2)):
        DLL3 = np.subtract(DLL1, DLL2)
    else:
        print("DLLs are the same!")
        DLL3 = DLL1
    
    return DLL3


#Get training/test data and normalise
def get_x_data(DLLs, DLL_part_2, physical_vars, particle_source):
    
    #Get first set of DLL data
    DLL_data_1 = np.array(import_data('RichDLL' + DLLs[0], particle_source))
    
    x_data_dim = (DLL_data_1.shape[0], len(DLLs) + len(physical_vars)) 
    x_data = np.zeros((x_data_dim))
    x_data[:,0] = DLL_data_1
    
    #Get other DLL data
    for i in range(1, len(DLLs)):    
        x_data[:,i] = np.array(import_data('RichDLL' + DLLs[i], particle_source))
    
    
    #Get physics data
    for i in range(len(DLLs), len(DLLs) + len(physical_vars)):
        phys_vars_index = i - len(DLLs)
        x_data[:,i] = np.array(import_data(physical_vars[phys_vars_index], particle_source))
    
    
    x_data, shift, div_num = norm(x_data)
    
    #Use subset of data
    tot_split = int(frac * x_data.shape[0])
    x_data = x_data[:tot_split]
    
    #Now split into training/test data 70/30?
    split = int(train_frac * len(x_data))
    x_train = x_data[:split]
    x_test = x_data[split:]
    
#    x_train, shift, div_num = norm(x_train)
#    x_test, test_shift, test_div_num = norm(x_test)

    return x_train, x_test, shift, div_num 


#Normalise data via dividing centre on zero and divide by max s.t. range=[-1,1]
def norm(x):
    
    shift = np.zeros(x.shape[1])
    div_num = np.zeros(x.shape[1])
    
    for i in range(x.shape[1]):
        
        x_max = np.max(x[:,i])
        x_min = np.min(x[:,i])
    
        shift[i] = (x_max + x_min)/2
        x[:,i] = np.subtract(x[:,i], shift[i])
    
        div_num[i] = x_max - shift[i]
        x[:,i] = np.divide(x[:,i], div_num[i])
    
    return x, shift, div_num


#Get (Adam) optimiser
def get_optimizer():
    
    return Adam(lr = learning_rate, beta_1 = beta_1) 


#Build generator network
#Changed 'standard' generator final layer to 1 node rather than 28^2 as single number = image
def get_generator(optimizer):
    
    generator = Sequential()
    generator.add(Dense(256, input_dim=noise_dim, kernel_initializer=initializers.RandomNormal(stddev=0.02)))
    generator.add(LeakyReLU(0.2))
    generator.add(BatchNormalization(momentum=0.8))

    generator.add(Dense(256))
    generator.add(LeakyReLU(0.2))
    generator.add(BatchNormalization(momentum=0.8))

    generator.add(Dense(256))
    generator.add(LeakyReLU(0.2))
    generator.add(BatchNormalization(momentum=0.8))

    generator.add(Dense(256))
    generator.add(LeakyReLU(0.2))
    generator.add(BatchNormalization(momentum=0.8))

    generator.add(Dense(256))
    generator.add(LeakyReLU(0.2))
    generator.add(BatchNormalization(momentum=0.8))

    generator.add(Dense(256))
    generator.add(LeakyReLU(0.2))
    generator.add(BatchNormalization(momentum=0.8))

    generator.add(Dense(256))
    generator.add(LeakyReLU(0.2))
    generator.add(BatchNormalization(momentum=0.8))

    generator.add(Dense(256))
    generator.add(LeakyReLU(0.2))
    generator.add(BatchNormalization(momentum=0.8))

    generator.add(Dense(256))
    generator.add(LeakyReLU(0.2))
    generator.add(BatchNormalization(momentum=0.8))

    generator.add(Dense(data_dim, activation='tanh'))
    
#    generator = multi_gpu_model(generator, gpus=3)

    generator.compile(loss='binary_crossentropy', optimizer=optimizer)
    
    return generator

#Build discriminator layers network
#Changed input_dim to 1 (see above)
def get_discriminator(optimizer):
    
    discriminator = Sequential()
    
    discriminator.add(Dense(256, input_dim=data_dim, kernel_initializer=initializers.RandomNormal(stddev=0.02)))
    discriminator.add(LeakyReLU(0.2))
    discriminator.add(Dropout(0.3))
       
    discriminator.add(Dense(256))
    discriminator.add(LeakyReLU(0.2))
    discriminator.add(Dropout(0.3))
       
    discriminator.add(Dense(256))
    discriminator.add(LeakyReLU(0.2))
    discriminator.add(Dropout(0.3))

    discriminator.add(Dense(256))
    discriminator.add(LeakyReLU(0.2))
    discriminator.add(Dropout(0.3))

    discriminator.add(Dense(256))
    discriminator.add(LeakyReLU(0.2))
    discriminator.add(Dropout(0.3))

    discriminator.add(Dense(256))
    discriminator.add(LeakyReLU(0.2))
    discriminator.add(Dropout(0.3))

    discriminator.add(Dense(256))
    discriminator.add(LeakyReLU(0.2))
    discriminator.add(Dropout(0.3))

    discriminator.add(Dense(256))
    discriminator.add(LeakyReLU(0.2))
    discriminator.add(Dropout(0.3))

    discriminator.add(Dense(256))
    discriminator.add(LeakyReLU(0.2))
    discriminator.add(Dropout(0.3))

    discriminator.add(Dense(1, activation='sigmoid'))
    
#    discriminator = multi_gpu_model(discriminator, gpus=3)
    
    discriminator.compile(loss='binary_crossentropy', optimizer=optimizer)
    
    return discriminator


#Build/compile overall GAN network
def get_gan_network(discriminator, noise_dim, generator, optimizer):
    
    #Initially set trainable to False since only want to train generator or discriminator at a time
    discriminator.trainable = False
    
    #GAN input (noise) will be n-dimensional vectors (dimensions from noise_dim)
    gan_input = Input(shape=(noise_dim,))
    
    #Output of the generator (previously an image, hopefully now a single number?)
    x = generator(gan_input)
    
    #Get output of discriminator (probability if the image is real or not)
    gan_output = discriminator(x)
    
    gan = Model(inputs=gan_input, outputs=gan_output)
    
#    gan = multi_gpu_model(gan, gpus=3)

    gan.compile(loss='binary_crossentropy', optimizer=optimizer)
    
    return gan
    

def plot_examples(generated_vars, var_name, epoch, bin_no=400, x_range = None, y_range = None):
    
    fig1, ax1 = plt.subplots()
    ax1.cla()
    
    title = 'GAN7_generated_' + var_name + '_epoch_%d.eps'
    
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
     
    data_batch = x_test[np.random.randint(0, x_test.shape[0], size=examples)]
    gen_input = np.zeros((examples, noise_dim))
            
    phys_data = data_batch[:, len(DLLs):]
    rand_noise_dim = noise_dim - len(physical_vars)
    noise = np.random.normal(0, 1, size=[examples, rand_noise_dim])
            
    gen_input[:, :-len(physical_vars)] = noise
    gen_input[:, -len(physical_vars):] = phys_data
    
    generated_vars = generator.predict(gen_input)
    
    #Shift back to proper distribution?
    for i in range(generated_vars.shape[1]):
        
        generated_vars[:,i] = np.multiply(generated_vars[:,i], div_num[i])
        generated_vars[:,i] = np.add(generated_vars[:,i], shift[i])    
        
        if i<len(DLLs):
            plot_examples(generated_vars[:,i], 'DLL'+ DLLs[i], epoch)
        else:
            plot_examples(generated_vars[:,i], physical_vars[i-len(DLLs)], epoch)
        
        
#Training function. Import data, split into batches to train and train data, plotting data every plot_freq epochs 
def train(epochs=1, batch_size=128):
    
    print("Importing data...")
    #Get the training and testing data
    x_train, x_test, shift, div_num = get_x_data(DLLs, ref_particle, physical_vars, particle_source)
    print("Data imported")

    # Split the training data into batches of size 128
    batch_count = x_train.shape[0] // batch_size

    # Build GAN netowrk
    optimizer = get_optimizer()
    generator = get_generator(optimizer)
    discriminator = get_discriminator(optimizer)
    gan = get_gan_network(discriminator, noise_dim, generator, optimizer)

    discrim_loss_tot = []
    gen_loss_tot = []
    
    for i in range(1, epochs+1):
        
        print('-'*15, 'Epoch %d' % i, '-'*15)
        
        discrim_loss = []
        gen_loss = []
        
        for _ in tqdm(range(batch_count)):
            # Get a random set of input noise and images
            
            #To do: take (random?) set of physical variables and replace noise with them

            data_batch = x_train[np.random.randint(0, x_train.shape[0], size=batch_size)]
            gen_input = np.zeros((batch_size, noise_dim))
            
            phys_data = data_batch[:, len(DLLs):]
            rand_noise_dim = noise_dim - len(physical_vars)
            noise = np.random.normal(0, 1, size=[batch_size, rand_noise_dim])
            
            gen_input[:, :-len(physical_vars)] = noise
            gen_input[:, -len(physical_vars):] = phys_data            
                        
            #Generate fake data
            generated_data = generator.predict(gen_input)
            X = np.concatenate([data_batch, generated_data])
            
            #Labels for generated and real data
            y_dis = np.zeros(2*batch_size)
            
            #One-sided label smoothing
            y_dis[:batch_size] = 0.9
            
            #Train discriminator
            discriminator.trainable = True
            discrim_loss.append(discriminator.train_on_batch(X, y_dis))
            
            #Train generator
            data_batch = x_train[np.random.randint(0, x_train.shape[0], size=batch_size)]
            gen_input = np.zeros((batch_size, noise_dim))
            
            phys_data = data_batch[:, len(DLLs):]
            rand_noise_dim = noise_dim - len(physical_vars)
            noise = np.random.normal(0, 1, size=[batch_size, rand_noise_dim])
            
            gen_input[:, :-len(physical_vars)] = noise
            gen_input[:, -len(physical_vars):] = phys_data            
            
            y_gen = np.ones(batch_size)

            discriminator.trainable = False
            gen_loss.append(gan.train_on_batch(gen_input, y_gen))
        
        #Generate histogram via generator every plot_freq epochs
        if i == 1 or i % plot_freq == 0:
            gen_examples(x_test, i, generator, shift, div_num)
            
        gen_loss_tot = np.concatenate((gen_loss_tot, gen_loss))
        discrim_loss_tot = np.concatenate((discrim_loss_tot, discrim_loss))
            
    loss_num = batch_count * epochs
    epoch_arr = np.linspace(1,loss_num,num=loss_num)
    epoch_arr = np.divide(epoch_arr, batch_count)
    
    #Plot loss functions
    fig1, ax1 = plt.subplots()
    ax1.cla()
    ax1.plot(epoch_arr, discrim_loss_tot)
    ax1.plot(epoch_arr, gen_loss_tot)
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    
    generator.save('trained_gan.h5')  # creates a HDF5 file 'trained_gan.h5'

    ax1.legend(["Discriminator loss", "Generator loss"])
    fig1.savefig('GAN1_loss.eps', format='eps', dpi=2500)

#Call training function
if __name__ == '__main__':
    train(epochs, batch_size) #Epochs, batch size e.g. 400,128
    
#Measure total run time for script
t_final = time.time()
runtime = t_final - t_init
print("Total run time = ", runtime)

#Save runtime as text
with open('GAN1_runtime.txt', 'w') as f:
    print(runtime, file=f)
