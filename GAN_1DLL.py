#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  3 17:10:01 2019

@author: Elliott
"""

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

#from matplotlib.ticker import AutoMinorLocator
#from scipy.stats import gaussian_kde
#import math
#from sklearn.preprocessing import QuantileTransformer

#Time total run
t_init = time.time()

plt.rcParams['agg.path.chunksize'] = 10000 #Needed for plotting lots of data?

#Some tunable variables/parameters...
#Not really passed properly
batch_size = 256
epochs = 500

learning_rate = 0.0002
beta_1=0.5

noise_dim = 200 #Dimension of random noise vector. 
plot_freq = 50 #Plot data for after this number of epochs

#Using tensorflow backend
os.environ["KERAS_BACKEND"] = "tensorflow"
#So reproducable
np.random.seed(10)

#Dimension of random noise vector
#NOTE: not passed properly
noise_dim = 200

frac = 0.1
train_frac = 0.7

DLL_part_1 = 'k'
DLL_part_2 = 'pi'
particle_source = 'KAON'


def import_data(var_type, particle_source):
    #Import data from kaons and pions
    datafile_kaon = '../data/PID-train-data-KAONS.hdf' 
    data_kaon = pd.read_hdf(datafile_kaon, 'KAONS') 
    #print(data_kaon.columns)

    datafile_pion = '../data/PID-train-data-PIONS.hdf' 
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


def get_DLL(DLL_part_1, DLL_part_2, particle_source):
        
    #Get data for DLLs including changing if the DLL is not x-pi
    if(DLL_part_2 == 'pi'):
        DLL = import_data('RichDLL' + DLL_part_1, particle_source)
    else:
        DLL_1 = import_data('RichDLL' + DLL_part_1, particle_source)
        DLL_2 = import_data('RichDLL' + DLL_part_2, particle_source)
        DLL = change_DLL(DLL_1, DLL_2)
    
    return DLL


def get_x_data(DLL_part_1, DLL_part_2, particle_source):

    #Get DLL data
    x_data = np.array(import_data('RichDLL' + DLL_part_1, particle_source))

    #Use subset of data
    tot_split = int(frac * len(x_data))
    x_data = x_data[:tot_split]
    
    #Now split into training/test data 70/30?
    split = int(train_frac * len(x_data))
    x_train = x_data[:split]
    x_test = x_data[split:]
    
    #Change from (n,) to (n,1) numpy array
    x_train = x_train.reshape(len(x_train),1)
    x_test = x_test.reshape(len(x_test),1)

    x_train, shift, div_num = norm(x_train)

    return x_train, x_test, shift, div_num 


def norm(x):
    x_max = np.max(x)
    x_min = np.min(x)
    
    shift = (x_max + x_min)/2
    x = np.subtract(x, shift)
    
    div_num = x_max - shift
    x = np.divide(x, div_num)
    
    return x, shift, div_num


#FINE?
def get_optimizer():
    return Adam(lr = learning_rate, beta_1 = beta_1) 

#FINE? CHanged final layer to 1 node rather than 28^2
def get_generator(optimizer):
    generator = Sequential()
    generator.add(Dense(256, input_dim=noise_dim, kernel_initializer=initializers.RandomNormal(stddev=0.02)))
    generator.add(LeakyReLU(0.2))
    generator.add(BatchNormalization(momentum=0.8))

    generator.add(Dense(512))
    generator.add(LeakyReLU(0.2))
    generator.add(BatchNormalization(momentum=0.8))

    generator.add(Dense(1024))
    generator.add(LeakyReLU(0.2))
    generator.add(BatchNormalization(momentum=0.8))

    generator.add(Dense(1, activation='tanh'))
    generator.compile(loss='binary_crossentropy', optimizer=optimizer)
    
    return generator

#FINE?
def get_discriminator(optimizer):
    discriminator = Sequential()
    
    discriminator.add(Dense(1024, input_dim=1, kernel_initializer=initializers.RandomNormal(stddev=0.02)))
    discriminator.add(LeakyReLU(0.2))
    discriminator.add(Dropout(0.3))
    
    discriminator.add(Dense(512))
    discriminator.add(LeakyReLU(0.2))
    discriminator.add(Dropout(0.3))

    discriminator.add(Dense(256))
    discriminator.add(LeakyReLU(0.2))
    discriminator.add(Dropout(0.3))

    discriminator.add(Dense(1, activation='sigmoid'))
    discriminator.compile(loss='binary_crossentropy', optimizer=optimizer)
    
    return discriminator


#NEEDS CHANGING? Might be fine...
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
    
    gan.compile(loss='binary_crossentropy', optimizer=optimizer)
    
    return gan
    
 
def plot_hist(epoch, generator, shift, div_num, bin_no=200, x_range = None, y_range = None, examples=50000):
     
    #y_range e.g. 300, x_range e.g. (-100,100)
    
    noise = np.random.normal(0, 1, size=[examples, noise_dim])
    generated_numbers = generator.predict(noise)
    
    #Shift back to proper distribution?
    generated_numbers = np.multiply(generated_numbers, div_num)
    generated_numbers = np.add(generated_numbers, shift)    
    
    fig1, ax1 = plt.subplots()
    ax1.cla()
    if y_range is not None:
        ax1.set_ylim(bottom = 0, top = y_range)
    if x_range is not None:
        ax1.set_xlim(x_range)
    ax1.set_xlabel("DLL")
    ax1.set_ylabel("Number of events")
    ax1.hist(generated_numbers, bins=bin_no)
    fig1.savefig('GAN1_gan_generated_data_epoch_%d.eps' % epoch, format='eps', dpi=2500)
        
#Needs changing - change training/test data source
def train(epochs=1, batch_size=128):
    
    print("Importing data...")
    #Get the training and testing data: CHANGE
    x_train, x_test, shift, div_num = get_x_data(DLL_part_1, DLL_part_2, particle_source)
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
            noise = np.random.normal(0, 1, size=[batch_size, noise_dim])
            data_batch = x_train[np.random.randint(0, x_train.shape[0], size=batch_size)]
            
            #Generate fake data
            generated_data = generator.predict(noise)
            X = np.concatenate([data_batch, generated_data])
            
            #Labels for generated and real data
            y_dis = np.zeros(2*batch_size)
            
            #One-sided label smoothing
            y_dis[:batch_size] = 0.9
            
            #Train discriminator
            discriminator.trainable = True
            discrim_loss.append(discriminator.train_on_batch(X, y_dis))
            
            #Train generator
            noise = np.random.normal(0, 1, size=[batch_size, noise_dim])
            y_gen = np.ones(batch_size)
            discriminator.trainable = False
            gen_loss.append(gan.train_on_batch(noise, y_gen))
            
        if i == 1 or i % plot_freq == 0:
            plot_hist(i, generator, shift, div_num)
            
        gen_loss_tot = np.concatenate((gen_loss_tot, gen_loss))
        discrim_loss_tot = np.concatenate((discrim_loss_tot, discrim_loss))
            
    loss_num = batch_count * epochs
    epoch_arr = np.linspace(1,loss_num,num=loss_num)
    epoch_arr = np.divide(epoch_arr, batch_count)
    
    fig1, ax1 = plt.subplots()
    ax1.cla()
    ax1.plot(epoch_arr, discrim_loss_tot)
    ax1.plot(epoch_arr, gen_loss_tot)
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    
    ax1.legend(["Discriminator loss", "Generator loss"])
    fig1.savefig('GAN1_loss.eps', format='eps', dpi=2500)


if __name__ == '__main__':
    train(epochs, batch_size) #Epochs, batch size e.g. 400,128
    
#Measure total run time for script
t_final = time.time()
print("Total run time = ", t_final - t_init)

with open('runtime.txt', 'w') as f:
    print(runtime, file=f)
