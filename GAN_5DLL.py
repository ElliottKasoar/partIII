#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  3 17:18:54 2019

@author: Elliott
"""

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

#import tensorflow as tf
#import keras.backend as K

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
batch_size = 64
epochs = 1000

noise_dim = 250 #Dimension of random noise vector. 

learning_rate = 0.0001
beta_1=0.5
    
#Current numbers chosen s.t.:
#Have 10,000,000 data points for each DLL (x5)
#Each row will be set of 10,000 points so array will be 5000 x 10,000
arr_size = 1000
tot_arr_size = 5*1000
sample_size = 10000

plot_freq = 50 #Plot data for after this number of epochs

#Using tensorflow backend
os.environ["KERAS_BACKEND"] = "tensorflow"
#So reproducable
np.random.seed(10)


#Basically get rid of this, replace with importing my data
#Kept for now as reference..
#def load_mnist_data():
#    # load the data
#    (x_train, y_train), (x_test, y_test) = mnist.load_data()
#    # normalize our inputs to be in the range[-1, 1] 
#    x_train = (x_train.astype(np.float32) - 127.5)/127.5
#    # convert x_train with a shape of (60000, 28, 28) to (60000, 784) so we have
#    # 784 columns per row
#    x_train = x_train.reshape(60000, 784)
#    return (x_train, y_train, x_test, y_test)


def get_data(var_type, particle_source):
    #Import data from kaons and pions
    datafile_kaon = '../Data/PID-train-data-KAONS.hdf' 
    data_kaon = pd.read_hdf(datafile_kaon, 'KAONS') 
    #print(data_kaon.columns)

    datafile_pion = '../Data/PID-train-data-PIONS.hdf' 
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
        DLL = get_data('RichDLL' + DLL_part_1, particle_source)
    else:
        DLL_1 = get_data('RichDLL' + DLL_part_1, particle_source)
        DLL_2 = get_data('RichDLL' + DLL_part_2, particle_source)
        DLL = change_DLL(DLL_1, DLL_2)
    
    return DLL


def get_x_data(DLL_part_1, DLL_part_2, particle_source_1):

    #Get DLL data
    x_data = np.array(get_data('RichDLL' + DLL_part_1, particle_source_1))

    #Use subset of data
    frac = 1
    tot_split = int(frac * len(x_data))
    x_train = x_data[:tot_split]
        
    #Change from (n,) to (n,1) numpy array
    x_train = x_train.reshape(len(x_train),1)

    return x_train

def get_x_sample(x, sample_size=sample_size):

    x_batch = x[np.random.randint(0, x.shape[0], size=sample_size)]      

    return x_batch

def norm(x):

    x_max = np.max(x)
    x_min = np.min(x)
    
    shift = (x_max + x_min)/2
    x = np.subtract(x, shift)
    
    div_num = x_max - shift
    x = np.divide(x, div_num)
    
    return x, shift, div_num

#Get all data from data files, shuffle the data for each DLL, then split into groups of size sample size
#Also rearrange into arr_size rows, before combining all DLLs and shuffling these rows
#Finally, normalise based on largest value present s.t. all values in range [-1,1]
def get_training_data(arr_size, sample_size):
    
    x_train = np.zeros([tot_arr_size, sample_size])
    
    #Get the training and testing data: CHANGE something????????
    x_train_e = get_x_data('e', 'pi', 'KAON')
    x_train_k = get_x_data('k', 'pi', 'KAON')
    x_train_p = get_x_data('p', 'pi', 'KAON')
    x_train_d = get_x_data('d', 'pi', 'KAON')
    x_train_bt = get_x_data('bt', 'pi', 'KAON')
    
    np.random.shuffle(x_train_e)
    np.random.shuffle(x_train_k)
    np.random.shuffle(x_train_p)
    np.random.shuffle(x_train_d)
    np.random.shuffle(x_train_bt)
    
    x_train_e = np.reshape(x_train_e, [arr_size, sample_size])
    x_train_k = np.reshape(x_train_k, [arr_size, sample_size])
    x_train_p = np.reshape(x_train_p, [arr_size, sample_size])
    x_train_d = np.reshape(x_train_d, [arr_size, sample_size])
    x_train_bt = np.reshape(x_train_bt, [arr_size, sample_size])
    
    x_train = np.concatenate((x_train_e, x_train_k, x_train_p, x_train_d, x_train_bt))
    np.random.shuffle(x_train)
        
#    x_sample = get_x_sample(x_train, sample_size)
#    x_sample, shift, div_num = norm(x_sample)
#    batch_count = x_sample.shape[0] // batch_size

    x_train, shift, div_num = norm(x_train)    
    
    return x_train, shift, div_num


#FINE?
def get_optimizer():
    return Adam(lr = learning_rate, beta_1=beta_1) 

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

    generator.add(Dense(sample_size, activation='tanh'))
#    generator.compile(loss='binary_crossentropy', optimizer=optimizer)
    
    return generator

#FINE?
def get_discriminator(optimizer):
    discriminator = Sequential()
    
    discriminator.add(Dense(1024, input_dim=sample_size, kernel_initializer='he_normal'))
    discriminator.add(LeakyReLU(0.2))
    discriminator.add(Dropout(0.3))
    
    
    discriminator.add(Dense(512, kernel_initializer='he_normal'))
    discriminator.add(LeakyReLU(0.2))
    discriminator.add(Dropout(0.3))

    discriminator.add(Dense(256, kernel_initializer='he_normal'))
    discriminator.add(LeakyReLU(0.2))
    discriminator.add(Dropout(0.3))

    discriminator.add(Dense(1, kernel_initializer='he_normal'))
#    discriminator.compile(loss='binary_crossentropy', optimizer=optimizer)
    
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


#Can probably remove? Kept for ref for now
#def plot_generated_images(epoch, generator, examples=100, dim=(10, 10), figsize=(10, 10)):
#    noise = np.random.normal(0, 1, size=[examples, noise_dim])
#    generated_images = generator.predict(noise)
#    generated_images = generated_images.reshape(examples, 28, 28)
#
#    plt.figure(figsize=figsize)
#    for i in range(generated_images.shape[0]):
#        plt.subplot(dim[0], dim[1], i+1)
#        plt.imshow(generated_images[i], interpolation='nearest', cmap='gray_r')
#        plt.axis('off')
#    plt.tight_layout()
#    plt.savefig('gan_generated_image_epoch_%d.png' % epoch)
    
 #If want more than one example probably need ^^
def plot_hist(epoch, generator, shift, div_num, bin_no=100, x_range = None, y_range = None, examples=1):
     
    #y_range e.g. 300, x_range e.g. (-100,100)
    
    noise = np.random.normal(0, 1, size=[examples, noise_dim])
    generated_numbers = generator.predict(noise)
    
    #Shift back to proper distribution?
    generated_numbers = np.multiply(generated_numbers, div_num)
        
    generated_numbers = np.add(generated_numbers, shift)    
    generated_numbers = np.reshape(generated_numbers, [sample_size,1])
    fig1, ax1 = plt.subplots()
    ax1.cla()
    if y_range is not None:
        ax1.set_ylim(bottom = 0, top = y_range)
    if x_range is not None:
        ax1.set_xlim(x_range)
    ax1.set_xlabel("DLL")
    ax1.set_ylabel("Number of events")
    ax1.hist(generated_numbers, bins=bin_no)
    fig1.savefig('GAN5_gan_generated_data_epoch_%d.eps' % epoch, format='eps', dpi=2500)
        
#Needs changing - change training/test data source
def train(epochs=1, batch_size=128):
    
    print("Importing data...")    
    x_train, shift, div_num = get_training_data(arr_size, sample_size)
    print("Data imported")
  
    # Split the training data into batches of size (batch_size =) 128
    batch_count = x_train.shape[0] // batch_size

    # Build GAN netowrk
    optimizer = get_optimizer()
    generator = get_generator(optimizer)
    discriminator = get_discriminator(optimizer)
    gan = get_gan_network(discriminator, noise_dim, generator, optimizer)

    for i in range(1, epochs+1):
        print('-'*15, 'Epoch %d' % i, '-'*15)
        for _ in tqdm(range(batch_count)):
            # Get a random set of input noise and images
            noise = np.random.normal(0, 1, size=[batch_size, noise_dim])
            data_batch = x_train[np.random.randint(0, x_train.shape[0], size=batch_size)]
            
            #Generate fake data
            generated_data = generator.predict(noise)
            X = np.concatenate([data_batch, generated_data])
            
            #Labels for generated and real data
            y_dis = np.zeros(2*batch_size)
            
            #One-sided label smoothing?
            y_dis[:batch_size] = 0.9
            
            #Train discriminator
            discriminator.trainable = True
            discriminator.train_on_batch(X, y_dis)
            
            #Train generator
            noise = np.random.normal(0, 1, size=[batch_size, noise_dim])
            y_gen = np.ones(batch_size)
            discriminator.trainable = False
            gan.train_on_batch(noise, y_gen)
            
        if i == 1 or i % plot_freq == 0:
            plot_hist(i, generator, shift, div_num)
#            plot_generated_images(i, generator)
            
if __name__ == '__main__':
    train(epochs, batch_size) #Epochs, batch size e.g. 400,128
    
#Measure total run time for script
t_final = time.time()
print("Total run time = ", t_final - t_init)
