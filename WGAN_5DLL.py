#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  3 17:19:23 2019

@author: Elliott
"""

#Now more based on https://github.com/keras-team/keras-contrib/blob/master/examples/improved_wgan.py
#With ref to WGAN described in https://arxiv.org/abs/1704.00028

import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import tensorflow as tf

import keras.backend as K
from keras.layers import Input, BatchNormalization
from keras.models import Model, Sequential
from keras.layers.core import Dense, Dropout
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam
from keras import initializers
from keras.layers.merge import _Merge

#import argparse
import pandas as pd
from functools import partial
import time

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

learning_rate = 0.0001
beta_1=0.5
beta_2=0.999
    
#Current numbers chosen s.t.:
#Have 10,000,000 data points for each DLL (x5)
#Each row will be set of 10,000 points so array will be 5000 x 10,000
arr_size = 1000
tot_arr_size = 5*1000
sample_size = 10000

noise_dim = 100 #Dimension of random noise vector. 

training_ratio = 5  # The training ratio is the number of discriminator updates per generator update. The paper uses 5.
gradient_penalty_weight = 10  # As per the paper

plot_freq = 50 #Plot data for after this number of epochs

#Using tensorflow backend
os.environ["KERAS_BACKEND"] = "tensorflow"
#So reproducable
np.random.seed(10)

def get_data(var_type, particle_source):
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

def get_x_sample(x, sample_size):

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


#Not sure about these functions...
################################################################################################ 

def customLoss(yTrue,yPred):
    return K.sum(K.log(yTrue) - K.log(yPred))

def cramer_critic(x, y, discriminator):
    
    discriminated_x = discriminator(x)
    
    return tf.norm(discriminated_x - discriminator(y), axis=1) - tf.norm(discriminated_x, axis=1)


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

def gradient_penalty_loss(y_true, y_pred, averaged_samples, gradient_penalty_weight):
    
    """Calculates the gradient penalty loss for a batch of "averaged" samples.
    In Improved WGANs, the 1-Lipschitz constraint is enforced by adding a term to the loss function
    that penalizes the network if the gradient norm moves away from 1. However, it is impossible to evaluate
    this function at all points in the input space. The compromise used in the paper is to choose random points
    on the lines between real and generated samples, and check the gradients at these points. Note that it is the
    gradient w.r.t. the input averaged samples, not the weights of the discriminator, that we're penalizing!
    In order to evaluate the gradients, we must first run samples through the generator and evaluate the loss.
    Then we get the gradients of the discriminator w.r.t. the input averaged samples.
    The l2 norm and penalty can then be calculated for this gradient.
    Note that this loss function requires the original averaged samples as input, but Keras only supports passing
    y_true and y_pred to loss functions. To get around this, we make a partial() of the function with the
    averaged_samples argument, and use that for model training."""
    
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

################################################################################################ 


#FINE?
def get_optimizer():
    return Adam(lr = learning_rate, beta_1 = beta_1, beta_2 = beta_2) 

#FINE? Changed final layer to 1 node rather than 28^2
#Tanh actuvation for final layer s.t. output lies in range [-1, 1] like the training data
def get_generator():
    generator = Sequential()
    generator.add(Dense(1024, input_dim=noise_dim, kernel_initializer=initializers.RandomNormal(stddev=0.02)))
    generator.add(LeakyReLU(0.2))
    generator.add(BatchNormalization(momentum=0.8))

    generator.add(Dense(512))
    generator.add(LeakyReLU(0.2))
    generator.add(BatchNormalization(momentum=0.8))

    generator.add(Dense(256))
    generator.add(LeakyReLU(0.2))
    generator.add(BatchNormalization(momentum=0.8))

    generator.add(Dense(sample_size, activation='tanh'))
    
    return generator

#FINE?
#Note batch norm not used for this as per paper
def get_discriminator():
    discriminator = Sequential()
    
    discriminator.add(Dense(1024, input_dim=sample_size, kernel_initializer='he_normal'))
    discriminator.add(LeakyReLU(0.2))
    discriminator.add(Dropout(0.3))
    
    discriminator.add(Dense(512, kernel_initializer='he_normal'))
    discriminator.add(LeakyReLU(0.2))
    discriminator.add(Dropout(0.3))

    discriminator.add(Dense(256, kernel_initializer='he_normal'))
    discriminator.add(LeakyReLU())
    discriminator.add(Dropout(0.3))

    discriminator.add(Dense(1, kernel_initializer='he_normal'))
    
    return discriminator
    
 #If want more than one example probably need ^^
 #Number of examples given by examples...
def plot_hist(epoch, generator, shift, div_num, bin_no=100, x_range = None, y_range = None, examples=1):
     
    #y_range e.g. 300, x_range e.g. (-100,100)
    
    noise = np.random.normal(0, 1, size=[examples, noise_dim])
    generated_numbers = generator.predict(noise)
    
    #Shift back to proper distribution and reshape to be plotted
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
    fig1.savefig('gan_generated_data_epoch_%d.eps' % epoch, format='eps', dpi=2500)
     
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


#Need to understand this better
class RandomWeightedAverage(_Merge):
    """Takes a randomly-weighted average of two tensors. In geometric terms, this outputs a random point on the line
    between each pair of input points.
    Inheriting from _Merge is a little messy but it was the quickest solution I could think of.
    Improvements appreciated."""

    def _merge_function(self, inputs):
        weights = K.random_uniform((batch_size, 1, 1, 1))
        return (weights * inputs[0]) + ((1 - weights) * inputs[1])


def train(epochs=5, batch_size=128):
    
    x_train, shift, div_num = get_training_data(arr_size, sample_size)
    print("Data imported")
    
    # Build GAN netowrk
    optimizer = get_optimizer()
    generator = get_generator()
    discriminator = get_discriminator()
    
    # The generator_model is used when we want to train the generator layers.
    # As such, we ensure that the discriminator layers are not trainable.
    # Note that once we compile this model, updating .trainable will have no effect within it. As such, it
    # won't cause problems if we later set discriminator.trainable = True for the discriminator_model, as long
    # as we compile the generator_model first.
    for layer in discriminator.layers:
        layer.trainable = False
    
    for layer in generator.layers:
        layer.trainable = True    
    
    discriminator.trainable = False
    generator.trainable = True
    
    generator_input = Input(shape=(noise_dim,))
    
    generator_layers = generator(generator_input)
    
    discriminator_layers_for_generator = discriminator(generator_layers)
    
    generator_model = Model(inputs=[generator_input], outputs=[discriminator_layers_for_generator])
    
    #Use the Adam paramaters from Gulrajani et al.
    generator_model.compile(optimizer, loss=wasserstein_loss)

    # Now that the generator_model is compiled, we can make the discriminator layers trainable.
    for layer in discriminator.layers:
        layer.trainable = True
        
    for layer in generator.layers:
        layer.trainable = False
    
    discriminator.trainable = True
    generator.trainable = False

    # The discriminator_model is more complex. It takes both real image samples and random noise seeds as input.
    # The noise seed is run through the generator model to get generated images. Both real and generated images
    # are then run through the discriminator. Although we could concatenate the real and generated images into a
    # single tensor, we don't (see model compilation for why).
    real_samples = Input(shape=x_train.shape[1:])
    
    generator_input_for_discriminator = Input(shape=(noise_dim,))
    
    generated_samples_for_discriminator = generator(generator_input_for_discriminator)
    
    discriminator_output_from_generator = discriminator(generated_samples_for_discriminator)
    
    discriminator_output_from_real_samples = discriminator(real_samples)

    # We also need to generate weighted-averages of real and generated samples, to use for the gradient norm penalty.
    averaged_samples = RandomWeightedAverage()([real_samples, generated_samples_for_discriminator])

    # We then run these samples through the discriminator as well. Note that we never really use the discriminator
    # output for these samples - we're only running them to get the gradient norm for the gradient penalty loss.
    averaged_samples_out = discriminator(averaged_samples)

    # The gradient penalty loss function requires the input averaged samples to get gradients. However,
    # Keras loss functions can only have two arguments, y_true and y_pred. We get around this by making a partial()
    # of the function with the averaged samples here.
    partial_gp_loss = partial(gradient_penalty_loss, averaged_samples=averaged_samples, gradient_penalty_weight=gradient_penalty_weight) 
    partial_gp_loss.__name__ = 'gradient_penalty'  # Functions need names or Keras will throw an error

    # Keras requires that inputs and outputs have the same number of samples. This is why we didn't concatenate the
    # real samples and generated samples before passing them to the discriminator: If we had, it would create an
    # output with 2 * BATCH_SIZE samples, while the output of the "averaged" samples for gradient penalty
    # would have only BATCH_SIZE samples.

    # If we don't concatenate the real and generated samples, however, we get three outputs: One of the generated
    # samples, one of the real samples, and one of the averaged samples, all of size BATCH_SIZE. This works neatly!
    discriminator_model = Model(inputs=[real_samples, generator_input_for_discriminator], outputs=[discriminator_output_from_real_samples, discriminator_output_from_generator, averaged_samples_out])

    # We use the Adam paramaters from Gulrajani et al. We use the Wasserstein loss for both the real and generated
    # samples, and the gradient penalty loss for the averaged samples.
    discriminator_model.compile(optimizer, loss=[wasserstein_loss, wasserstein_loss, partial_gp_loss])

    # We make three label vectors for training. positive_y is the label vector for real samples, with value 1.
    # negative_y is the label vector for generated samples, with value -1. The dummy_y vector is passed to the
    # gradient_penalty loss function and is not used.
    positive_y = np.ones((batch_size, 1), dtype=np.float32)
    negative_y = -positive_y
    dummy_y = np.zeros((batch_size, 1), dtype=np.float32)

    generator_loss_tot = []
    for i in range(1, epochs+1):
        np.random.shuffle(x_train) #Not needed anymore?
        print("Epoch: ", i)
        print("Number of batches: ", int(x_train.shape[0] // batch_size))
        discriminator_loss = []
        generator_loss = []
        minibatches_size = batch_size * training_ratio
        
        for j in tqdm(range(int(x_train.shape[0] // (batch_size * training_ratio)))):
            
            discriminator_minibatches = x_train[j * minibatches_size:(j + 1) * minibatches_size]
        
            for k in tqdm(range(training_ratio)):
        
                data_batch = discriminator_minibatches[k * batch_size:(k + 1) * batch_size]
                noise = np.random.rand(batch_size, noise_dim).astype(np.float32)
                discriminator_loss.append(discriminator_model.train_on_batch([data_batch, noise], [positive_y, negative_y, dummy_y]))
            
            generator_loss.append(generator_model.train_on_batch(np.random.rand(batch_size, noise_dim), positive_y))
            
        if i == 1 or i % plot_freq == 0:
            plot_hist(i, generator, shift, div_num)
          
        generator_loss_tot = np.concatenate((generator_loss_tot, generator_loss))
        # Plot the progress
#        print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (i, discriminator_loss[0], 100*discriminator_loss[1], generator_loss))
        
#        print(generator_loss)
    num = x_train.shape[0] // (batch_size * training_ratio) * epochs
    epoch_arr = np.linspace(1,num,num=num)
    
    fig1, ax1 = plt.subplots()
    ax1.cla()
    ax1.plot(epoch_arr, generator_loss_tot)
    fig1.savefig('gen_loss.eps', format='eps', dpi=2500)


if __name__ == '__main__':
    train(epochs, batch_size) #Epochs, batch size e.g. 400,128

#Measure total run time for script
t_final = time.time()
runtime = t_final - t_init
print("Total run time = ", runtime)

with open('runtime.txt', 'w') as f:
    print(runtime, file=f)