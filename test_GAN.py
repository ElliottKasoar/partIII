#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 14 16:50:16 2019

@author: Elliott
"""

#Import trained generator and produce plots. 
#Currently seems to reproduce plots from training i.e. distributions
#Next, look at correlations between distributions

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from keras.models import load_model


generator = load_model('../../GAN_data/GAN_7DLL/set13.1/trained_gan.h5')
plt.rcParams['agg.path.chunksize'] = 10000 #Needed for plotting lots of data?

#DLLs = ['e', 'mu', 'k', 'p', 'd', 'bt']
DLLs = ['e', 'mu', 'k', 'd', 'bt']
physical_vars = ['TrackP', 'TrackPt']
ref_particle = 'pi'
particle_source = 'KAON'

noise_dim = 100 #Dimension of random noise vector.
frac = 0.025
train_frac = 0.7
examples=500000


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



#Make scatter plot w/ colour of correlations between two variables (e.g. DLLs)
def col_scatt(var1, var2, var1_text, var2_text, max_var_index, x_range=None, y_range=None, zero_lines=0, save_index=0, size=1):
    
    x = var1[0:max_var_index]
    y = var2[0:max_var_index]

    # Calculate the point density
    xy = np.vstack([x,y])
    z = gaussian_kde(xy)(xy)
    
    # Sort the points by density, so that the densest points are plotted last
    idx = z.argsort()
    x, y, z = x[idx], y[idx], z[idx]
    
    #Scatter plot with Gaussian colour scale
    fig1, ax1 = plt.subplots()
    ax1.cla()
    ax1.scatter(x, y, c=z, s=size, edgecolor='')
    ax1.set_xlabel(var1_text)
    ax1.set_ylabel(var2_text)
    ax1.set_xlim(x_range)
    ax1.set_ylim(y_range)
    
    if(zero_lines):
        ax1.axhline(lw=1.0, color='k',ls='--')
        ax1.axvline(lw=1.0, color='k',ls='--')
    
    if(max_var_index >= save_index):
        fig1.savefig(var1_text + "_" + var2_text + "_colour.eps", format='eps', dpi=1000)
    


def plot_examples(generated_vars, var_name, bin_no=400, x_range = None, y_range = None):
    
    fig1, ax1 = plt.subplots()
    ax1.cla()
    
    title = 'GAN7_generated_' + var_name + '_trained_.eps'
    
    if y_range is not None:
        ax1.set_ylim(bottom = 0, top = y_range)
    
    if x_range is not None:
        ax1.set_xlim(x_range)
    
    ax1.set_xlabel(var_name)
    ax1.set_ylabel("Number of events")
    ax1.hist(generated_vars, bins=bin_no, range=x_range)
    
    fig1.savefig(title, format='eps', dpi=2500)

print("Importing data...")
#Get the training and testing data
x_train, x_test, shift, div_num = get_x_data(DLLs, ref_particle, physical_vars, particle_source)
print("Data imported")

print("Generating data...")
data_batch = x_test[np.random.randint(0, x_test.shape[0], size=examples)]
gen_input = np.zeros((examples, noise_dim))
            
phys_data = data_batch[:, len(DLLs):]
rand_noise_dim = noise_dim - len(physical_vars)
noise = np.random.normal(0, 1, size=[examples, rand_noise_dim])
            
gen_input[:, :-len(physical_vars)] = noise
gen_input[:, -len(physical_vars):] = phys_data
    
generated_vars = generator.predict(gen_input)
print("Data generated")

print("Plotting data...")
#Shift back to proper distribution?
for i in range(generated_vars.shape[1]):        
    generated_vars[:,i] = np.multiply(generated_vars[:,i], div_num[i])
    generated_vars[:,i] = np.add(generated_vars[:,i], shift[i])
        
    if i<len(DLLs):
        plot_examples(generated_vars[:,i], 'DLL'+ DLLs[i])
    else:
        plot_examples(generated_vars[:,i], physical_vars[i-len(DLLs)])


max_var_index = 10000
col_scatt(generated_vars[:,2], generated_vars[:,3], "DLLk_k", "DLLd_k", max_var_index, x_range=[-100,100], y_range=[-80,80], zero_lines=1, save_index=10000, size=0.5)

print("Data plotted")
