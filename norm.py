#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
#Script to calculate normalisation variables: shift and div_num
#Select particle source (kaon or pion) and values needed to normalise each variable individually is saved to a csv

import numpy as np
import pandas as pd

particle_source = 'KAON' #KAON or PION

#Import all data via pandas from data files
#Inputs: Particle source e.g. KAONS corresponding to the datafile from which data will be imported
#Returns: pandas structure containing all variables from the source
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


#Normalise relevant data via dividing centre on zero and divide by max s.t. range=[-1,1]
#Input: Data array to be normalised (x) and particle source, so know which set of normalisation values to use
#Returns: Normalised data array (x) and shift/div_num used to do so (so can save and unnormalise later)
def norm(x):
    
    columns = x.shape[1]
    
    shift = np.zeros(columns)
    div_num = np.zeros(columns)
    
    #Loop over all columns (variables) in datafile
    for i in range(columns):
        
        #Calcualate the min and max of the column
        x_max = np.max(x[:,i])
        x_min = np.min(x[:,i])
    
        #Shift the data by the average of the min and max to centre the data on 0
        shift[i] = (x_max + x_min)/2
        x[:,i] = np.subtract(x[:,i], shift[i])
        
        #Divide by the new maximum to reduce the values to between -1 and 1
        if x_max == x_min:
            div_num[i] = 1
        else:
            div_num[i] = x_max - shift[i]
            x[:,i] = np.divide(x[:,i], div_num[i])
    
    return x, shift, div_num


#Normalise all data in a chosen datafile (kaon or pion) and save required values to do so
#Inputs: Particle source, to access relevent datafile
#Returns: Numpy array containing shift and div_num required to normalise each column of the datafile
def get_norm_info(particle_source):
    
    all_data = import_all_var(particle_source)

    x_data = np.array(all_data)

    x_data, shift, div_num = norm(x_data)
    
    output_info = np.zeros((2, x_data.shape[1]))
    
    output_info[0,:] = shift
    output_info[1,:] = div_num
    
    return output_info

output_info = get_norm_info(particle_source) #Get normalisation information to be saved

#Save norm details as text file
pd.DataFrame(output_info).to_csv(particle_source + '_norm.csv')
