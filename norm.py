#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 01:42:52 2019

@author: Elliott
"""

import numpy as np
import pandas as pd

particle_source = 'KAON'

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
def get_norm_info(particle_source):
    
    all_data = import_all_var(particle_source)

    x_data = np.array(all_data)

    x_data, shift, div_num = norm(x_data)
    
    output_info = np.zeros((2, x_data.shape[1]))
    
    output_info[0,:] = shift
    output_info[1,:] = div_num
    
    return output_info

output_info = get_norm_info(particle_source)

#Save norm details as text file
pd.DataFrame(output_info).to_csv(particle_source + '_norm.csv')

#with open(particle_source + '_norm.txt', 'w') as f:
#    print(shift, div_num, file=f)
