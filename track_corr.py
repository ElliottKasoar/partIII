#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 28 19:03:14 2019

@author: Elliott
"""
#RichDLLe', 'RichDLLmu','RichDLLk', 'RichDLLp', 'RichDLLd', 'RichDLLbt'

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.spatial.distance import cdist
from tqdm import tqdm

#Time total run
t_init = time.time()

#help(data_kaon) 
#help(data_kaon.values)

plt.rcParams['agg.path.chunksize'] = 10000 #Needed for plotting lots of data?

##Import data from kaons and pions
#datafile_kaon = '../data/PID-train-data-KAONS.hdf' 
#data_kaon = pd.read_hdf(datafile_kaon, 'KAONS') 
#print(data_kaon.columns)
#
#datafile_pion = '../data/PID-train-data-PIONS.hdf' 
#data_pion = pd.read_hdf(datafile_pion, 'PIONS') 
#print(data_pion.columns)

frac = 0.1
train_frac = 0.7

gen_input_dim = 200

#DLL(DLL[i] - ref_particle) from particle_source data
DLLs = ['e', 'mu', 'k', 'p', 'd', 'bt']

physical_vars = ['TrackP', 'TrackPt', 'NumLongTracks', 'NumPVs', 'TrackVertexX', 'TrackVertexY', 'TrackVertexZ', 
                 'TrackRich1EntryX', 'TrackRich1EntryY', 'TrackRich1EntryZ', 'TrackRich1ExitX', 'TrackRich1ExitY', 
                 'TrackRich1ExitZ', 'TrackRich2EntryX', 'TrackRich2EntryY', 'TrackRich2EntryZ', 'TrackRich2ExitX', 
                 'TrackRich2ExitY', 'TrackRich2ExitZ']

include_z = False #Slightly faster if false (all Z coords same for RICH1Entry etc.)

corr_vars = []
nearest_neighbours = 3
for i in range(nearest_neighbours):
    corr_vars.append('RICH1EntryDist' + str(i))
for i in range(nearest_neighbours):
    corr_vars.append('RICH1ExitDist' + str(i))
for i in range(nearest_neighbours):
    corr_vars.append('RICH2EntryDist' + str(i))
for i in range(nearest_neighbours):
    corr_vars.append('RICH2ExitDist' + str(i))

corr_vars.append('RICH1ConeNum')
corr_vars.append('RICH2ConeNum')

max_track_rad = 200

#corr_vars = ['RICH1EntryDist', 'RICH1ExitDist', 'RICH2EntryDist', 'RICH2ExitDist']

ref_particle = 'pi'
particle_source = 'PION'

phys_dim = len(physical_vars)
DLLs_dim = len(DLLs)
corr_dim = len(corr_vars)
data_dim = DLLs_dim + phys_dim
noise_dim = gen_input_dim - phys_dim


###############################################################################
#Basic data manipulation e.g. selecting columns of data and changing DLLs

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


def nsmall(a, n):
    return np.partition(a, n)[n]


all_data = import_all_var(particle_source)

RunNum_list = all_data.RunNumber.unique()
num_RunNum = RunNum_list.shape[0]

#Get all data with same event number and run number
for i in tqdm(range(num_RunNum)):
    
    data_with_RunNum = all_data.loc[all_data['RunNumber'] == RunNum_list[i]]
    
    EventNum_list = data_with_RunNum.EventNumber.unique()

    num_EventNum = EventNum_list.shape[0]
    
    for j in range(num_EventNum):
  
        data_with_EventNum = data_with_RunNum.loc[data_with_RunNum['EventNumber'] == EventNum_list[j]]
        
        if include_z:
            data = data_with_EventNum.loc[:, 'TrackRich1EntryX':'TrackRich2ExitZ']
        else:
            data = data_with_EventNum.loc[:, ['TrackRich1EntryX', 'TrackRich1EntryY', 'TrackRich1ExitX', 
                                          'TrackRich1ExitY', 'TrackRich2EntryX', 'TrackRich2EntryY', 
                                          'TrackRich2ExitX', 'TrackRich2ExitY']]

        data_arr = data.values

        num_tracks = data_arr.shape[0]
        RICH1Entry_minval = np.zeros((num_tracks,nearest_neighbours))
        RICH1Exit_minval = np.zeros((num_tracks,nearest_neighbours))
        RICH2Entry_minval = np.zeros((num_tracks,nearest_neighbours))
        RICH2Exit_minval = np.zeros((num_tracks,nearest_neighbours))

        RICH1Cone = np.zeros(num_tracks)
        RICH2Cone = np.zeros(num_tracks)

        if include_z:
            RICH1Entry = data_arr[:,0:3]
            RICH1Exit = data_arr[:,3:6]
            RICH2Entry = data_arr[:,6:9]
            RICH2Exit = data_arr[:,9:12]

        else:
            RICH1Entry = data_arr[:,0:2]
            RICH1Exit = data_arr[:,2:4]
            RICH2Entry = data_arr[:,4:6]
            RICH2Exit = data_arr[:,6:8]

        RICH1Entry_dist = np.linalg.norm(RICH1Entry - RICH1Entry[:,None], axis=-1)
        RICH1Exit_dist = np.linalg.norm(RICH1Exit - RICH1Exit[:,None], axis=-1)
        RICH2Entry_dist = np.linalg.norm(RICH2Entry - RICH2Entry[:,None], axis=-1)
        RICH2Exit_dist = np.linalg.norm(RICH2Exit - RICH2Exit[:,None], axis=-1)

        RICH1Entry_inrad = (RICH1Entry_dist < max_track_rad)
        RICH1Exit_inrad = (RICH1Exit_dist < max_track_rad)
        RICH2Entry_inrad = (RICH2Entry_dist < max_track_rad)
        RICH2Exit_inrad = (RICH2Exit_dist < max_track_rad)

        RICH1_inrad = np.logical_and(RICH1Entry_inrad, RICH1Exit_inrad)
        RICH2_inrad = np.logical_and(RICH2Entry_inrad, RICH2Exit_inrad)

        for k in range(num_tracks):
            
            RICH1Cone[k] = np.sum(RICH1_inrad[k,:]) - 1
            RICH2Cone[k] = np.sum(RICH2_inrad[k,:]) - 1
            
            for l in range(nearest_neighbours):
            
                if l < (num_tracks - 1):
                
                    RICH1Entry_minval[k,l] = nsmall(RICH1Entry_dist[k,:], l+1)
                    RICH1Exit_minval[k,l] = nsmall(RICH1Exit_dist[k,:], l+1)
                    RICH2Entry_minval[k,l] = nsmall(RICH2Entry_dist[k,:], l+1)
                    RICH2Exit_minval[k,l] = nsmall(RICH2Exit_dist[k,:], l+1)

                else:
                    RICH1Entry_minval[k,l] = 9999
                    RICH1Exit_minval[k,l] = 9999
                    RICH2Entry_minval[k,l] = 9999
                    RICH2Exit_minval[k,l] = 9999

        for k in range(nearest_neighbours):

            kwargs = {corr_vars[k]: RICH1Entry_minval[:,k]}
            data_with_EventNum = data_with_EventNum.assign(**kwargs) 

            kwargs = {corr_vars[k+nearest_neighbours]: RICH1Exit_minval[:,k]}
            data_with_EventNum = data_with_EventNum.assign(**kwargs)

            kwargs = {corr_vars[k+(nearest_neighbours*2)]: RICH2Entry_minval[:,k]}
            data_with_EventNum = data_with_EventNum.assign(**kwargs)

            kwargs = {corr_vars[k+(nearest_neighbours*3)]: RICH2Exit_minval[:,k]}
            data_with_EventNum = data_with_EventNum.assign(**kwargs)


        kwargs = {corr_vars[-2]: RICH1Cone[:]}
        data_with_EventNum = data_with_EventNum.assign(**kwargs) 

        kwargs = {corr_vars[-1]: RICH2Cone[:]}
        data_with_EventNum = data_with_EventNum.assign(**kwargs)         

        if (i==0 and j==0):
            modified_data = data_with_EventNum
        else:
            modified_data = modified_data.append(data_with_EventNum)

print(modified_data.shape)

modified_data.to_hdf('../../data/mod-PID-train-data-'+ particle_source + 'S.hdf', key=particle_source + 'S', mode='w')

##Get first set of DLL data
#DLL_data_1 = np.array(all_data.loc[:, 'RichDLL' + DLLs[0]])
#                
#x_data_dim = (DLL_data_1.shape[0], DLLs_dim + phys_dim + corr_dim) 
#x_data = np.zeros((x_data_dim))
#x_data[:,0] = DLL_data_1
#    
##Get other DLL data
#for i in range(1, DLLs_dim):    
#    x_data[:,i] = np.array(all_data.loc[:, 'RichDLL' + DLLs[i]])
#    
##Get physics data
#for i in range(DLLs_dim, DLLs_dim + phys_dim):
#    phys_vars_index = i - DLLs_dim
#    x_data[:,i] = np.array(all_data.loc[:, physical_vars[phys_vars_index]])
#
##Get corr data
#for i in range(DLLs_dim + phys_dim, DLLs_dim + phys_dim + corr_dim):
#    corr_vars_index = i - (DLLs_dim + phys_dim)
#    x_data[:,i] = np.array(all_data.loc[:, corr_vars[corr_vars_index]])
#    
#x_data, shift, div_num = norm(x_data)
#
#
##Use subset of data
#tot_split = int(frac * x_data.shape[0])
#x_data = x_data[:tot_split]
#    
##Now split into training/test data 70/30?
#split = int(train_frac * len(x_data))
#x_train = x_data[:split]
#x_test = x_data[split:]

#Measure total run time for script
t_final = time.time()
print("Total run time = ", t_final - t_init)
