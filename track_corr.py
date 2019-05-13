#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#Script to estimate busy-ness of tracks (for improved inter-track correlation)
#Calculates three nearest neighbours at RICH 1/2 entry/exit
#Also calculates number of tracks that are within a fixed distance at entrance and exit to RICH 1 or RICH 2

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.spatial.distance import cdist #Alternative distance measure
from tqdm import tqdm

#Time total run
t_init = time.time()

plt.rcParams['agg.path.chunksize'] = 10000 #Needed for plotting lots of data?

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

max_track_rad_1 = 200 #60
max_track_rad_2 = 200 #80

ref_particle = 'pi'
particle_source = 'KAON'

###############################################################################
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


#Get nth smallest value of a
def nsmall(a, n):
    return np.partition(a, n)[n]

all_data = import_all_var(particle_source)

#Unique run numbers
RunNum_list = all_data.RunNumber.unique()
num_RunNum = RunNum_list.shape[0]

#Get all data with same event number and run number
#Note: python loops slow, likely better implementation possible
#Runs much quicker for PION data than KAON data - since fewer run unique runs?
for i in tqdm(range(num_RunNum)):
    
    data_with_RunNum = all_data.loc[all_data['RunNumber'] == RunNum_list[i]]
    
    #Unique event numbers (for a given run number)
    EventNum_list = data_with_RunNum.EventNumber.unique()
    num_EventNum = EventNum_list.shape[0]
    
    #Loop over events
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

        #Calculate distances to all other tracks
        RICH1Entry_dist = np.linalg.norm(RICH1Entry - RICH1Entry[:,None], axis=-1)
        RICH1Exit_dist = np.linalg.norm(RICH1Exit - RICH1Exit[:,None], axis=-1)
        RICH2Entry_dist = np.linalg.norm(RICH2Entry - RICH2Entry[:,None], axis=-1)
        RICH2Exit_dist = np.linalg.norm(RICH2Exit - RICH2Exit[:,None], axis=-1)

        #Check if each track lies within radius
        RICH1Entry_inrad = (RICH1Entry_dist < max_track_rad_1)
        RICH1Exit_inrad = (RICH1Exit_dist < max_track_rad_1)
        RICH2Entry_inrad = (RICH2Entry_dist < max_track_rad_2)
        RICH2Exit_inrad = (RICH2Exit_dist < max_track_rad_2)

        #Combine to check iff within radius for both
        RICH1_inrad = np.logical_and(RICH1Entry_inrad, RICH1Exit_inrad)
        RICH2_inrad = np.logical_and(RICH2Entry_inrad, RICH2Exit_inrad)


        for k in range(num_tracks):
            
            #Count number within 'cone'
            RICH1Cone[k] = np.sum(RICH1_inrad[k,:]) - 1
            RICH2Cone[k] = np.sum(RICH2_inrad[k,:]) - 1
            
            for l in range(nearest_neighbours):
            
                if l < (num_tracks - 1):
                
                    #Set NN distance
                    RICH1Entry_minval[k,l] = nsmall(RICH1Entry_dist[k,:], l+1)
                    RICH1Exit_minval[k,l] = nsmall(RICH1Exit_dist[k,:], l+1)
                    RICH2Entry_minval[k,l] = nsmall(RICH2Entry_dist[k,:], l+1)
                    RICH2Exit_minval[k,l] = nsmall(RICH2Exit_dist[k,:], l+1)

                else:
                    #If no NN assign arb value
                    RICH1Entry_minval[k,l] = 9999 #1000
                    RICH1Exit_minval[k,l] = 9999 #2000
                    RICH2Entry_minval[k,l] = 9999 #7000
                    RICH2Exit_minval[k,l] = 9999 #10000
        
        #Save nearest nearest neighbours
        for k in range(nearest_neighbours):

            kwargs = {corr_vars[k]: RICH1Entry_minval[:,k]}
            data_with_EventNum = data_with_EventNum.assign(**kwargs) 

            kwargs = {corr_vars[k+nearest_neighbours]: RICH1Exit_minval[:,k]}
            data_with_EventNum = data_with_EventNum.assign(**kwargs)

            kwargs = {corr_vars[k+(nearest_neighbours*2)]: RICH2Entry_minval[:,k]}
            data_with_EventNum = data_with_EventNum.assign(**kwargs)

            kwargs = {corr_vars[k+(nearest_neighbours*3)]: RICH2Exit_minval[:,k]}
            data_with_EventNum = data_with_EventNum.assign(**kwargs)

        #Save cone numbers
        kwargs = {corr_vars[-2]: RICH1Cone[:]}
        data_with_EventNum = data_with_EventNum.assign(**kwargs) 

        kwargs = {corr_vars[-1]: RICH2Cone[:]}
        data_with_EventNum = data_with_EventNum.assign(**kwargs)         

        #For first loop, create variable to contain all data, else append to this variable
        if (i==0 and j==0):
            modified_data = data_with_EventNum
        else:
            modified_data = modified_data.append(data_with_EventNum)

#Check new shape
print(modified_data.shape)

#Save all data to hdf 
modified_data.to_hdf('../../data/mod-2-PID-train-data-'+ particle_source + 'S.hdf', key=particle_source + 'S', mode='w')

#Measure total run time for script
t_final = time.time()
print("Total run time = ", t_final - t_init)
