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
import math
from matplotlib.ticker import AutoMinorLocator
import time

#Originally: set13.1 generator - works reasonably well for efficiency plot and DLL correlations to each other (both generated)
#set15 for PION
#Now with set17 for KAON (same as 13.1 but with 6DLLs)
#TrackP not correlated with input TrackP


#Time total run
t_init = time.time()

print("Loading generators...")

set_text = ""

###############################################################################

#epochs = 250 Generate P, Pt 

#generator_k = load_model('../../GAN_data/GAN_7DLL/set17/trained_gan.h5')
#set_text += "set17"
#
#generator_p = load_model('../../GAN_data/GAN_7DLL/set15/trained_gan.h5')
#set_text += "set15"
#
#frac=0.025
#input_physical_vars = ['TrackP', 'TrackPt']
#DLLs = ['e', 'mu', 'k', 'p', 'd', 'bt']
#physical_vars = ['TrackP', 'TrackPt', 'NumLongTracks', 'NumPVs']
#generate_P = True
#alt_model_k = False

###############################################################################

#epochs = 500
#
#generator_k = load_model('../../GAN_data/GAN_7DLL/set18/trained_gan.h5')
#set_text += "set18"
#
#generator_p = load_model('../../GAN_data/GAN_7DLL/set19/trained_gan.h5')
#set_text += "set19"
#
#frac=0.1
#input_physical_vars = ['TrackP', 'TrackPt']
#DLLs = ['e', 'mu', 'k', 'p', 'd', 'bt']
#physical_vars = ['TrackP', 'TrackPt', 'NumLongTracks', 'NumPVs']
#generate_P = True
#alt_model_k = False

###############################################################################

#epochs = 100

#generator_k = load_model('../../GAN_data/GAN_7DLL/set20/trained_gan.h5')
#set_text += "set20"

#frac=0.025
#input_physical_vars = ['TrackP', 'TrackPt']
#DLLs = ['e', 'mu', 'k', 'p', 'd', 'bt']
#physical_vars = ['TrackP', 'TrackPt', 'NumLongTracks', 'NumPVs']
#generate_P = False
#alt_model_k = False

###############################################################################

#epochs = 500
#
#batch_size = 128
#generator_k = load_model('../../GAN_data/GAN_7DLL/set22/trained_gan.h5')
#set_text += "set22"

#batch_size = 128
#generator_p = load_model('../../GAN_data/GAN_7DLL/set23/trained_gan.h5')
#set_text += "set23"

#batch_size = 1024
#generator_p = load_model('../../GAN_data/GAN_7DLL/set24/trained_gan.h5')
#set_text += "set24"

#batch_size = 32
#generator_p = load_model('../../GAN_data/GAN_7DLL/set25/trained_gan.h5')
#set_text += "set25"

#batch_size = 4096
#generator_p = load_model('../../GAN_data/GAN_7DLL/set26/trained_gan.h5')
#set_text += "set26"
#
#frac = 0.1
#input_physical_vars = ['TrackP', 'TrackPt']
#DLLs = ['e', 'mu', 'k', 'p', 'd', 'bt']
#physical_vars = ['TrackP', 'TrackPt', 'NumLongTracks', 'NumPVs']
#generate_P = False
#alt_model_k = False


###############################################################################

#epochs = 500
#
#generator_k = load_model('../../GAN_data/GAN_7DLL/set28/trained_gan.h5')
#set_text += "set28"
#alt_model_k = False
#
#generator_p = load_model('../../GAN_data/GAN_7DLL/set27/trained_gan.h5')
#set_text += "set27"
#
#frac = 0.1
#input_physical_vars = ['TrackP', 'TrackPt', 'NumLongTracks', 'NumPVs']
#DLLs = ['e', 'mu', 'k', 'p', 'd', 'bt']
#physical_vars = input_physical_vars
#generate_P = False


###############################################################################

#New data

#epochs = 500
#
#generator_k = load_model('../../GAN_data/GAN_7DLL/set31/trained_gan.h5')
#set_text += "set31"
#alt_model_k = False
#

#generator_p = load_model('../../GAN_data/GAN_7DLL/set30/trained_gan.h5')
#set_text += "set30"
#
#frac = 0.1
#input_physical_vars = ['TrackP', 'TrackPt', 'NumLongTracks', 'NumPVs']
#DLLs = ['e', 'mu', 'k', 'p', 'd', 'bt']
#physical_vars = input_physical_vars
#generate_P = False

###############################################################################

#epochs = 500

#generator_k = load_model('../../GAN_data/GAN_7DLL/set34/trained_gan.h5')
#set_text += "set34"
#alt_model_k = False
#
##generator_k = load_model('../../GAN_data/GAN_7DLL/set37/trained_gan.h5')
##set_text += "set37"
##alt_model_k = True

#generator_k = load_model('../../GAN_data/GAN_7DLL/set44/trained_wgan.h5')
#set_text += "set44"
#alt_model_k = True
#
#generator_p = load_model('../../GAN_data/GAN_7DLL/set35/trained_gan.h5')
#set_text += "set35"
#
#frac = 0.1
#input_physical_vars = ['TrackP', 'TrackPt', 'NumLongTracks', 'NumPVs', 'TrackVertexX', 'TrackVertexY', 'TrackVertexZ', 
#                 'TrackRich1EntryX', 'TrackRich1EntryY', 'TrackRich1EntryZ', 'TrackRich1ExitX', 'TrackRich1ExitY', 
#                 'TrackRich1ExitZ']
#
#physical_vars = input_physical_vars
#
#generate_P = False

###############################################################################

#Added RICH2

epochs = 500
frac = frac = 1 #0.1

#generator_k = load_model('../../GAN_data/GAN_7DLL/set43/trained_gan.h5')
#set_text += "set43"
#alt_model_k = False
#
#generator_p = load_model('../../GAN_data/GAN_7DLL/set45/trained_gan.h5')
#set_text += "set45"
#alt_model_p = False

#generator_k = load_model('../../GAN_data/GAN_7DLL/set57/trained_gan.h5')
#set_text += "set57"
#alt_model_k = False

generator_p = load_model('../../GAN_data/GAN_7DLL/set60/trained_gan.h5')
set_text += "set60"
alt_model_p = False
#
#generator_k = load_model('../../GAN_data/GAN_7DLL/set58/trained_gan.h5')
#set_text += "set58"
#alt_model_k = False
#
#generator_p = load_model('../../GAN_data/GAN_7DLL/set61/trained_gan.h5')
#set_text += "set61"
#alt_model_p = False

generator_k = load_model('../../GAN_data/GAN_7DLL/set63/half_trained_gan.h5')
set_text += "set63"
alt_model_k = False

#generator_p = load_model('../../GAN_data/GAN_7DLL/set6??/half_trained_gan.h5')
#set_text += "set6??"
#alt_model_p = False

DLLs = ['e', 'mu', 'k', 'p', 'd', 'bt']
#
input_physical_vars = ['TrackP', 'TrackPt', 'NumLongTracks', 'NumPVs', 'TrackVertexX', 'TrackVertexY', 'TrackVertexZ', 
                       'TrackRich1EntryX', 'TrackRich1EntryY', 'TrackRich1EntryZ', 'TrackRich1ExitX', 'TrackRich1ExitY', 
                       'TrackRich1ExitZ', 'TrackRich2EntryX', 'TrackRich2EntryY', 'TrackRich2EntryZ', 'TrackRich2ExitX', 
                       'TrackRich2ExitY', 'TrackRich2ExitZ']

physical_vars = input_physical_vars

#physical_vars = ['TrackP', 'TrackPt', 'NumLongTracks', 'NumPVs', 'TrackVertexX', 'TrackVertexY', 'TrackVertexZ', 
#                 'TrackRich1EntryX', 'TrackRich1EntryY', 'TrackRich1EntryZ', 'TrackRich1ExitX', 'TrackRich1ExitY', 
#                 'TrackRich1ExitZ', 'TrackRich2EntryX', 'TrackRich2EntryY', 'TrackRich2EntryZ', 'TrackRich2ExitX', 
#                 'TrackRich2ExitY', 'TrackRich2ExitZ', 'RICH1EntryDist0', 'RICH1ExitDist0', 'RICH2EntryDist0',
#                 'RICH2ExitDist0', 'RICH1EntryDist1', 'RICH1ExitDist1', 'RICH2EntryDist1', 'RICH2ExitDist1', 
#                 'RICH1EntryDist2', 'RICH1ExitDist2', 'RICH2EntryDist2', 'RICH2ExitDist2', 'RICH1ConeNum',
#                 'RICH2ConeNum']

generate_P = False

###############################################################################

#All old and new data expect removed spacial coods

#epochs = 500
#frac = frac = 1 #0.1
#
#generator_k = load_model('../../GAN_data/GAN_7DLL/set59/trained_gan.h5')
#set_text += "set59"
#alt_model_k = False
#
#generator_p = load_model('../../GAN_data/GAN_7DLL/set62/trained_gan.h5')
#set_text += "set62"
#alt_model_p = False
#
#input_physical_vars = ['TrackP', 'TrackPt', 'NumLongTracks', 'NumPVs', 'RICH1EntryDist0', 'RICH1ExitDist0', 
#                       'RICH2EntryDist0', 'RICH2ExitDist0', 'RICH1EntryDist1', 'RICH1ExitDist1', 'RICH2EntryDist1', 
#                       'RICH2ExitDist1',  'RICH1EntryDist2', 'RICH1ExitDist2', 'RICH2EntryDist2', 'RICH2ExitDist2', 
#                       'RICH1ConeNum', 'RICH2ConeNum']
#
#physical_vars = input_physical_vars
#
#DLLs = ['e', 'mu', 'k', 'p', 'd', 'bt']

#physical_vars = ['TrackP', 'TrackPt', 'NumLongTracks', 'NumPVs', 'TrackVertexX', 'TrackVertexY', 'TrackVertexZ', 
#                 'TrackRich1EntryX', 'TrackRich1EntryY', 'TrackRich1EntryZ', 'TrackRich1ExitX', 'TrackRich1ExitY', 
#                 'TrackRich1ExitZ', 'TrackRich2EntryX', 'TrackRich2EntryY', 'TrackRich2EntryZ', 'TrackRich2ExitX', 
#                 'TrackRich2ExitY', 'TrackRich2ExitZ', 'RICH1EntryDist0', 'RICH1ExitDist0', 'RICH2EntryDist0',
#                 'RICH2ExitDist0', 'RICH1EntryDist1', 'RICH1ExitDist1', 'RICH2EntryDist1', 'RICH2ExitDist1', 
#                 'RICH1EntryDist2', 'RICH1ExitDist2', 'RICH2EntryDist2', 'RICH2ExitDist2', 'RICH1ConeNum',
#                 'RICH2ConeNum']

generate_P = False

###############################################################################

#Added corr info

#epochs = 500
#frac = 1 #0.1
#
#generator_k = load_model('../../GAN_data/GAN_7DLL/set47/trained_gan.h5')
#set_text += "set47"
#alt_model_k = False
#
#generator_p = load_model('../../GAN_data/GAN_7DLL/set46/trained_gan.h5')
#set_text += "set46"
#alt_model_p = False
#
#
#DLLs = ['e', 'mu', 'k', 'p', 'd', 'bt']
#
#input_physical_vars = ['TrackP', 'TrackPt', 'NumLongTracks', 'NumPVs', 'TrackVertexX', 'TrackVertexY', 'TrackVertexZ', 
#                 'TrackRich1EntryX', 'TrackRich1EntryY', 'TrackRich1EntryZ', 'TrackRich1ExitX', 'TrackRich1ExitY', 
#                 'TrackRich1ExitZ', 'TrackRich2EntryX', 'TrackRich2EntryY', 'TrackRich2EntryZ', 'TrackRich2ExitX', 
#                 'TrackRich2ExitY', 'TrackRich2ExitZ', 'RICH1EntryDist0', 'RICH1ExitDist0', 'RICH2EntryDist0',
#                 'RICH2ExitDist0', 'RICH1EntryDist1', 'RICH1ExitDist1', 'RICH2EntryDist1', 'RICH2ExitDist1', 
#                 'RICH1EntryDist2', 'RICH1ExitDist2', 'RICH2EntryDist2', 'RICH2ExitDist2', 'RICH1ConeNum',
#                 'RICH2ConeNum']
#
#physical_vars = input_physical_vars
#
#generate_P = False

###############################################################################

#Added corr info but only used NN data

#epochs = 500
#frac = 1 #0.1
#
#generator_k = load_model('../../GAN_data/GAN_7DLL/set48/trained_gan.h5')
#set_text += "set48"
#alt_model_k = False
#
#generator_p = load_model('../../GAN_data/GAN_7DLL/set49/trained_gan.h5')
#set_text += "set49"
#alt_model_p = False
#
#Reduced inner layers to 4
#generator_k = load_model('../../GAN_data/GAN_7DLL/set50/trained_gan.h5')
#set_text += "set50"
#alt_model_k = False

#Reduced inner layers to 4
#generator_p = load_model('../../GAN_data/GAN_7DLL/set51/trained_gan.h5')
#set_text += "set51"
#alt_model_p = False

#Reduced inner layers to 6
#generator_k = load_model('../../GAN_data/GAN_7DLL/set52/trained_gan.h5')
#set_text += "set52"
#alt_model_k = False
#
#Reduced inner layers to 6
#generator_p = load_model('../../GAN_data/GAN_7DLL/set53/trained_gan.h5')
#set_text += "set53"
#alt_model_p = False
#
##Reduced inner layers to 4, frac = 0.2
#generator_k = load_model('../../GAN_data/GAN_7DLL/set54/trained_gan.h5')
#set_text += "set54"
#alt_model_k = False
#
#DLLs = ['e', 'mu', 'k', 'p', 'd', 'bt']
#
#input_physical_vars = ['TrackP', 'TrackPt', 'NumLongTracks', 'NumPVs', 'TrackVertexX', 'TrackVertexY', 'TrackVertexZ', 
#                 'TrackRich1EntryX', 'TrackRich1EntryY', 'TrackRich1EntryZ', 'TrackRich1ExitX', 'TrackRich1ExitY', 
#                 'TrackRich1ExitZ', 'TrackRich2EntryX', 'TrackRich2EntryY', 'TrackRich2EntryZ', 'TrackRich2ExitX', 
#                 'TrackRich2ExitY', 'TrackRich2ExitZ', 'RICH1EntryDist0', 'RICH1ExitDist0', 'RICH2EntryDist0',
#                 'RICH2ExitDist0', 'RICH1EntryDist1', 'RICH1ExitDist1', 'RICH2EntryDist1', 'RICH2ExitDist1', 
#                 'RICH1EntryDist2', 'RICH1ExitDist2', 'RICH2EntryDist2', 'RICH2ExitDist2']
#
#physical_vars = input_physical_vars

#physical_vars= ['TrackP', 'TrackPt', 'NumLongTracks', 'NumPVs', 'TrackVertexX', 'TrackVertexY', 'TrackVertexZ', 
#                 'TrackRich1EntryX', 'TrackRich1EntryY', 'TrackRich1EntryZ', 'TrackRich1ExitX', 'TrackRich1ExitY', 
#                 'TrackRich1ExitZ', 'TrackRich2EntryX', 'TrackRich2EntryY', 'TrackRich2EntryZ', 'TrackRich2ExitX', 
#                 'TrackRich2ExitY', 'TrackRich2ExitZ', 'RICH1EntryDist0', 'RICH1ExitDist0', 'RICH2EntryDist0',
#                 'RICH2ExitDist0', 'RICH1EntryDist1', 'RICH1ExitDist1', 'RICH2EntryDist1', 'RICH2ExitDist1', 
#                 'RICH1EntryDist2', 'RICH1ExitDist2', 'RICH2EntryDist2', 'RICH2ExitDist2', 'RICH1ConeNum',
#                 'RICH2ConeNum']
#
#generate_P = False

###############################################################################

#8 layers again, all data inc all NN data

#epochs = 500 #(1000 for full gen)
#frac = 1 #0.2
#
#generator_k = load_model('../../GAN_data/GAN_7DLL/set55/trained_gan.h5')
#set_text += "set55.1"
#alt_model_k = False
#
#generator_p = load_model('../../GAN_data/GAN_7DLL/set56/trained_gan.h5')
#set_text += "set56.1"
#alt_model_p = False
#
#DLLs = ['e', 'mu', 'k', 'p', 'd', 'bt']
#
#input_physical_vars = ['TrackP', 'TrackPt', 'NumLongTracks', 'NumPVs', 'TrackVertexX', 'TrackVertexY', 'TrackVertexZ', 
#                 'TrackRich1EntryX', 'TrackRich1EntryY', 'TrackRich1EntryZ', 'TrackRich1ExitX', 'TrackRich1ExitY', 
#                 'TrackRich1ExitZ', 'TrackRich2EntryX', 'TrackRich2EntryY', 'TrackRich2EntryZ', 'TrackRich2ExitX', 
#                 'TrackRich2ExitY', 'TrackRich2ExitZ', 'RICH1EntryDist0', 'RICH1ExitDist0', 'RICH2EntryDist0',
#                 'RICH2ExitDist0', 'RICH1EntryDist1', 'RICH1ExitDist1', 'RICH2EntryDist1', 'RICH2ExitDist1', 
#                 'RICH1EntryDist2', 'RICH1ExitDist2', 'RICH2EntryDist2', 'RICH2ExitDist2', 'RICH1ConeNum',
#                 'RICH2ConeNum']
#
#physical_vars = input_physical_vars
#
#generate_P = False

###############################################################################

subset=False
sub_var = 'RICH1ExitDist0'
sub_min = None
sub_max = 10

if subset:
    subset_text = '_' + sub_var + '_' + str(sub_min) + '-' + str(sub_max)
else:
    subset_text = ''
    
RNN = False
sort_var = 'TrackP'


unused_mask_1 = False
unused_mask_2 = False

###############################################################################

print("Generators loaded")

plt.rcParams['agg.path.chunksize'] = 10000 #Needed for plotting lots of data?

ref_particle = 'pi'
particle_source_1 = 'KAON'
particle_source_2 = 'PION'

train_frac = 0.7
examples=2000000

#Old dimensional stuff
#gen_input_dim = 100 #Dimension of random noise vector.
#phys_dim = len(physical_vars)
#input_phys_dim = len(input_physical_vars)
#DLLs_dim = len(DLLs)
#data_dim = DLLs_dim + phys_dim
#noise_dim = gen_input_dim - input_phys_dim

#New dim stuff
noise_dim = 100 #Dimension of random noise vector.
input_phys_dim = len(input_physical_vars)
phys_dim = len(physical_vars)
DLLs_dim = len(DLLs)
data_dim = DLLs_dim + phys_dim
gen_input_dim = noise_dim + input_phys_dim 

######################################################################################################################################################################################

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
    
    if subset:
        if sub_min is not None:
            if sub_max is not None:
                bool_mask_kaon = (data_kaon[sub_var] >= sub_min & data_kaon[sub_var] <= sub_max)
                bool_mask_pion = (data_pion[sub_var] >= sub_min & data_pion[sub_var] <= sub_max)
            else:
                bool_mask_kaon = (data_kaon[sub_var] >= sub_min)
                bool_mask_pion = (data_pion[sub_var] >= sub_min)
        elif sub_max is not None:
                bool_mask_kaon = (data_kaon[sub_var] <= sub_max)
                bool_mask_pion = (data_pion[sub_var] <= sub_max)
        else:
            print("Subset set to true but no limits given!")
            
        data_kaon = data_kaon[bool_mask_kaon]
        data_pion = data_pion[bool_mask_pion]
    
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

    if subset:
        if sub_min is not None:
            if sub_max is not None:
                bool_mask = (data[sub_var] >= sub_min & data[sub_var] <= sub_max)
            else:
                bool_mask = (data[sub_var] >= sub_min)
        elif sub_max is not None:
                bool_mask = (data[sub_var] <= sub_max)
        else:
            print("Subset set to true but no limits given!")
            
        data = data[bool_mask]
    
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

    data_norm = np.array(pd.read_csv(particle_source + '_norm.csv'))
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


#Get training/test data and normalise
def get_x_data(DLLs, ref_particle, physical_vars, particle_source, unused_mask):
    
    #Get all data from particle source
    all_data = import_all_var(particle_source)
    
    #If using RNN, will need data to be sorted e.g. by TrackP
    if RNN:        
        all_data = all_data.sort_values(by=sort_var,ascending=True)
    
    #Total number of data rows    
    data_length = all_data.shape[0]
    
    #Get first set of DLL data
    DLL_data_1 = np.array(all_data.loc[:, 'RichDLL' + DLLs[0]])
    
    #Create array to store all relevant data, starting with first DLL
    x_data_dim = (data_length, data_dim) 
    x_data = np.zeros((x_data_dim))
    x_data[:,0] = DLL_data_1
    
    #Get other DLL data
    for i in range(1, DLLs_dim):    
        x_data[:,i] = np.array(all_data.loc[:, 'RichDLL' + DLLs[i]])
    
    #Get physics data
    for i in range(DLLs_dim, data_dim):
        phys_vars_index = i - DLLs_dim
        x_data[:,i] = np.array(all_data.loc[:, physical_vars[phys_vars_index]])

    #Have all data at this point, potentially sorted (if RNN)
    #Now default to selecting [examples] number of rows for test data
    #If have masks from training run, using inverse
    
    if unused_mask:

        unused_data_mask = np.array(pd.read_csv('unused_data_mask.csv'))

        #Apply this mask to x_data, leaving (10000000 * (1- frac * train_frac)) points remaining e.g. 9300000 for frac = 0.1, train_frac = 0.7
        x_data_testable = x_data[unused_data_mask]
                
        #Now reduce this down to [examples] through random selection
        zero_arr =np.zeros(x_data_testable.shape[0] - examples, dtype=bool)
        ones_arr = np.ones(examples, dtype=bool)
        examples_mask = np.concatenate((zero_arr,ones_arr))
        np.random.shuffle(examples_mask)
    
        #Apply boolean mask
        x_test = x_data_testable[examples_mask]
        
    else:
        
        zero_arr =np.zeros(data_length - examples, dtype=bool)
        ones_arr = np.ones(examples, dtype=bool)
        combined_01_arr = np.concatenate((zero_arr,ones_arr))
        np.random.shuffle(combined_01_arr)
    
        #Apply boolean mask
        x_test = x_data[combined_01_arr]
    
    #Normalise data. Shuffle not needed as all x_test used and order doesn't matter
    x_test, shift, div_num = norm(x_test, particle_source)
    
    return x_test, shift, div_num 
    
######################################################################################################################################################################################

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
        fig1.savefig("./plots/" + set_text + "_" + var1_text + "_" + var2_text + subset_text + "_colour.eps", format='eps', dpi=1000)
    
######################################################################################################################################################################################
#Plot efficiency against momentum
def eff_mom_plot(p_points, source_1_eff_0, source_1_eff_5, source_2_eff_0, source_2_eff_5, DLL_part_1, DLL_part_2, particle_source_1, particle_source_2, p_max, p_label):
 
    title = DLL_part_1 + "_" + DLL_part_2 + "_" + particle_source_1 + "_" + particle_source_2 + "_" + str(p_max) + p_label
    
    if(particle_source_1 == 'PION'):
        particle_source_1 = r'$\pi\ $'
    elif(particle_source_1 == 'KAON'):
        particle_source_1 = 'K'
    
    if(particle_source_2 == 'PION'):
        particle_source_2 = r'$\pi\ $'
    elif(particle_source_2 == 'KAON'):
        particle_source_2 = 'K'        
            
    if(DLL_part_1 == 'pi'):
        DLL_part_1 = r'$\pi\ $'
    elif(DLL_part_1 == 'k'):
        DLL_part_1 = 'K'
      
    if(DLL_part_2 == 'pi'):
        DLL_part_2 = r'$\pi\ $'
    elif(DLL_part_2 == 'k'):
        DLL_part_2 = 'K'
    
    process_1_text = particle_source_1 + " " + r'$\rightarrow$' + " " + DLL_part_1
    process_2_text = particle_source_2 + " " + r'$\rightarrow$' + " " + DLL_part_1
    DLL_text = r'$\Delta LL ($' + DLL_part_1 + '-' + DLL_part_2
    
    fig1, ax1 = plt.subplots()
    ax1.cla()
    ax1.set_ylim(0,1.4)
    ax1.set_xlabel('Momentum (GeV/c)')
    ax1.set_ylabel('Efficiency')
    ax1.yaxis.set_minor_locator(AutoMinorLocator(4))
    
    s1_0 = ax1.scatter(p_points, source_1_eff_0, s = 5, marker = 'o', facecolors = 'none', edgecolors = 'r')
    s1_5 = ax1.scatter(p_points, source_1_eff_5, s = 5, marker = 'o', color = 'r')
    s2_0 = ax1.scatter(p_points, source_2_eff_0, s = 5, marker = 's', facecolors = 'none', edgecolors = 'k')
    s2_5 = ax1.scatter(p_points, source_2_eff_5, s = 5, marker = 's', color = 'k')    
    ax1.legend((s1_0, s1_5, s2_0, s2_5), (process_1_text + ', ' + DLL_text + ' > 0)', process_1_text + ', ' + DLL_text + ' > 5)', process_2_text + ', ' + DLL_text + ' > 0', process_2_text + ', ' + DLL_text + ' > 5)'), loc='upper right', ncol=2, fontsize=8)
    fig1.savefig("./plots/" + set_text + "_" + title + subset_text + ".eps", format='eps', dpi=1000)
    

#Calculate efficiency of particle identification for general DLL
def calc_eff(bins_no, DLL, DLL_lim, data_no, bounds, bin_var):
    
    particle_no = np.zeros(bins_no, dtype=float)
    tot_no = np.zeros(bins_no, dtype=float)

    for i in range(0, bins_no):

        DLL_arr = np.zeros(data_no, dtype=float)
        
        #Create bins (boolean depending on whether bin variavle lies in range or not)
        bins = (bin_var >= bounds[i]) & (bin_var < bounds[i+1])
    
        DLL_arr = np.multiply(bins, DLL) #Set DLL to 0 if not in variable range
        DLL_arr = np.subtract(DLL_arr, DLL_lim) #DLL Subtract limit e.g. 0 or 5 
        DLL_arr = np.clip(DLL_arr, 0, None) #Set all values less than 0 to 0
    
        particle_no[i] = np.count_nonzero(DLL_arr) #Count non-zero values
        tot_no[i] = np.sum(bins) #Sum events in momentum bin
        
#        print("Range: ", bounds[i], "-", bounds[i+1], "Total:", tot_no[i], " Particles:", particle_no[i])

        #Efficiency: divide number of events within DLL limit by total number
        eff = np.divide(particle_no, tot_no, out=np.zeros_like(particle_no), where=tot_no!=0)

    return eff


# Generate bounds/mid points for momentum bins between 0 and pmax
# Number of bins = p_bins_no
# uni_bins = equal bin spacing, exp_bins = exponential (base 2) bin spacing
# Else linear increas in bin spacing
def calc_p(p_bins_no, p_max, uni_bins,exp_bins, exponent):
    
    p_bounds = np.zeros(p_bins_no + 1)
    
    #Uniform bin spacing
    if(uni_bins):
        
        p_points = np.linspace((0.5 * p_max / p_bins_no), p_max - (0.5 * p_max / p_bins_no), p_bins_no)

        for j in range(0, p_bins_no):
            p_bounds[j+1] = (j+1) * p_max * 10e2 / p_bins_no
    
    #Expontential bin spacing
    elif(exp_bins):
        p_bounds = np.logspace(0, math.log(p_max,exponent), num = p_bins_no, base=exponent)        
        p_bounds = np.insert(p_bounds,0,0)
        p_bounds = np.multiply(p_bounds,10e2)

        p_points = np.zeros(p_bins_no)

        for k in range(1, p_bins_no + 1):
            p_points[k-1] = 0.5 * (p_bounds[k-1] + p_bounds[k]) 
        
    else:
        #Linearly increasing bin spacing
        p_points = np.zeros(p_bins_no)
        dp = p_max / p_bins_no #Initial bin number
        x = 2 / (p_bins_no + 1) #Add factor to increase bin sizes as P increases
        
        for l in range(1, p_bins_no + 1):
            p_bounds[l] = p_bounds[l-1] + x * l * dp 
            p_points[l-1] = 0.5 * (p_bounds[l-1] + p_bounds[l]) 

        p_bounds = np.multiply(p_bounds,10e2)

    return p_points, p_bounds


#Calculate and plot efficiency for K-> K
def eff_mom_calc(p_bins_no, p_max, uni_bins, exp_bins, exponent, DLL_1, mom_1, DLL_2, mom_2, DLL_part_1, DLL_part_2, particle_source_1, particle_source_2, p_label):

    #Calculate the momentum bin centres and bounds
    p_points, p_bounds = calc_p(p_bins_no, p_max, uni_bins,exp_bins, exponent)
               
    #Number of data points from source 1 (KAONS)
    data_no_1 = len(DLL_1)

    #Number of data points from source 2 (PIONS)
    data_no_2 = len(DLL_2)
        
    #Calculate the particle identification efficiencies for different DLL limits
    source_1_eff_0 = calc_eff(p_bins_no, DLL_1, 0, data_no_1, p_bounds, mom_1)
    source_1_eff_5 = calc_eff(p_bins_no, DLL_1, 5, data_no_1, p_bounds, mom_1)

    source_2_eff_0 = calc_eff(p_bins_no, DLL_2, 0, data_no_2, p_bounds, mom_2)
    source_2_eff_5 = calc_eff(p_bins_no, DLL_2, 5, data_no_2, p_bounds, mom_2)
    
    eff_mom_plot(p_points, source_1_eff_0, source_1_eff_5, source_2_eff_0, source_2_eff_5, DLL_part_1, DLL_part_2, particle_source_1, particle_source_2, p_max, p_label)

######################################################################################################################################################################################


def id_misid_eff(DLL1, DLL2, phys_var_1, phys_var_2, bins_no, DLL_lim, DLL_no, phys_var_range, var_name, x_range, DLL_label, var_range):

    #Number of data points
    data_no_1 = len(DLL1)
    data_no_2 = len(DLL2)
    
    source_1_eff_av = np.zeros([bins_no, DLL_no-1])
    source_2_eff_av = np.zeros([bins_no, DLL_no-1])

    full_bounds = np.linspace(phys_var_range[0], phys_var_range[1], num = bins_no + 1)

    if var_name == 'NumLongTracks':
        plot_title = "No. Tracks in Event"
    elif var_name == 'NumPVs':
        plot_title = "No. Reco PVs in Event"
    elif var_name == 'RICH1EntryDist0':
        plot_title = "Distance to nearest track at RICH1 Entry"
    else:
        plot_title = var_name
            
    labels = []    
    for i in range(bins_no):
        if var_range:
            labels.append('[' + str(int(full_bounds[i])) + ',' + str(int(full_bounds[i+1])) + ']')
        else:
            labels.append(str(int(full_bounds[i])))
        
    DLL_lims = np.linspace(0, DLL_lim, DLL_no)

    for i in range(bins_no):
        bounds=full_bounds[i:i+2]
        for j in range(0, DLL_no - 1):
            source_1_eff = calc_eff(1, DLL1, DLL_lims[j], data_no_1, bounds, phys_var_1)
            source_1_eff_av[i, j] = np.average(source_1_eff)
            source_2_eff = calc_eff(1, DLL2, DLL_lims[j], data_no_2, bounds, phys_var_2)
            source_2_eff_av[i, j] = np.average(source_2_eff)
        
    fig1, ax1 = plt.subplots()
    ax1.cla()
    ax1.set_xlim(x_range[0], x_range[1])
    ax1.set_xlabel('Kaon ID Efficiency')
    ax1.set_ylabel('Pion Mis-ID Efficiency')
#    ax1.xaxis.set_minor_locator(AutoMinorLocator(4))
    
    if bins_no == 4:
        ax1.semilogy(source_1_eff_av[0,:], source_2_eff_av[0,:], 'yo-', markersize=4, label=labels[0])
        ax1.semilogy(source_1_eff_av[1,:], source_2_eff_av[1,:], 'rs-', markersize=4, label=labels[1])
        ax1.semilogy(source_1_eff_av[2,:], source_2_eff_av[2,:], 'b^-', markersize=4, label=labels[2])
        ax1.semilogy(source_1_eff_av[3,:], source_2_eff_av[3,:], 'gv-', markersize=4, label=labels[3])
    else:
        for i in range(bins_no):
            ax1.semilogy(source_1_eff_av[i,:], source_2_eff_av[i,:], markersize=4, label=labels[i])
            
    ax1.legend(title=plot_title, loc='upper left', fontsize=8)
    
    fig1.savefig("./plots/" + set_text + "_" + "kID_pMID_eff_" + var_name + "_" + DLL_label + subset_text + ".eps", format='eps', dpi=1000)


######################################################################################################################################################################################

def plot_DLL_hist(DLL_data, DLL_particle, ref_particle, particle_source, bin_no=200, x_range=None, y_range=None):
        
    title = "DLL" + DLL_particle + "-" + ref_particle + "_" + particle_source + subset_text + "_hist.eps"
    DLL_text = r'$\Delta LL ($' + DLL_particle + '-' + ref_particle + ')'
            
    fig1, ax1 = plt.subplots()
    ax1.cla()
    
    if y_range is not None:
        ax1.set_ylim(bottom=0, top=y_range)

    if x_range is not None:
        ax1.set_xlim(x_range)
    
    ax1.set_xlabel(DLL_text)
    ax1.set_ylabel("Density of events")
    
    ax1.hist(DLL_data, bins=bin_no, range=x_range, density=True)
    
    fig1.savefig("./plots/" + set_text + "_" + title, format='eps', dpi=2500)

def plot_two_DLL_hist(DLL_data_1, DLL_data_2, DLL_particle, ref_particle, particle_source, bin_no=200, x_range=None, y_range=None):
        
    title = "DLL" + DLL_particle + "-" + ref_particle + "_" + particle_source + subset_text + "_two_hist.pdf"
    DLL_text = r'$\Delta LL ($' + DLL_particle + '-' + ref_particle + ')'
            
    fig1, ax1 = plt.subplots()
    ax1.cla()
    
    if y_range is not None:
        ax1.set_ylim(bottom=0, top=y_range)

    if x_range is not None:
        ax1.set_xlim(x_range)
    
    ax1.set_xlabel(DLL_text)
    ax1.set_ylabel("Density of events")
    
    ax1.hist(DLL_data_1, bins=bin_no, range=x_range, density=True, histtype='step', color = 'b', linewidth=0.5)
    ax1.hist(DLL_data_2, bins=bin_no, range=x_range, density=True, histtype='step', color='r', linewidth=0.5)

#    ax1.hist(DLL_data_1, bins=bin_no, range=x_range, density=True, histtype='stepfilled', alpha=0.8, color = 'r')
#    ax1.hist(DLL_data_2, bins=bin_no, range=x_range, density=True, histtype='stepfilled', alpha=0.6, color='b')

    fig1.savefig("./plots/" + set_text + "_" + title, format='pdf', dpi=2500)



def plot_gen_hist(var_data, var_name, particle_source, bin_no=200, x_range=None, y_range=None):
        
    if var_name == 'TrackP':
        x_label_text = 'Momentum (GeV/c)'

    if var_name == 'TrackPt':
        x_label_text = 'Transverse Momentum (GeV/c)'

    title = var_name + "_" + particle_source + subset_text + "_hist.eps"
    
    fig1, ax1 = plt.subplots()
    ax1.cla()
    
    if y_range is not None:
        ax1.set_ylim(bottom=0, top=y_range)
    
    if x_range is not None:
        ax1.set_xlim(x_range)
    
    ax1.set_xlabel(x_label_text)
    ax1.set_ylabel("Number of events")
        
    ax1.hist(var_data, bins=bin_no, range=x_range)    
    
    fig1.savefig("./plots/" + set_text + "_" + title, format='eps', dpi=2500)


#Using pdfs for some reason?
def plot_hist_ratio(DLL_data_1, DLL_data_2, DLL_particle, ref_particle, particle_source, bin_no=200, x_range=None, y_range=None):

    title = "DLL" + DLL_particle + "-" + ref_particle + "_" + particle_source + subset_text + "_ratio.pdf"
    DLL_text = r'$\Delta LL ($' + DLL_particle + '-' + ref_particle + ')'
            
    fig1, ax1 = plt.subplots()
    ax1.cla()
    
    if y_range is not None:
        ax1.set_ylim(bottom=0, top=y_range)

    if x_range is not None:
        ax1.set_xlim(x_range)
    
    hist1, _ = np.histogram(DLL_data_1, bins=bin_no, range=x_range, density=True)
    hist2, _ = np.histogram(DLL_data_2, bins=bin_no, range=x_range, density=True)
    ratio = np.divide(hist2, hist1)
    
    ax1.set_xlabel(DLL_text)
    ax1.set_ylabel("Ratio of event densities")
    
    DLL = np.linspace(x_range[0],x_range[1],num=bin_no)
    ax1.plot(DLL, ratio)

    fig1.savefig("./plots/" + set_text + "_" + title, format='pdf', dpi=2500)


def histogram_intersection(DLL_data_1, DLL_data_2, bin_no=200, x_range=None):
   
   hist1, bins = np.histogram(DLL_data_1, bins=bin_no, range=x_range, density=True)
   hist2, _ = np.histogram(DLL_data_2, bins=bin_no, range=x_range, density=True)
   bins = np.diff(bins)
   sm = 0
   for i in range(len(bins)):
       sm += min(bins[i]*hist1[i], bins[i]*hist2[i])
   return sm

######################################################################################################################################################################################


def plot_examples(generated_vars, var_name, bin_no=400, x_range = None, y_range = None):
    
    fig1, ax1 = plt.subplots()
    ax1.cla()
    
    title = 'GAN7_generated_' + var_name + subset_text + '_trained_.eps'
    
    if y_range is not None:
        ax1.set_ylim(bottom = 0, top = y_range)
    
    if x_range is not None:
        ax1.set_xlim(x_range)
    
    ax1.set_xlabel(var_name)
    ax1.set_ylabel("Number of events")
    ax1.hist(generated_vars, bins=bin_no, range=x_range)
    
    fig1.savefig("./plots/" + set_text + "_" + title, format='eps', dpi=2500)

######################################################################################################################################################################################

print("Importing data...")
#Get the training and testing data
x_test_1, shift_1, div_num_1 = get_x_data(DLLs, ref_particle, physical_vars, particle_source_1, examples, unused_mask_1)
x_test_2, shift_2, div_num_2 = get_x_data(DLLs, ref_particle, physical_vars, particle_source_2, examples, unused_mask_2)
print("Data imported")

######################################################################################################################################################################################

print("Generating data...")
#KAON data
#data_batch_1 = x_test_1[np.random.randint(0, x_test_1.shape[0], size=examples)]
            
noise_1 = np.random.normal(0, 1, size=[examples, noise_dim])
phys_data_1 = x_test_1[:, DLLs_dim:DLLs_dim + input_phys_dim]

if alt_model_k:
    generated_vars_k = generator_k.predict([noise_1, phys_data_1])
else:
    gen_input_1 = np.zeros((examples, gen_input_dim))            
    gen_input_1[:, :-input_phys_dim] = noise_1
    gen_input_1[:, -input_phys_dim:] = phys_data_1
    generated_vars_k = generator_k.predict(gen_input_1)


######################################################################################################################################################################################

#PION data
#data_batch_2 = x_test_2[np.random.randint(0, x_test_2.shape[0], size=examples)]

noise_2 = np.random.normal(0, 1, size=[examples, noise_dim])            
phys_data_2 = x_test_2[:, DLLs_dim:DLLs_dim + input_phys_dim]

gen_input_2 = np.zeros((examples, gen_input_dim))            
gen_input_2[:, :-input_phys_dim] = noise_2
gen_input_2[:, -input_phys_dim:] = phys_data_2
    
generated_vars_p = generator_p.predict(gen_input_2)

print("Data generated")

######################################################################################################################################################################################

print("Plotting data...")

#Shift KAON data (real and generated) back to proper distribution
for i in range(x_test_1.shape[1]):
    x_test_1[:,i] = np.multiply(x_test_1[:,i], div_num_1[i])
    x_test_1[:,i] = np.add(x_test_1[:,i], shift_1[i])

for i in range(generated_vars_k.shape[1]):        
    
    generated_vars_k[:,i] = np.multiply(generated_vars_k[:,i], div_num_1[i])
    generated_vars_k[:,i] = np.add(generated_vars_k[:,i], shift_1[i])
        
    if i<DLLs_dim:
        plot_examples(generated_vars_k[:,i], 'KAON_DLL'+ DLLs[i])
    else:
        plot_examples(generated_vars_k[:,i], physical_vars[i-DLLs_dim])

#Shift PION data (real and generated) back to proper distribution
for i in range(x_test_2.shape[1]):
    x_test_2[:,i] = np.multiply(x_test_2[:,i], div_num_2[i])
    x_test_2[:,i] = np.add(x_test_2[:,i], shift_2[i])

for i in range(generated_vars_p.shape[1]):        
    
    generated_vars_p[:,i] = np.multiply(generated_vars_p[:,i], div_num_2[i])
    generated_vars_p[:,i] = np.add(generated_vars_p[:,i], shift_2[i])
        
    if i<DLLs_dim:
        plot_examples(generated_vars_p[:,i], 'PION_DLL'+ DLLs[i])
    else:
        plot_examples(generated_vars_p[:,i], physical_vars[i-DLLs_dim])

######################################################################################################################################################################################

max_var_index = 10000
#Scatter plot between DLLd and DLLk for KAON and PION data
col_scatt(generated_vars_k[:,2], generated_vars_k[:,3], "DLLk_k", "DLLd_k", max_var_index, x_range=[-100,100], y_range=[-80,80], zero_lines=1, save_index=10000, size=0.5)
col_scatt(generated_vars_p[:,2], generated_vars_p[:,3], "DLLk_p", "DLLd_p", max_var_index, x_range=[-100,100], y_range=[-80,80], zero_lines=1, save_index=10000, size=0.5)


#Scatter plot between DLLk for KAON data (generated and original)
col_scatt(generated_vars_k[:,2], x_test_1[:,2], "DLLk_k", "DLLk_k", max_var_index, x_range=[-100,100], y_range=[-80,80], zero_lines=1, save_index=10000, size=0.5)
col_scatt(generated_vars_p[:,2], x_test_2[:,2], "DLLk_p", "DLLk_p", max_var_index, x_range=[-100,100], y_range=[-80,80], zero_lines=1, save_index=10000, size=0.5)

#Scatter plot between TrackP for KAON data if P generated (generated and original)
#if generate_P:
#    col_scatt(generated_vars_k[:,6], x_test_1[:,6], "TrackP", "TrackP", max_var_index, x_range=[0,100000], y_range=[0,100000], zero_lines=1, save_index=10000, size=0.5)


######################################################################################################################################################################################

p_bins_no = 100 #Number of momentum bins
p_max = 100.0 #Maximum track momentum
uni_bins = 0 #Uniform bin sizes
exp_bins = 0 #Exponentially increasing bin sizes (if neither uni or exp, linear increas)
exponent = 2 #Exponent for logspace. Doesn't change anything currently as overspecified?

#Calculate efficiency using generated DLLs and generated momementa
if generate_P:
    eff_mom_calc(p_bins_no, p_max, uni_bins, exp_bins, exponent, generated_vars_k[:,2], generated_vars_k[:,6], generated_vars_p[:,2], generated_vars_p[:,6], 'k', ref_particle, particle_source_1, particle_source_2, 'gen_P')

#Calculate efficiency using generated DLLs and real momementa... Doesn't work when with models which generate P/Pt.
eff_mom_calc(p_bins_no, p_max, uni_bins, exp_bins, exponent, generated_vars_k[:,2], x_test_1[:,6], generated_vars_p[:,2], x_test_2[:,6], 'k', ref_particle, particle_source_1, particle_source_2, 'real_P')

######################################################################################################################################################################################

misid_bin_no = 4
DLL_lim = 15
DLL_no = 21

######################################################################################################################################################################################

#'NumLongTracks'
phys_var_range = [0,400]
x_range = [0.2, 1]

#Generated DLLk, real no of tracks:
id_misid_eff(generated_vars_k[:,2], generated_vars_p[:,2], x_test_1[:,8], x_test_2[:,8], misid_bin_no, DLL_lim, DLL_no, phys_var_range, physical_vars[2], x_range, 'gen_DLL', True)

#Real DLLk, real no of tracks:
id_misid_eff(x_test_1[:,2], x_test_2[:,2], x_test_1[:,8], x_test_2[:,8], misid_bin_no, DLL_lim, DLL_no, phys_var_range, physical_vars[2], x_range, 'real_DLL', True)

######################################################################################################################################################################################

#'NumPVs'

phys_var_range = [1,5]
x_range = [0.5, 1]

#Generated DLLk, real no of tracks:
id_misid_eff(generated_vars_k[:,2], generated_vars_p[:,2], x_test_1[:,9], x_test_2[:,9], misid_bin_no, DLL_lim, DLL_no, phys_var_range, physical_vars[3], x_range, 'gen_DLL', False)

#Real DLLk, real no of tracks:
id_misid_eff(x_test_1[:,2], x_test_2[:,2], x_test_1[:,9], x_test_2[:,9], misid_bin_no, DLL_lim, DLL_no, phys_var_range, physical_vars[3], x_range, 'real_DLL', False)

######################################################################################################################################################################################

#'RICH1EntryDist0'
#Increase in KID as dist increases
#misid_bin_no = 4
#phys_var_range = [0,40]
#x_range = [0.2, 1]
#id_misid_eff(generated_vars_k[:,2], generated_vars_p[:,2], x_test_1[:,25], x_test_2[:,25], misid_bin_no, DLL_lim, DLL_no, phys_var_range, physical_vars[19], x_range, 'gen_DLL', True)
#id_misid_eff(x_test_1[:,2], x_test_2[:,2], x_test_1[:,25], x_test_2[:,25], misid_bin_no, DLL_lim, DLL_no, phys_var_range, physical_vars[19], x_range, 'real_DLL', True)
#
##Increases with distance
##'RICH1ExitDist0'
#misid_bin_no = 4
#phys_var_range = [0,100]
#x_range = [0.3, 1]
#id_misid_eff(generated_vars_k[:,2], generated_vars_p[:,2], x_test_1[:,26], x_test_2[:,26], misid_bin_no, DLL_lim, DLL_no, phys_var_range, physical_vars[20], x_range, 'gen_DLL', True)
#id_misid_eff(x_test_1[:,2], x_test_2[:,2], x_test_1[:,26], x_test_2[:,26], misid_bin_no, DLL_lim, DLL_no, phys_var_range, physical_vars[20], x_range, 'real_DLL', True)
#
##'RICH2EntryDist0'
##Increase as distance increases
#misid_bin_no = 4
#phys_var_range = [0,400]
#x_range = [0.1, 1]
#id_misid_eff(generated_vars_k[:,2], generated_vars_p[:,2], x_test_1[:,27], x_test_2[:,27], misid_bin_no, DLL_lim, DLL_no, phys_var_range, physical_vars[21], x_range, 'gen_DLL', True)
#id_misid_eff(x_test_1[:,2], x_test_2[:,2], x_test_1[:,27], x_test_2[:,27], misid_bin_no, DLL_lim, DLL_no, phys_var_range, physical_vars[21], x_range, 'real_DLL', True)
#
#
##'RICH2ExitDist0'
##Increase as distance increases
#misid_bin_no = 4
#phys_var_range = [0,800]
#x_range = [0.1, 1]
#id_misid_eff(generated_vars_k[:,2], generated_vars_p[:,2], x_test_1[:,28], x_test_2[:,28],misid_bin_no, DLL_lim, DLL_no, phys_var_range, physical_vars[22], x_range, 'gen_DLL', True)
#id_misid_eff(x_test_1[:,2], x_test_2[:,2], x_test_1[:,28], x_test_2[:,28],misid_bin_no, DLL_lim, DLL_no, phys_var_range, physical_vars[22], x_range, 'real_DLL', True)
#
##'RICH1EntryDist1'
##Increases with dist
#misid_bin_no = 4
#phys_var_range = [0,40]
#x_range = [0.2, 1]
#id_misid_eff(generated_vars_k[:,2], generated_vars_p[:,2], x_test_1[:,29], x_test_2[:,29], misid_bin_no, DLL_lim, DLL_no, phys_var_range, physical_vars[23], x_range, 'gen_DLL', True)
#id_misid_eff(x_test_1[:,2], x_test_2[:,2], x_test_1[:,29], x_test_2[:,29], misid_bin_no, DLL_lim, DLL_no, phys_var_range, physical_vars[23], x_range, 'real_DLL', True)
#
##'RICH1ExitDist1'
##Increase with distance
#misid_bin_no = 4
#phys_var_range = [0,100]
#x_range = [0.3, 1]
#id_misid_eff(generated_vars_k[:,2], generated_vars_p[:,2], x_test_1[:,30], x_test_2[:,30], misid_bin_no, DLL_lim, DLL_no, phys_var_range, physical_vars[24], x_range, 'gen_DLL', True)
#id_misid_eff(x_test_1[:,2], x_test_2[:,2], x_test_1[:,30], x_test_2[:,30], misid_bin_no, DLL_lim, DLL_no, phys_var_range, physical_vars[24], x_range, 'real_DLL', True)
#
##'RICH2EntryDist1'
##Increase as distance increases
#misid_bin_no = 4
#phys_var_range = [0,400]
#x_range = [0.1, 1]
#id_misid_eff(generated_vars_k[:,2], generated_vars_p[:,2], x_test_1[:,31], x_test_2[:,31], misid_bin_no, DLL_lim, DLL_no, phys_var_range, physical_vars[25], x_range, 'gen_DLL', True)
#id_misid_eff(x_test_1[:,2], x_test_2[:,2], x_test_1[:,31], x_test_2[:,31], misid_bin_no, DLL_lim, DLL_no, phys_var_range, physical_vars[25], x_range, 'real_DLL', True)
#
#
##'RICH2ExitDist1'
##Increase as distance increases
#misid_bin_no = 4
#phys_var_range = [0,800]
#x_range = [0.1, 1]
#id_misid_eff(generated_vars_k[:,2], generated_vars_p[:,2], x_test_1[:,32], x_test_2[:,32], misid_bin_no, DLL_lim, DLL_no, phys_var_range, physical_vars[26], x_range, 'gen_DLL', True)
#id_misid_eff(x_test_1[:,2], x_test_2[:,2], x_test_1[:,32], x_test_2[:,32], misid_bin_no, DLL_lim, DLL_no, phys_var_range, physical_vars[26], x_range, 'real_DLL', True)
#
#
#
##'RICH1EntryDist2'
##Incrases, last two together
#misid_bin_no = 4
#phys_var_range = [0,100]
#x_range = [0.2, 1]
#id_misid_eff(generated_vars_k[:,2], generated_vars_p[:,2], x_test_1[:,33], x_test_2[:,33], misid_bin_no, DLL_lim, DLL_no, phys_var_range, physical_vars[27], x_range, 'gen_DLL', True)
#id_misid_eff(x_test_1[:,2], x_test_2[:,2], x_test_1[:,33], x_test_2[:,33], misid_bin_no, DLL_lim, DLL_no, phys_var_range, physical_vars[27], x_range, 'real_DLL', True)
#
##'RICH1ExitDist2'
##Increases with distance
#misid_bin_no = 4
#phys_var_range = [0,100]
#x_range = [0.3, 1]
#id_misid_eff(generated_vars_k[:,2], generated_vars_p[:,2], x_test_1[:,34], x_test_2[:,34], misid_bin_no, DLL_lim, DLL_no, phys_var_range, physical_vars[28], x_range, 'gen_DLL', True)
#id_misid_eff(x_test_1[:,2], x_test_2[:,2], x_test_1[:,34], x_test_2[:,34], misid_bin_no, DLL_lim, DLL_no, phys_var_range, physical_vars[28], x_range, 'real_DLL', True)
#
#
##'RICH2EntryDist2'
##Increase as distance increases
#misid_bin_no = 4
#phys_var_range = [0,800]
#x_range = [0.1, 1]
#id_misid_eff(generated_vars_k[:,2], generated_vars_p[:,2], x_test_1[:,35], x_test_2[:,35], misid_bin_no, DLL_lim, DLL_no, phys_var_range, physical_vars[29], x_range, 'gen_DLL', True)
#id_misid_eff(x_test_1[:,2], x_test_2[:,2], x_test_1[:,35], x_test_2[:,35], misid_bin_no, DLL_lim, DLL_no, phys_var_range, physical_vars[29],x_range, 'real_DLL', True)
#
#
##'RICH2ExitDist2'
##Increase as distance increases
#misid_bin_no = 4
#phys_var_range = [0,800]
#x_range = [0.1, 1]
#id_misid_eff(generated_vars_k[:,2], generated_vars_p[:,2], x_test_1[:,36], x_test_2[:,36], misid_bin_no, DLL_lim, DLL_no, phys_var_range, physical_vars[30], x_range, 'gen_DLL', True)
#id_misid_eff(x_test_1[:,2], x_test_2[:,2], x_test_1[:,36], x_test_2[:,36], misid_bin_no, DLL_lim, DLL_no, phys_var_range, physical_vars[30], x_range, 'real_DLL', True)

######################################################################################################################################################################################

##Increase in KID as num decreases
##'RICH1ConeNum'
#misid_bin_no = 4
#phys_var_range = [0,24]
#x_range = [0.1, 1]
#id_misid_eff(generated_vars_k[:,2], generated_vars_p[:,2], x_test_1[:,37], x_test_2[:,37], misid_bin_no, DLL_lim, DLL_no, phys_var_range, physical_vars[31], x_range, 'gen_DLL', True)
#id_misid_eff(x_test_1[:,2], x_test_2[:,2], x_test_1[:,37], x_test_2[:,37], misid_bin_no, DLL_lim, DLL_no, phys_var_range, physical_vars[31], x_range, 'real_DLL', True)
#
#
##Increase in KID as num decreases
##'RICH2ConeNum'
#misid_bin_no = 4
#phys_var_range = [0,4]
#x_range = [0.1, 1]
#id_misid_eff(generated_vars_k[:,2], generated_vars_p[:,2], x_test_1[:,38], x_test_2[:,38], misid_bin_no, DLL_lim, DLL_no, phys_var_range, physical_vars[32], x_range, 'gen_DLL', True)
#id_misid_eff(x_test_1[:,2], x_test_2[:,2], x_test_1[:,38], x_test_2[:,38], misid_bin_no, DLL_lim, DLL_no, phys_var_range, physical_vars[32], x_range, 'real_DLL', True)


######################################################################################################################################################################################

#Plot histograms of DLLs from generated KAON data
plot_DLL_hist(generated_vars_k[:,0], DLLs[0], ref_particle, particle_source_1, 500, [-40, 20], 0.3)
plot_DLL_hist(generated_vars_k[:,1], DLLs[1], ref_particle, particle_source_1, 500, [-20, 15], 0.6)
plot_DLL_hist(generated_vars_k[:,2], DLLs[2], ref_particle, particle_source_1, 750, [-40, 80], 0.07)
plot_DLL_hist(generated_vars_k[:,3], DLLs[3], ref_particle, particle_source_1, 600, [-40, 60], 0.07)
plot_DLL_hist(generated_vars_k[:,4], DLLs[4], ref_particle, particle_source_1, 600, [-40, 60], 0.07)
plot_DLL_hist(generated_vars_k[:,5], DLLs[5], ref_particle, particle_source_1, 600, [-40, 60], 0.07)

#Plot histograms of momenta from generated/real KAON data
#plot_gen_hist(generated_vars_k[:,6], physical_vars[0], particle_source_1, 750, [0,100000], 12000)
#plot_gen_hist(generated_vars_k[:,7], physical_vars[1], particle_source_1, 750, [0,5000], 7000)
#plot_gen_hist(x_test_1[:,6], physical_vars[0], particle_source_1, 750, [0,100000], 6000)
#plot_gen_hist(x_test_1[:,7], physical_vars[1], particle_source_1, 750, [0,5000], 4000)

#Plot histograms of DLLs from generated PION data
plot_DLL_hist(generated_vars_p[:,0], DLLs[0], ref_particle, particle_source_2, 500, [-80, 20], 0.12)
plot_DLL_hist(generated_vars_p[:,1], DLLs[1], ref_particle, particle_source_2, 500, [-50, 20], 0.25)
plot_DLL_hist(generated_vars_p[:,2], DLLs[2], ref_particle, particle_source_2, 750, [-60, 20], 0.09)
plot_DLL_hist(generated_vars_p[:,3], DLLs[3], ref_particle, particle_source_2, 600, [-60, 40], 0.12)
plot_DLL_hist(generated_vars_p[:,4], DLLs[4], ref_particle, particle_source_2, 600, [-60, 40], 0.12)
plot_DLL_hist(generated_vars_p[:,5], DLLs[5], ref_particle, particle_source_2, 600, [-60, 40], 0.12)

#Plot histograms of momenta from generated/real PION data
#plot_gen_hist(generated_vars_p[:,6], physical_vars[0], particle_source_2, 750, [0,70000], 12000)
#plot_gen_hist(generated_vars_p[:,7], physical_vars[1], particle_source_2, 750, [0,3500], 6000)
#plot_gen_hist(x_test_2[:,6], physical_vars[0], particle_source_2, 750, [0,70000], 7000)
#plot_gen_hist(x_test_2[:,7], physical_vars[1], particle_source_2, 750, [0,3500], 5000)

######################################################################################################################################################################################

plot_DLL_hist(x_test_1[:,2], DLLs[2], ref_particle, particle_source_1, 750, [-40, 80], 0.07)
plot_DLL_hist(generated_vars_k[:,2], DLLs[2], ref_particle, particle_source_1, 750, [-40, 80], 0.07)

plot_two_DLL_hist(generated_vars_k[:,2], x_test_1[:,2], DLLs[2], ref_particle, particle_source_1, 250, [-30, 80], 0.07)
plot_hist_ratio(generated_vars_k[:,2], x_test_1[:,2], DLLs[2], ref_particle, particle_source_1, 100, [-30, 80], 2)
hist_overlap_1 = histogram_intersection(generated_vars_k[:,2], x_test_1[:,2], 750, [-40, 80])

plot_DLL_hist(x_test_2[:,2], DLLs[2], ref_particle, particle_source_2, 750, [-60, 20], 0.09)
plot_DLL_hist(generated_vars_p[:,2], DLLs[2], ref_particle, particle_source_2, 750, [-60, 20], 0.09)

plot_two_DLL_hist(generated_vars_p[:,2], x_test_2[:,2], DLLs[2], ref_particle, particle_source_2, 750, [-60, 20], 0.09)
plot_hist_ratio(generated_vars_p[:,2], x_test_2[:,2], DLLs[2], ref_particle, particle_source_2, 100, [-60, 20], 2)
hist_overlap_2 = histogram_intersection(generated_vars_p[:,2], x_test_2[:,2], 750, [-60, 20])

print("Overlaps:", hist_overlap_1, hist_overlap_2)

print("Data plotted")

#Measure total run time for script
t_final = time.time()
runtime = t_final - t_init
print("Total run time = ", runtime)

#Save runtime as text
with open('GAN1_runtime.txt', 'w') as f:
    print(runtime, file=f)
