#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#Import trained generators and produce plots

#Files assumed:

#Data files for kaon and pion tracks (mod refers to additonal variables added):
# '../../data/mod-PID-train-data-KAONS.hdf'
# '../../data/mod-PID-train-data-PIONS.hdf'

#csv files for normalisation of data (shifts and divisors) calc from datafiles
# '../../data/KAON_norm.csv'
# '../../data/PION_norm.csv'

#May also use mask files to select test data:
#'../../GAN_training/GAN_7DLL/setx/unused_data_mask.csv'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from keras.models import load_model
import math
from matplotlib.ticker import AutoMinorLocator
import time
import os
from matplotlib import gridspec


#Originally: set13.1 generator - works reasonably well for efficiency plot and DLL correlations to each other (both generated)
#set15 for PION
#Now with set17 for KAON (same as 13.1 but with 6DLLs)
#TrackP not correlated with input TrackP


#Time total run
t_init = time.time()

plt.rcParams['agg.path.chunksize'] = 10000 #Needed for plotting lots of data?

ref_particle = 'pi'
particle_source_1 = 'KAON'
particle_source_2 = 'PION'

train_frac = 0.7
examples=2000000

set_text = ""
unused_mask_loc_k = ''
unused_mask_loc_p = ''

RNN = False
sort_var = 'RICH1EntryDist0'

unused_mask_k = False
unused_mask_p = False
generate_P = False
alt_model_k = False
alt_model_p = False

gen_av = False
concat = False

subset=False
sub_var = 'RICH1ExitDist0'
sub_min = None
sub_max = 10

if subset:
    subset_text = '_' + sub_var + '_' + str(sub_min) + '-' + str(sub_max)
else:
    subset_text = ''

print("Loading generators...")

###############################################################################

#epochs = 250 Generate P, Pt 

#generator_k = load_model('../../GAN_training/GAN_7DLL/set17/trained_gan.h5')
#set_text += "set17"
#
#generator_p = load_model('../../GAN_training/GAN_7DLL/set15/trained_gan.h5')
#set_text += "set15"
#
#frac=0.025
#input_physical_vars = ['TrackP', 'TrackPt']
#DLLs = ['e', 'mu', 'k', 'p', 'd', 'bt']
#physical_vars = ['TrackP', 'TrackPt', 'NumLongTracks', 'NumPVs']
#generate_P = True

###############################################################################

#epochs = 500
#
#generator_k = load_model('../../GAN_training/GAN_7DLL/set18/trained_gan.h5')
#set_text += "set18"
#
#generator_p = load_model('../../GAN_training/GAN_7DLL/set19/trained_gan.h5')
#set_text += "set19"
#
#frac=0.1
#input_physical_vars = ['TrackP', 'TrackPt']
#DLLs = ['e', 'mu', 'k', 'p', 'd', 'bt']
#physical_vars = ['TrackP', 'TrackPt', 'NumLongTracks', 'NumPVs']
#generate_P = True

###############################################################################

#epochs = 100

#generator_k = load_model('../../GAN_training/GAN_7DLL/set20/trained_gan.h5')
#set_text += "set20"

#frac=0.025
#input_physical_vars = ['TrackP', 'TrackPt']
#DLLs = ['e', 'mu', 'k', 'p', 'd', 'bt']
#physical_vars = ['TrackP', 'TrackPt', 'NumLongTracks', 'NumPVs']

###############################################################################

#epochs = 500
#
#batch_size = 128
#generator_k = load_model('../../GAN_training/GAN_7DLL/set22/trained_gan.h5')
#set_text += "set22"

#batch_size = 128
#generator_p = load_model('../../GAN_training/GAN_7DLL/set23/trained_gan.h5')
#set_text += "set23"

#batch_size = 1024
#generator_p = load_model('../../GAN_training/GAN_7DLL/set24/trained_gan.h5')
#set_text += "set24"

#batch_size = 32
#generator_p = load_model('../../GAN_training/GAN_7DLL/set25/trained_gan.h5')
#set_text += "set25"

#batch_size = 4096
#generator_p = load_model('../../GAN_training/GAN_7DLL/set26/trained_gan.h5')
#set_text += "set26"
#
#frac = 0.1
#input_physical_vars = ['TrackP', 'TrackPt']
#DLLs = ['e', 'mu', 'k', 'p', 'd', 'bt']
#physical_vars = ['TrackP', 'TrackPt', 'NumLongTracks', 'NumPVs']


###############################################################################

#epochs = 500
#
#generator_k = load_model('../../GAN_training/GAN_7DLL/set28/trained_gan.h5')
#set_text += "set28"
#
#generator_p = load_model('../../GAN_training/GAN_7DLL/set27/trained_gan.h5')
#set_text += "set27"
#
#frac = 0.1
#input_physical_vars = ['TrackP', 'TrackPt', 'NumLongTracks', 'NumPVs']
#DLLs = ['e', 'mu', 'k', 'p', 'd', 'bt']
#physical_vars = input_physical_vars


###############################################################################

#New data

#epochs = 500
#
#generator_k = load_model('../../GAN_training/GAN_7DLL/set31/trained_gan.h5')
#set_text += "set31"
#

#generator_p = load_model('../../GAN_training/GAN_7DLL/set30/trained_gan.h5')
#set_text += "set30"
#
#frac = 0.1
#input_physical_vars = ['TrackP', 'TrackPt', 'NumLongTracks', 'NumPVs']
#DLLs = ['e', 'mu', 'k', 'p', 'd', 'bt']
#physical_vars = input_physical_vars

###############################################################################

#epochs = 500

#generator_k = load_model('../../GAN_training/GAN_7DLL/set34/trained_gan.h5')
#set_text += "set34"
#
##generator_k = load_model('../../GAN_training/GAN_7DLL/set37/trained_gan.h5')
##set_text += "set37"
##alt_model_k = True

generator_k = load_model('../../GAN_training/GAN_7DLL/set44/trained_wgan.h5')
set_text += "set44"
alt_model_k = True

generator_p = load_model('../../GAN_training/GAN_7DLL/set35/trained_gan.h5')
set_text += "set35"

frac = 0.1
input_physical_vars = ['TrackP', 'TrackPt', 'NumLongTracks', 'NumPVs', 'TrackVertexX', 'TrackVertexY', 'TrackVertexZ', 
                 'TrackRich1EntryX', 'TrackRich1EntryY', 'TrackRich1EntryZ', 'TrackRich1ExitX', 'TrackRich1ExitY', 
                 'TrackRich1ExitZ']

physical_vars = input_physical_vars

DLLs = ['e', 'mu', 'k', 'p', 'd', 'bt']

###############################################################################

#Added RICH2

#epochs = 500
#frac = frac = 1 #0.1

#generator_k = load_model('../../GAN_training/GAN_7DLL/set43/trained_gan.h5')
#set_text += "set43"
#
#generator_p = load_model('../../GAN_training/GAN_7DLL/set45/trained_gan.h5')
#set_text += "set45"
#
#generator_k = load_model('../../GAN_training/GAN_7DLL/set58/trained_gan.h5')
#set_text += "set58"
#
#generator_p = load_model('../../GAN_training/GAN_7DLL/set61/trained_gan.h5')
#set_text += "set61"

#generator_k = load_model('../../GAN_training/GAN_7DLL/set63/trained_gan.h5')
#set_text += "set63"
#
#generator_p = load_model('../../GAN_training/GAN_7DLL/set66/trained_gan.h5')
#set_text += "set66"

#generator_k = load_model('../../GAN_training/GAN_7DLL/set64/trained_gan.h5')
#generator_k_2 = load_model('../../GAN_training/GAN_7DLL/set64/penult_trained_gan.h5')
#set_text += "set64"
#set_text += "av"
#set_text += "alt"
#
#generator_p = load_model('../../GAN_training/GAN_7DLL/set67/trained_gan.h5')
#generator_p_2 = load_model('../../GAN_training/GAN_7DLL/set67/penult_trained_gan.h5')
#set_text += "set67"
#set_text += "av"
#set_text += "alt"

#generator_k = load_model('../../GAN_training/GAN_7DLL/set65/trained_gan.h5')
#generator_k_2 = load_model('../../GAN_training/GAN_7DLL/set65/penult_trained_gan.h5')
#set_text += "set65"
#set_text += "av"
#set_text += "alt"

#generator_p = load_model('../../GAN_training/GAN_7DLL/set68/trained_gan.h5')
#generator_p_2 = load_model('../../GAN_training/GAN_7DLL/set68/penult_trained_gan.h5')
#set_text += "set68"
#set_text += "av"
#set_text += "alt"

#generator_k = load_model('../../GAN_training/GAN_7DLL/set69/trained_gan.h5')
#generator_k_2 = load_model('../../GAN_training/GAN_7DLL/set69/penult_trained_gan.h5')
#unused_mask_loc_k = '../../GAN_training/GAN_7DLL/set69/unused_data_mask.csv'
#set_text += "set69"
#set_text += "av"
##set_text += "alt"
#set_text += "new"

#generator_k = load_model('../../GAN_training/GAN_7DLL/set79/trained_gan.h5')
#generator_k_2 = load_model('../../GAN_training/GAN_7DLL/set79/penult_trained_gan.h5')
#unused_mask_loc_k = '../../GAN_training/GAN_7DLL/set79/unused_data_mask.csv'
#set_text += "set79"
#set_text += "av"
##set_text += "alt"
#set_text += "new"
#
#generator_p = load_model('../../GAN_training/GAN_7DLL/set72/trained_gan.h5')
#generator_p_2 = load_model('../../GAN_training/GAN_7DLL/set72/penult_trained_gan.h5')
#unused_mask_loc_p = '../../GAN_training/GAN_7DLL/set72/unused_data_mask.csv'
#set_text += "set72"
#set_text += "av"
##set_text += "alt"
#set_text += "new"

#generator_k = load_model('../../GAN_training/GAN_7DLL/set70/trained_gan.h5')
##generator_k_2 = load_model('../../GAN_training/GAN_7DLL/set70/penult_trained_gan.h5')
#unused_mask_loc_k = '../../GAN_training/GAN_7DLL/set70/unused_data_mask.csv'
#set_text += "set70"
##set_text += "av"
#
#generator_p = load_model('../../GAN_training/GAN_7DLL/set73/trained_gan.h5')
##generator_p_2 = load_model('../../GAN_training/GAN_7DLL/set73/penult_trained_gan.h5')
#unused_mask_loc_p = '../../GAN_training/GAN_7DLL/set73/unused_data_mask.csv'
#set_text += "set73"
##set_text += "av"
##set_text += "alt"


#generator_k = load_model('../../GAN_training/GAN_7DLL/set75/trained_gan.h5')
##generator_k_2 = load_model('../../GAN_training/GAN_7DLL/set75/penult_trained_gan.h5')
#unused_mask_loc_k = '../../GAN_training/GAN_7DLL/set75/unused_data_mask.csv'
#set_text += "set75"
##set_text += "av"
##set_text += "alt"
#
#generator_p = load_model('../../GAN_training/GAN_7DLL/set76/trained_gan.h5')
##generator_p_2 = load_model('../../GAN_training/GAN_7DLL/set76/penult_trained_gan.h5')
#unused_mask_loc_p = '../../GAN_training/GAN_7DLL/set76/unused_data_mask.csv'
#set_text += "set76"
##set_text += "av"
##set_text += "alt"

#generator_k = load_model('../../GAN_training/GAN_7DLL/set88/half_trained_gan.h5')
#generator_k_2 = load_model('../../GAN_training/GAN_7DLL/set88/penult_half_trained_gan.h5')
#set_text += "set88"
#
#generator_k = load_model('../../GAN_training/GAN_7DLL/set88/trained_gan.h5')
#generator_k_2 = load_model('../../GAN_training/GAN_7DLL/set88/penult_trained_gan.h5')
#set_text += "set88-1"
#
#unused_mask_loc_k = '../../GAN_training/GAN_7DLL/set88/unused_data_mask.csv'
#set_text += "av"
#set_text += "alt"

#generator_k = load_model('../../GAN_training/GAN_7DLL/set113/half_trained_gan.h5')
#generator_k_2 = load_model('../../GAN_training/GAN_7DLL/set113/penult_half_trained_gan.h5')
#set_text += "set113"

#generator_k = load_model('../../GAN_training/GAN_7DLL/set113/trained_gan.h5')
#generator_k_2 = load_model('../../GAN_training/GAN_7DLL/set113/penult_trained_gan.h5')
#set_text += "set113-1"

#unused_mask_loc_k = '../../GAN_training/GAN_7DLL/set113/unused_data_mask.csv'
#set_text += "av"
#set_text += "alt"
#alt_model_k = True
#
#generator_p = load_model('../../GAN_training/GAN_7DLL/set98/half_trained_gan.h5')
#generator_p_2 = load_model('../../GAN_training/GAN_7DLL/set98/penult_half_trained_gan.h5')
#set_text += "set98"
#
#generator_p = load_model('../../GAN_training/GAN_7DLL/set98/trained_gan.h5')
#generator_p_2 = load_model('../../GAN_training/GAN_7DLL/set98/penult_trained_gan.h5')
#set_text += "set98-1"
#
#unused_mask_loc_p = '../../GAN_training/GAN_7DLL/set98/unused_data_mask.csv'
#set_text += "av"
#set_text += "alt"
#
#generator_p = load_model('../../GAN_training/GAN_7DLL/set117/trained_gan.h5')
#generator_p_2 = load_model('../../GAN_training/GAN_7DLL/set117/penult_trained_gan.h5')
#set_text += "set117"
#
#unused_mask_loc_p = '../../GAN_training/GAN_7DLL/set117/unused_data_mask.csv'
#set_text += "av"
#set_text += "alt"
#alt_model_p = True

#gen_av = True
#concat = True
#unused_mask_k = True
#unused_mask_p = True
##
#DLLs = ['e', 'mu', 'k', 'p', 'd', 'bt']
##
#input_physical_vars = ['TrackP', 'TrackPt', 'NumLongTracks', 'NumPVs', 'TrackVertexX', 'TrackVertexY', 'TrackVertexZ', 
#                       'TrackRich1EntryX', 'TrackRich1EntryY', 'TrackRich1EntryZ', 'TrackRich1ExitX', 'TrackRich1ExitY', 
#                       'TrackRich1ExitZ', 'TrackRich2EntryX', 'TrackRich2EntryY', 'TrackRich2EntryZ', 'TrackRich2ExitX', 
#                       'TrackRich2ExitY', 'TrackRich2ExitZ']
#
##physical_vars = input_physical_vars
#
#physical_vars = ['TrackP', 'TrackPt', 'NumLongTracks', 'NumPVs', 'TrackVertexX', 'TrackVertexY', 'TrackVertexZ', 
#                 'TrackRich1EntryX', 'TrackRich1EntryY', 'TrackRich1EntryZ', 'TrackRich1ExitX', 'TrackRich1ExitY', 
#                 'TrackRich1ExitZ', 'TrackRich2EntryX', 'TrackRich2EntryY', 'TrackRich2EntryZ', 'TrackRich2ExitX', 
#                 'TrackRich2ExitY', 'TrackRich2ExitZ', 'RICH1EntryDist0', 'RICH1ExitDist0', 'RICH2EntryDist0',
#                 'RICH2ExitDist0', 'RICH1EntryDist1', 'RICH1ExitDist1', 'RICH2EntryDist1', 'RICH2ExitDist1', 
#                 'RICH1EntryDist2', 'RICH1ExitDist2', 'RICH2EntryDist2', 'RICH2ExitDist2', 'RICH1ConeNum',
#                 'RICH2ConeNum']

###############################################################################

#All old and new data expect removed spacial coods

#epochs = 500
#frac = frac = 1 #0.1
#
#generator_k = load_model('../../GAN_training/GAN_7DLL/set59/trained_gan.h5')
#set_text += "set59"
#
#generator_p = load_model('../../GAN_training/GAN_7DLL/set62/trained_gan.h5')
#set_text += "set62"
#
#generator_k = load_model('../../GAN_training/GAN_7DLL/set71/trained_gan.h5')
#generator_k_2 = load_model('../../GAN_training/GAN_7DLL/set71/penult_trained_gan.h5')
#unused_mask_loc_k = '../../GAN_training/GAN_7DLL/set71/unused_data_mask.csv'
#set_text += "set71"
#set_text += "av"
##set_text += "alt"
#set_text += "test"
#
#generator_p = load_model('../../GAN_training/GAN_7DLL/set74/trained_gan.h5')
#generator_p_2 = load_model('../../GAN_training/GAN_7DLL/set74/penult_trained_gan.h5')
#unused_mask_loc_p = '../../GAN_training/GAN_7DLL/set74/unused_data_mask.csv'
#set_text += "set74"
#set_text += "av"
##set_text += "alt"
#set_text += "test"
#
#
#generator_k = load_model('../../GAN_training/GAN_7DLL/set118/half_trained_gan.h5')
#generator_k_2 = load_model('../../GAN_training/GAN_7DLL/set118/penult_half_trained_gan.h5')
#unused_mask_loc_k = '../../GAN_training/GAN_7DLL/set118/unused_data_mask.csv'
#alt_model_k = True
#set_text += "set118"
#set_text += "av"
##set_text += "alt"
##set_text += "test"
##
#generator_p = load_model('../../GAN_training/GAN_7DLL/set119/half_trained_gan.h5')
#generator_p_2 = load_model('../../GAN_training/GAN_7DLL/set119/penult_half_trained_gan.h5')
#alt_model_p = True
#unused_mask_loc_p = '../../GAN_training/GAN_7DLL/set119/unused_data_mask.csv'
#set_text += "set119"
#set_text += "av"
##set_text += "alt"
##set_text += "test"

#gen_av = True
#concat = True
#
#input_physical_vars = ['TrackP', 'TrackPt', 'NumLongTracks', 'NumPVs', 'RICH1EntryDist0', 'RICH1ExitDist0', 
#                       'RICH2EntryDist0', 'RICH2ExitDist0', 'RICH1EntryDist1', 'RICH1ExitDist1', 'RICH2EntryDist1', 
#                       'RICH2ExitDist1',  'RICH1EntryDist2', 'RICH1ExitDist2', 'RICH2EntryDist2', 'RICH2ExitDist2', 
#                       'RICH1ConeNum', 'RICH2ConeNum']
#
##physical_vars = input_physical_vars
#
#DLLs = ['e', 'mu', 'k', 'p', 'd', 'bt']
#
#physical_vars = ['TrackP', 'TrackPt', 'NumLongTracks', 'NumPVs', 'TrackVertexX', 'TrackVertexY', 'TrackVertexZ', 
#                 'TrackRich1EntryX', 'TrackRich1EntryY', 'TrackRich1EntryZ', 'TrackRich1ExitX', 'TrackRich1ExitY', 
#                 'TrackRich1ExitZ', 'TrackRich2EntryX', 'TrackRich2EntryY', 'TrackRich2EntryZ', 'TrackRich2ExitX', 
#                 'TrackRich2ExitY', 'TrackRich2ExitZ', 'RICH1EntryDist0', 'RICH1ExitDist0', 'RICH2EntryDist0',
#                 'RICH2ExitDist0', 'RICH1EntryDist1', 'RICH1ExitDist1', 'RICH2EntryDist1', 'RICH2ExitDist1', 
#                 'RICH1EntryDist2', 'RICH1ExitDist2', 'RICH2EntryDist2', 'RICH2ExitDist2', 'RICH1ConeNum',
#                 'RICH2ConeNum']

###############################################################################

#Added corr info

#epochs = 500
#frac = 1 #0.1
#
#generator_k = load_model('../../GAN_training/GAN_7DLL/set47/trained_gan.h5')
#set_text += "set47"
#
#generator_p = load_model('../../GAN_training/GAN_7DLL/set46/trained_gan.h5')
#set_text += "set46"
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
###############################################################################

#Added corr info but only used NN data

#epochs = 500
#frac = 1 #0.1
#
#generator_k = load_model('../../GAN_training/GAN_7DLL/set48/trained_gan.h5')
#set_text += "set48"
#
#generator_p = load_model('../../GAN_training/GAN_7DLL/set49/trained_gan.h5')
#set_text += "set49"
#
#Reduced inner layers to 4
#generator_k = load_model('../../GAN_training/GAN_7DLL/set50/trained_gan.h5')
#set_text += "set50"

#Reduced inner layers to 4
#generator_p = load_model('../../GAN_training/GAN_7DLL/set51/trained_gan.h5')
#set_text += "set51"

#Reduced inner layers to 6
#generator_k = load_model('../../GAN_training/GAN_7DLL/set52/trained_gan.h5')
#set_text += "set52"
#
#Reduced inner layers to 6
#generator_p = load_model('../../GAN_training/GAN_7DLL/set53/trained_gan.h5')
#set_text += "set53"
#
##Reduced inner layers to 4, frac = 0.2
#generator_k = load_model('../../GAN_training/GAN_7DLL/set54/trained_gan.h5')
#set_text += "set54"
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

###############################################################################

#8 layers again, all data inc all NN data

#epochs = 500 #(1000 for full gen)
#frac = 1 #0.2

#generator_k = load_model('../../GAN_training/GAN_7DLL/set55/half_trained_gan.h5')
#set_text += "set55"
##generator_k = load_model('../../GAN_training/GAN_7DLL/set55/trained_gan.h5')
##set_text += "set55-1"
#
#set_text += "new"
#
#generator_p = load_model('../../GAN_training/GAN_7DLL/set56/half_trained_gan.h5')
#set_text += "set56"
#
##generator_p = load_model('../../GAN_training/GAN_7DLL/set56/trained_gan.h5')
##set_text += "set56-1"
#set_text += "new"
#
##gen_av = True
##concat = True
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

###############################################################################

#RNN. Noise=250

##Choose GPU to use
#os.environ["CUDA_VISIBLE_DEVICES"]="1"
#
#RNN=True
#seq_length = 4 #Rows, default 32 (//4)
#batch_size = 128
#apparent_batch_size = batch_size - seq_length + 1
#sort_var = 'RICH1EntryDist0'
#
#generator_k = load_model('../../GAN_training/set90/trained_gan.h5')
##generator_k_2 = load_model('../../GAN_training/set90/penult_trained_gan.h5')
#unused_mask_loc_k = '../../GAN_training/set90/unused_data_mask.csv'
#set_text += "set90"
##set_text += "av"
###set_text += "alt"
#
#
#generator_p = load_model('../../GAN_training/set93/trained_gan.h5')
##generator_p_2 = load_model('../../GAN_training/set93/penult_trained_gan.h5')
#unused_mask_loc_p = '../../GAN_training/set93/unused_data_mask.csv'
#set_text += "set93"
##set_text += "av"
###set_text += "alt"

#generator_k = load_model('../../GAN_training/set96/trained_gan.h5')
##generator_k_2 = load_model('../../GAN_training/set96/penult_trained_gan.h5')
#unused_mask_loc_k = '../../GAN_training/set96/unused_data_mask.csv'
#set_text += "set96"
##set_text += "av"
###set_text += "alt"
#
#
#generator_p = load_model('../../GAN_training/set97/trained_gan.h5')
##generator_p_2 = load_model('../../GAN_training/set97/penult_trained_gan.h5')
#unused_mask_loc_p = '../../GAN_training/set97/unused_data_mask.csv'
#set_text += "set97"
##set_text += "av"
###set_text += "alt"
#
#
#unused_mask_k = True
#unused_mask_p = True
#gen_av = False
#concat = True
#
#
#input_physical_vars = ['TrackP', 'TrackPt', 'NumLongTracks', 'NumPVs', 'TrackVertexX', 'TrackVertexY', 'TrackVertexZ', 
#                       'TrackRich1EntryX', 'TrackRich1EntryY', 'TrackRich1EntryZ', 'TrackRich1ExitX', 'TrackRich1ExitY', 
#                       'TrackRich1ExitZ', 'TrackRich2EntryX', 'TrackRich2EntryY', 'TrackRich2EntryZ', 'TrackRich2ExitX', 
#                       'TrackRich2ExitY', 'TrackRich2ExitZ']
#
##physical_vars = input_physical_vars
#
#DLLs = ['e', 'mu', 'k', 'p', 'd', 'bt']
#
#physical_vars = ['TrackP', 'TrackPt', 'NumLongTracks', 'NumPVs', 'TrackVertexX', 'TrackVertexY', 'TrackVertexZ', 
#                 'TrackRich1EntryX', 'TrackRich1EntryY', 'TrackRich1EntryZ', 'TrackRich1ExitX', 'TrackRich1ExitY', 
#                 'TrackRich1ExitZ', 'TrackRich2EntryX', 'TrackRich2EntryY', 'TrackRich2EntryZ', 'TrackRich2ExitX', 
#                 'TrackRich2ExitY', 'TrackRich2ExitZ', 'RICH1EntryDist0', 'RICH1ExitDist0', 'RICH2EntryDist0',
#                 'RICH2ExitDist0', 'RICH1EntryDist1', 'RICH1ExitDist1', 'RICH2EntryDist1', 'RICH2ExitDist1', 
#                 'RICH1EntryDist2', 'RICH1ExitDist2', 'RICH2EntryDist2', 'RICH2ExitDist2', 'RICH1ConeNum',
#                 'RICH2ConeNum']

###############################################################################

#All orig data and new data inc run/even num, alt models:

#generator_k = load_model('../../GAN_training/GAN_7DLL/set114/half_trained_gan.h5')
#generator_k_2 = load_model('../../GAN_training/GAN_7DLL/set114/penult_half_trained_gan.h5')
#set_text += "set114"
##
##generator_k = load_model('../../GAN_training/GAN_7DLL/set114/trained_gan.h5')
##generator_k_2 = load_model('../../GAN_training/GAN_7DLL/set114/penult_trained_gan.h5')
##set_text += "set114-1"
#
#unused_mask_loc_k = '../../GAN_training/GAN_7DLL/set114/unused_data_mask.csv'
#set_text += "av"
###set_text += "alt"
#alt_model_k = True
#
#
#generator_p = load_model('../../GAN_training/GAN_7DLL/set116/half_trained_gan.h5')
#generator_p_2 = load_model('../../GAN_training/GAN_7DLL/set116/penult_half_trained_gan.h5')
#set_text += "set116"
#
##generator_p = load_model('../../GAN_training/GAN_7DLL/set116/trained_gan.h5')
##generator_p_2 = load_model('../../GAN_training/GAN_7DLL/set116/penult_trained_gan.h5')
##set_text += "set116-1"
#
#unused_mask_loc_p = '../../GAN_training/GAN_7DLL/set116/unused_data_mask.csv'
#set_text += "av"
#set_text += "alt"
#alt_model_p = True
#
#


#generator_k = load_model('../../GAN_training/GAN_7DLL/set115/trained_gan.h5')
#generator_k_2 = load_model('../../GAN_training/GAN_7DLL/set115/penult_trained_gan.h5')
#set_text += "set115"

#unused_mask_loc_k = '../../GAN_training/GAN_7DLL/set115/unused_data_mask.csv'
#set_text += "av"
#set_text += "alt"
#alt_model_k = True

#unused_mask_k = True
#unused_mask_p = True
#gen_av = True
#concat = True
#
#input_physical_vars = ['RunNumber', 'EventNumber', 'TrackP', 'TrackPt', 'NumLongTracks', 'NumPVs', 'TrackVertexX',
#                 'TrackVertexY', 'TrackVertexZ', 'TrackRich1EntryX', 'TrackRich1EntryY', 'TrackRich1EntryZ',
#                 'TrackRich1ExitX', 'TrackRich1ExitY', 'TrackRich1ExitZ', 'TrackRich2EntryX', 'TrackRich2EntryY',
#                 'TrackRich2EntryZ', 'TrackRich2ExitX', 'TrackRich2ExitY', 'TrackRich2ExitZ', 'RICH1EntryDist0',
#                 'RICH1ExitDist0', 'RICH2EntryDist0', 'RICH2ExitDist0', 'RICH1EntryDist1', 'RICH1ExitDist1',
#                 'RICH2EntryDist1', 'RICH2ExitDist1', 'RICH1EntryDist2', 'RICH1ExitDist2', 'RICH2EntryDist2',
#                 'RICH2ExitDist2', 'RICH1ConeNum', 'RICH2ConeNum']
#
#DLLs = ['e', 'mu', 'k', 'p', 'd', 'bt']
#
#physical_vars = input_physical_vars
#
###############################################################################

#WGAN testing... Need alt PION stuff

generator_k = load_model('../../GAN_training/GAN_7DLL/set103/trained_wgan.h5')
generator_k_2 = load_model('../../GAN_training/GAN_7DLL/set103/penult_trained_wgan.h5')
set_text += "set103"

unused_mask_loc_k = '../../GAN_training/GAN_7DLL/set103/unused_data_mask.csv'
set_text += "av"
##set_text += "alt"
alt_model_k = True
#
#
#generator_p = load_model('../../GAN_training/GAN_7DLL/set116/half_trained_gan.h5')
#generator_p_2 = load_model('../../GAN_training/GAN_7DLL/set116/penult_half_trained_gan.h5')
#set_text += "set116"
#
#unused_mask_loc_p = '../../GAN_training/GAN_7DLL/set116/unused_data_mask.csv'
#set_text += "av"
#set_text += "alt"
#alt_model_p = True
#

unused_mask_k = True
unused_mask_p = True
gen_av = True
concat = True

input_physical_vars = ['RunNumber', 'EventNumber', 'TrackP', 'TrackPt', 'NumLongTracks', 'NumPVs', 'TrackVertexX',
                 'TrackVertexY', 'TrackVertexZ', 'TrackRich1EntryX', 'TrackRich1EntryY', 'TrackRich1EntryZ',
                 'TrackRich1ExitX', 'TrackRich1ExitY', 'TrackRich1ExitZ', 'TrackRich2EntryX', 'TrackRich2EntryY',
                 'TrackRich2EntryZ', 'TrackRich2ExitX', 'TrackRich2ExitY', 'TrackRich2ExitZ']

DLLs = ['e', 'mu', 'k', 'p', 'd', 'bt']

physical_vars = input_physical_vars

###############################################################################


print("Generators loaded")

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

input_phys_index = []

for i in range(input_phys_dim):
    for k in range(phys_dim):
        if input_physical_vars[i] == physical_vars[k]:
            input_phys_index.append(k)
            break

#If averaging, half examples each.
if gen_av:
    if not concat:
        examples = examples//2

# =============================================================================
# Functions
# =============================================================================

#Import all data via pandas from data files
#Inputs: Particle source e.g. KAONS corresponding to the datafile from which 
#        data will be imported
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


#Create sequences of data needed for RNN
#Inputs: data array to be made into sequences, length of sequences (=look_back)
#Returns: arrays of sequenced data, array containing the final row of each seq
#Note: Number of sequences = original number of rows - look_back + 1 
#      e.g. input 10 rows, lookback = 4 -> 7 output
def create_dataset(dataset, look_back=1):
  
    dataX, dataY = [], []
    
    for i in range(len(dataset)-look_back+1):
    
        #Extract [look_back] data rows starting from the ith row
        a = dataset[i:(i+look_back), :]
        
        
        dataX.append(a)
        dataY.append(dataset[i + look_back - 1, :])
    
    return np.array(dataX), np.array(dataY)


#Change DLLs e.g. from (K-pi) and (p-pi) to p-K
#Input: Two DLL arrays w.r.t. pi, to be changed s.t. the new DLL is w.r.t. the first particle in each DLL
#Returns: New DLL array e.g. DLL(p-K)
def change_DLL(DLL1, DLL2):
    
    if(not np.array_equal(DLL1, DLL2)):
        DLL3 = np.subtract(DLL1, DLL2)
    else:
        print("DLLs are the same!")
        DLL3 = DLL1
    
    return DLL3


#Import information needed to normalise all data to between -1 and 1
#Input: particle source i.e. either KAON or PION so read in respective csv file with vales
#Returns: div_num and shift needed to normalise all relevant data (DLLs and input physics)
def import_norm_info(particle_source):

    #Read in csv datafile
    data_norm = np.array(pd.read_csv('../../data/' + particle_source + '_norm.csv'))
    
    #shift = [0,x], div_num = [1,x], where x starts at 1 for meaningful data

    #Order of variables:
    columns = ['RunNumber', 'EventNumber', 'MCPDGCode', 'NumPVs', 'NumLongTracks', 'NumRich1Hits', 
               'NumRich2Hits', 'TrackP', 'TrackPt', 'TrackChi2PerDof', 'TrackNumDof', 'TrackVertexX', 
               'TrackVertexY', 'TrackVertexZ', 'TrackRich1EntryX', 'TrackRich1EntryY', 'TrackRich1EntryZ',
               'TrackRich1ExitX', 'TrackRich1ExitY', 'TrackRich1ExitZ', 'TrackRich2EntryX', 'TrackRich2EntryY', 
               'TrackRich2EntryZ','TrackRich2ExitX', 'TrackRich2ExitY', 'TrackRich2ExitZ', 'RichDLLe', 
               'RichDLLmu', 'RichDLLk', 'RichDLLp', 'RichDLLd', 'RichDLLbt', 'RICH1EntryDist0', 'RICH1ExitDist0', 
               'RICH2EntryDist0', 'RICH2ExitDist0', 'RICH1EntryDist1', 'RICH1ExitDist1', 'RICH2EntryDist1', 
               'RICH2ExitDist1', 'RICH1EntryDist2', 'RICH1ExitDist2', 'RICH2EntryDist2', 'RICH2ExitDist2', 
               'RICH1ConeNum', 'RICH2ConeNum']
    
    #Create arrays for shift and div_num to be stored in. Only need to save DLLs and physics input values (data_dim)
    shift = np.zeros(data_dim)
    div_num = np.zeros(data_dim)
    
    #Loop over all DLLs and input physics
    for i in range(data_dim):
        #First values correspond to DLLs
        if i < DLLs_dim:
            for j in range(len(columns)):
                if columns[j] == 'RichDLL' + DLLs[i]:
                    shift[i] = data_norm[0,j+1]
                    div_num[i] = data_norm[1,j+1]
                    break
        #Next set correspond to physics inputs
        else:
            for k in range(len(columns)):
                if columns[k] == physical_vars[i-DLLs_dim]:
                    shift[i] = data_norm[0,k+1]
                    div_num[i] = data_norm[1,k+1]
                    break

    return shift, div_num


#Normalise relevant data via dividing centre on zero and divide by max s.t. range=[-1,1]
#Input: Data array to be normalised (x) and particle source, so know which set of normalisation values to use
#Returns: Normalised data array (x) and shift/div_num used to do so (so can unnormalise later)
def norm(x, particle_source):

    #Import normalistion arrays (shift and div_number) from csv file        
    shift, div_num, = import_norm_info(particle_source)
    
    #For each column in input data array, normalise by shifting and dividing
    for i in range(x.shape[1]):
        
        x[:,i] = np.subtract(x[:,i], shift[i])
        x[:,i] = np.divide(x[:,i], div_num[i])
    
    return x, shift, div_num



#Get all relevant test data
#Inputs: List of DLLs of interest, list of physical vars of interest, particle source for data e.g. KAONS
#Returns: test data, as well as values used to normalise the data
def get_x_data(DLLs, ref_particle, physical_vars, particle_source, examples, unused_mask, unused_mask_loc):
    
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

        unused_data_mask = np.array(pd.read_csv(unused_mask_loc))
        unused_data_mask = np.array(unused_data_mask[:,1], dtype=bool)

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
    

# =============================================================================
# Plotting
# =============================================================================

#Make scatter plot w/ colour of correlations between two variables (e.g. DLLs)
def col_scatt(var1, var2, part_source_1, DLL_particle_1, ref_particle_1, part_source_2,  DLL_particle_2, ref_particle_2, max_var_index, real_gen_text, x_range=None, y_range=None, zero_lines=0, save_index=0, size=1):
    
    title = "./plots/" + set_text + "_" + part_source_1 + DLL_particle_1 + "_" + part_source_1 + DLL_particle_2 + "_" + real_gen_text + subset_text + "_colour.eps"        
    
    if real_gen_text == 'gen':
        part_source_1_text = "Generated "
        part_source_2_text = "Generated "
    elif real_gen_text == 'real':
        part_source_1_text = "Real "
        part_source_2_text = "Real "
    elif real_gen_text == 'real_gen':
        part_source_1_text = "Generated "
        part_source_2_text = "Real "
        
    if part_source_1 == 'KAON':
        part_source_1_text += 'Kaon'
    elif part_source_1 == 'PION':
        part_source_1_text += 'Pion'

    if part_source_2 == 'KAON':
        part_source_2_text += 'Kaon'
    elif part_source_2 == 'PION':
        part_source_2_text += 'Pion'
    
    if ref_particle_1 == 'pi':
        ref_particle_1_text = r'$\pi ) $'
    elif ref_particle_1 == 'mu':
        ref_particle_1_text = r'$\mu ) $'
    elif ref_particle_1 == 'k':
        ref_particle_1_text = 'K)'
    else:
        ref_particle_1_text = ref_particle_1 + ')'

    if DLL_particle_1 == 'pi':
        DLL_particle_1_text = r'$\pi - $'
    elif DLL_particle_1 == 'mu':
        DLL_particle_1_text = r'$\mu - $'
    elif DLL_particle_1 == 'k':
        DLL_particle_1_text = r'$K-$'
    else:
        DLL_particle_1_text = DLL_particle_1 + r'$-$'
    
    DLL_1_text = part_source_1_text + " " + r'$\Delta LL ($' + DLL_particle_1_text + ref_particle_1_text
    
    if ref_particle_2 == 'pi':
        ref_particle_2_text = r'$\pi ) $'
    elif ref_particle_2 == 'mu':
        ref_particle_2_text = r'$\mu ) $'
    elif ref_particle_2 == 'k':
        ref_particle_2_text = 'K)'
    else:
        ref_particle_2_text = ref_particle_2 + ')'

    if DLL_particle_2 == 'pi':
        DLL_particle_2_text = r'$\pi - $'
    elif DLL_particle_2 == 'mu':
        DLL_particle_2_text = r'$\mu - $'
    elif DLL_particle_2 == 'k':
        DLL_particle_2_text = r'$K-$'
    else:
        DLL_particle_2_text = DLL_particle_2 + r'$-$'
    
    DLL_2_text = part_source_2_text + " " + r'$\Delta LL ($' + DLL_particle_2_text + ref_particle_2_text
    
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
    ax1.set_xlabel(DLL_1_text, fontsize=13)
    ax1.set_ylabel(DLL_2_text, fontsize=13)
    ax1.set_xlim(x_range)
    ax1.set_ylim(y_range)
    
    if(zero_lines):
        ax1.axhline(lw=1.0, color='k',ls='--')
        ax1.axvline(lw=1.0, color='k',ls='--')
    
    if(max_var_index >= save_index):
        fig1.savefig(title, format='eps', dpi=1000)

    return 0

######################################################################################################################################################################################

#Plot efficiency against momentum
def eff_mom_plot(p_points, source_1_eff_0, source_1_eff_5, source_2_eff_0, source_2_eff_5, DLL_part_1, DLL_part_2, particle_source_1, particle_source_2, p_max, p_label):
 
    title = DLL_part_1 + "_" + DLL_part_2 + "_" + particle_source_1 + "_" + particle_source_2 + "_" + str(int(p_max)) + p_label
    
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
        DLL_part_2 = r'$\pi) $'
    elif(DLL_part_2 == 'k'):
        DLL_part_2 = 'K'
    
    process_1_text = particle_source_1 + " " + r'$\rightarrow$' + " " + DLL_part_1
    process_2_text = particle_source_2 + " " + r'$\rightarrow$' + " " + DLL_part_1
    DLL_text = r'$\Delta LL ($' + DLL_part_1 + '-' + DLL_part_2
    
    fig1, ax1 = plt.subplots()
    ax1.cla()
    ax1.set_ylim(0,1.4)
    ax1.set_xlabel('Momentum (GeV/c)', fontsize=13)
    ax1.set_ylabel('Efficiency', fontsize=13)
    ax1.yaxis.set_minor_locator(AutoMinorLocator(4))
    
    s1_0 = ax1.scatter(p_points, source_1_eff_0, s = 5, marker = 'o', facecolors = 'none', edgecolors = 'r')
    s1_5 = ax1.scatter(p_points, source_1_eff_5, s = 5, marker = 'o', color = 'r')
    s2_0 = ax1.scatter(p_points, source_2_eff_0, s = 5, marker = 's', facecolors = 'none', edgecolors = 'k')
    s2_5 = ax1.scatter(p_points, source_2_eff_5, s = 5, marker = 's', color = 'k')    
    ax1.legend((s1_0, s1_5, s2_0, s2_5), (process_1_text + ', ' + DLL_text + ' > 0', process_1_text + ', ' + DLL_text + ' > 5', process_2_text + ', ' + DLL_text + ' > 0', process_2_text + ', ' + DLL_text + ' > 5'), loc='upper right', ncol=2, fontsize=11)
    fig1.savefig("./plots/" + set_text + "_" + title + subset_text + ".eps", format='eps', dpi=1000)
    
    return 0

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


#Calculate and plot efficiency e.g. for K -> K
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

    return 0


#Plot calc and plot general id/mis-id plots e.g. for numpvs
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
        plot_title = "Nearest track at RICH 1 entry / mm"
    elif var_name == 'RICH1ExitDist0':
        plot_title = "Nearest track at RICH 1 exit / mm"
    elif var_name == 'RICH2EntryDist0':
        plot_title = "Nearest track at RICH 2 entry / mm"
    elif var_name == 'RICH2ExitDist0':
        plot_title = "Nearest track at RICH 2 exit / mm"

    elif var_name == 'RICH1EntryDist1':
        plot_title = "2nd nearest track at RICH 1 entry / mm"
    elif var_name == 'RICH1ExitDist1':
        plot_title = "2nd nearest track at RICH 1 exit / mm"
    elif var_name == 'RICH2EntryDist1':
        plot_title = "2nd nearest track at RICH 2 entry / mm"
    elif var_name == 'RICH2ExitDist1':
        plot_title = "2nd nearest track at RICH 2 exit / mm"

    elif var_name == 'RICH1EntryDist2':
        plot_title = "3rd nearest track at RICH 1 entry /mm"
    elif var_name == 'RICH1ExitDist2':
        plot_title = "3rd nearest track at RICH 1 exit / mm"
    elif var_name == 'RICH2EntryDist2':
        plot_title = "3rd nearest track at RICH 2 entry / mm"
    elif var_name == 'RICH2ExitDist2':
        plot_title = "3rd nearest track at RICH 2 exit / mm"
        
    elif var_name == 'RICH1ConeNum':
        plot_title = "Tracks in RICH 1 Cone"
    elif var_name == 'RICH2ConeNum':
        plot_title = "Tracks in RICH 2 Cone"
        
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
    ax1.set_xlabel('Kaon ID Efficiency', fontsize=13)
    ax1.set_ylabel('Pion Mis-ID Efficiency', fontsize=13)
#    ax1.xaxis.set_minor_locator(AutoMinorLocator(4))
    
    if bins_no == 4:
        ax1.semilogy(source_1_eff_av[0,:], source_2_eff_av[0,:], 'yo-', markersize=4, label=labels[0])
        ax1.semilogy(source_1_eff_av[1,:], source_2_eff_av[1,:], 'rs-', markersize=4, label=labels[1])
        ax1.semilogy(source_1_eff_av[2,:], source_2_eff_av[2,:], 'b^-', markersize=4, label=labels[2])
        ax1.semilogy(source_1_eff_av[3,:], source_2_eff_av[3,:], 'gv-', markersize=4, label=labels[3])
    else:
        for i in range(bins_no):
            ax1.semilogy(source_1_eff_av[i,:], source_2_eff_av[i,:], markersize=4, label=labels[i])
            
    ax1.legend(title=plot_title, loc='upper left', fontsize=11)
    
    fig1.savefig("./plots/" + set_text + "_" + "kID_pMID_eff_" + var_name + "_" + DLL_label + subset_text + ".eps", format='eps', dpi=1000)

    return 0


#Plot histogram of a single DLL
def plot_DLL_hist(DLL_data, DLL_particle, ref_particle, particle_source, bin_no=200, x_range=None, y_range=None):
        
    title = "DLL" + DLL_particle + "-" + ref_particle + "_" + particle_source + subset_text + "_hist.eps"
        
    if ref_particle == 'pi':
        ref_particle_text = r'$\pi ) $'
    elif ref_particle == 'mu':
        ref_particle_text = r'$\mu ) $'
    elif ref_particle == 'k':
        ref_particle_text = 'K)'
    else:
        ref_particle_text = ref_particle + ')'

    if DLL_particle == 'pi':
        DLL_particle_text = r'$\pi - $'
    elif DLL_particle == 'mu':
        DLL_particle_text = r'$\mu - $'
    elif DLL_particle == 'k':
        DLL_particle_text = r'$K-$'
    else:
        DLL_particle_text = DLL_particle + r'$-$'
    
    DLL_text = r'$\Delta LL ($' + DLL_particle_text + ref_particle_text
            
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

    return 0


#Plot two histogram between - real and generated DLLs on the same plot
def plot_two_DLL_hist(DLL_data_1, DLL_data_2, DLL_particle, ref_particle, particle_source, legend_loc, bin_no=200, x_range=None, y_range=None):
        
    title = "DLL" + DLL_particle + "-" + ref_particle + "_" + particle_source + subset_text + "_two_hist.pdf"
    
    if ref_particle == 'pi':
        ref_particle_text = r'$\pi ) $'
    elif ref_particle == 'mu':
        ref_particle_text = r'$\mu ) $'
    elif ref_particle == 'k':
        ref_particle_text = 'K)'
    else:
        ref_particle_text = ref_particle + ')'

    if DLL_particle == 'pi':
        DLL_particle_text = r'$\pi - $'
    elif DLL_particle == 'mu':
        DLL_particle_text = r'$\mu - $'
    elif DLL_particle == 'k':
        DLL_particle_text = r'$K-$'
    else:
        DLL_particle_text = DLL_particle + r'$-$'
    
    DLL_text = r'$\Delta LL ($' + DLL_particle_text + ref_particle_text
        
    labels = ['Generated', 'Real']
    
    fig1, ax1 = plt.subplots()
    ax1.cla()
    
    if y_range is not None:
        ax1.set_ylim(bottom=0, top=y_range)

    if x_range is not None:
        ax1.set_xlim(x_range)
    
    ax1.set_xlabel(DLL_text)
    ax1.set_ylabel("Density of events")
    
    ax1.hist(DLL_data_1, bins=bin_no, range=x_range, density=True, histtype='step', color = 'b', linewidth=0.5, label=labels[0])
    ax1.hist(DLL_data_2, bins=bin_no, range=x_range, density=True, histtype='step', color='r', linewidth=0.5, label=labels[1])

    ax1.legend(loc=legend_loc)

#    ax1.hist(DLL_data_1, bins=bin_no, range=x_range, density=True, histtype='stepfilled', alpha=0.8, color = 'r')
#    ax1.hist(DLL_data_2, bins=bin_no, range=x_range, density=True, histtype='stepfilled', alpha=0.6, color='b')

    fig1.savefig("./plots/" + set_text + "_" + title, format='pdf', dpi=2500)

    return 0


#Plot four histogram between - two sets of real and generated DLLs on the same plot
def plot_four_DLL_hist(DLL_data_1_gen, DLL_data_1_real, DLL_data_2_gen, DLL_data_2_real, DLL_particle, ref_particle, particle_source, legend_loc, bin_no=200, x_range=None, y_range=None):

    title = "DLL" + DLL_particle + "-" + ref_particle + "_" + particle_source + subset_text + "_four_hist.pdf"

    if ref_particle == 'pi':
        ref_particle_text = r'$\pi ) $'
    elif ref_particle == 'mu':
        ref_particle_text = r'$\mu ) $'
    elif ref_particle == 'k':
        ref_particle_text = 'K)'
    else:
        ref_particle_text = ref_particle + ')'

    if DLL_particle == 'pi':
        DLL_particle_text = r'$\pi - $'
    elif DLL_particle == 'mu':
        DLL_particle_text = r'$\mu - $'
    elif DLL_particle == 'k':
        DLL_particle_text = r'$K-$'
    else:
        DLL_particle_text = DLL_particle + r'$-$'
    
    DLL_text = r'$\Delta LL ($' + DLL_particle_text + ref_particle_text

    labels = ['Kaon generated', 'Kaon real', 'Pion generated', 'Pion real']

    fig1, ax1 = plt.subplots()
    ax1.cla()

    if y_range is not None:
        ax1.set_ylim(bottom=0, top=y_range)

    if x_range is not None:
        ax1.set_xlim(x_range)

    ax1.set_xlabel(DLL_text)
    ax1.set_ylabel("Density of events")

    ax1.hist(DLL_data_1_gen, bins=bin_no, range=x_range, density=True, histtype='step', color = 'r', linewidth=1, label=labels[0])
    ax1.hist(DLL_data_1_real, bins=bin_no, range=x_range, density=True, histtype='bar', alpha=0.3, color = 'r', linewidth=0.5, label=labels[1])
    ax1.hist(DLL_data_2_gen, bins=bin_no, range=x_range, density=True, histtype='step', color = 'b', linewidth=1, label=labels[2])
    ax1.hist(DLL_data_2_real, bins=bin_no, range=x_range, density=True, histtype='bar', alpha=0.3, color='b', linewidth=0.5, label=labels[3])

    ax1.legend(loc=legend_loc)

#    ax1.hist(DLL_data_1, bins=bin_no, range=x_range, density=True, histtype='stepfilled', alpha=0.8, color = 'r')
#    ax1.hist(DLL_data_2, bins=bin_no, range=x_range, density=True, histtype='stepfilled', alpha=0.6, color='b')

    fig1.savefig("./plots/" + set_text + "_" + title, format='pdf', dpi=2500)

    return 0


#Plot histogram between two general variables
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


#Plot ratios for two sets of real/generated data
#Using pdfs since /0 errors breaks eps
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

    return 0


#Plot histogram and ratios for two sets of real/generated data
#Using pdfs since /0 errors breaks eps
def plot_hist_ratios(DLL_data_1_gen, DLL_data_1_real, DLL_data_2_gen, DLL_data_2_real, DLL_particle, ref_particle, particle_source, bin_no=200, x_range=None, y_range=None):

    title = "DLL" + DLL_particle + "-" + ref_particle + "_" + particle_source + subset_text + "_ratios.pdf"
    
    if ref_particle == 'pi':
        ref_particle_text = r'$\pi ) $'
    elif ref_particle == 'mu':
        ref_particle_text = r'$\mu ) $'
    elif ref_particle == 'k':
        ref_particle_text = 'K)'
    else:
        ref_particle_text = ref_particle + ')'

    if DLL_particle == 'pi':
        DLL_particle_text = r'$\pi - $'
    elif DLL_particle == 'mu':
        DLL_particle_text = r'$\mu - $'
    elif DLL_particle == 'k':
        DLL_particle_text = r'$K-$'
    else:
        DLL_particle_text = DLL_particle + r'$-$'
    
    DLL_text = r'$\Delta LL ($' + DLL_particle_text + ref_particle_text

    labels = ['Kaon', 'Pion']
            
    fig1, ax1 = plt.subplots()
    ax1.cla()
    
    if y_range is not None:
        ax1.set_ylim(bottom=0, top=y_range)

    if x_range is not None:
        ax1.set_xlim(x_range)
    
    hist_kaon_1, _ = np.histogram(DLL_data_1_gen, bins=bin_no, range=x_range, density=True)
    hist_kaon_2, _ = np.histogram(DLL_data_1_real, bins=bin_no, range=x_range, density=True)
    ratio_kaon = np.divide(hist_kaon_2, hist_kaon_1)

    hist_pion_1, _ = np.histogram(DLL_data_2_gen, bins=bin_no, range=x_range, density=True)
    hist_pion_2, _ = np.histogram(DLL_data_2_real, bins=bin_no, range=x_range, density=True)
    ratio_pion = np.divide(hist_pion_2, hist_pion_1)
    
#    ax1.set_xlabel(DLL_text)
    ax1.set_ylabel("Ratio of densities")
    
    DLL = np.linspace(x_range[0],x_range[1],num=bin_no)
    ax1.plot(DLL, ratio_kaon, label=labels[0], color='r', alpha=0.75)
    ax1.plot(DLL, ratio_pion, label=labels[1], color='b', alpha=0.75)

#    ax1.legend()

    ax1.xaxis.tick_top()
    ax1.set_xticklabels([])

    aspect_ratio = (x_range[1] - x_range[0]) / 10
    ax1.set_aspect(aspect=aspect_ratio)

    fig1.savefig("./plots/" + set_text + "_" + title, format='pdf', dpi=2500)

    return 0


#Plot ratios and histograms of two sets of real/gen DLLs, all together
def plot_hist_and_ratios(DLL_data_1_gen, DLL_data_1_real, DLL_data_2_gen, DLL_data_2_real, DLL_particle, ref_particle, particle_source, legend_loc, bin_no_hist=200, bin_no_ratio=200, x_range=None, y_range_hist=None, y_range_ratio=None):

    title = "DLL" + DLL_particle + "-" + ref_particle + "_" + particle_source + subset_text + "_hists_and_ratios.pdf"
    
    if ref_particle == 'pi':
        ref_particle_text = r'$\pi ) $'
    elif ref_particle == 'mu':
        ref_particle_text = r'$\mu ) $'
    elif ref_particle == 'k':
        ref_particle_text = 'K)'
    else:
        ref_particle_text = ref_particle + ')'

    if DLL_particle == 'pi':
        DLL_particle_text = r'$\pi - $'
    elif DLL_particle == 'mu':
        DLL_particle_text = r'$\mu - $'
    elif DLL_particle == 'k':
        DLL_particle_text = r'$K-$'
    else:
        DLL_particle_text = DLL_particle + r'$-$'
    
    DLL_text = r'$\Delta LL ($' + DLL_particle_text + ref_particle_text

    ratios_labels = ['Kaon', 'Pion']
    hist_labels = ['Kaon generated', 'Kaon real', 'Pion generated', 'Pion real']
    
    hist_kaon_1, _ = np.histogram(DLL_data_1_gen, bins=bin_no_ratio, range=x_range, density=True)
    hist_kaon_2, _ = np.histogram(DLL_data_1_real, bins=bin_no_ratio, range=x_range, density=True)
    ratio_kaon = np.divide(hist_kaon_2, hist_kaon_1)

    hist_pion_1, _ = np.histogram(DLL_data_2_gen, bins=bin_no_ratio, range=x_range, density=True)
    hist_pion_2, _ = np.histogram(DLL_data_2_real, bins=bin_no_ratio, range=x_range, density=True)
    ratio_pion = np.divide(hist_pion_2, hist_pion_1)

    # Two subplots, the axes array is 1-d
    fig, axarr = plt.subplots(2, sharex=True, figsize=(5, 5))

    gs = gridspec.GridSpec(2, 1, height_ratios=[4, 1], hspace=0.4)
    
    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1])

    if y_range_hist is not None:
        ax1.set_ylim(bottom=0, top=y_range_hist)

    if x_range is not None:
        ax1.set_xlim(x_range)
        
    ax1.set_ylabel("Density of events", fontsize=10.5)
    ax1.set_xlabel(DLL_text, fontsize=10.5)

    if y_range_ratio is not None:
        ax2.set_ylim(bottom=0, top=y_range_ratio)

    if x_range is not None:
        ax2.set_xlim(x_range)

#    ax2.set_xlabel(DLL_text)
    ax2.set_ylabel("Ratio of densities", fontsize=10.5)
#    ax2.legend()
#    ax2.set_aspect(aspect=10)

    ax1.hist(DLL_data_1_gen, bins=bin_no_hist, range=x_range, density=True, histtype='step', color = 'r', linewidth=0.9, label=hist_labels[0])
    ax1.hist(DLL_data_1_real, bins=bin_no_hist, range=x_range, density=True, histtype='bar', alpha=0.35, color = 'r', linewidth=0.5, label=hist_labels[1])
    ax1.hist(DLL_data_2_gen, bins=bin_no_hist, range=x_range, density=True, histtype='step', color = 'b', linewidth=0.9, label=hist_labels[2])
    ax1.hist(DLL_data_2_real, bins=bin_no_hist, range=x_range, density=True, histtype='bar', alpha=0.35, color='b', linewidth=0.5, label=hist_labels[3])
    ax1.get_yaxis().set_label_coords(-0.11,0.5)

    ax1.legend(loc=legend_loc, fontsize=9)

    
    DLL = np.linspace(x_range[0],x_range[1],num=bin_no_ratio)
    ax2.plot(DLL, ratio_kaon, label=ratios_labels[0], color='r', alpha=0.75)
    ax2.plot(DLL, ratio_pion, label=ratios_labels[1], color='b', alpha=0.75)

    ax2.xaxis.tick_top()
    ax2.set_xticklabels([])
    ax2.get_yaxis().set_label_coords(-0.11,0.5)

    fig.savefig("./plots/" + set_text + "_" + title, format='pdf', dpi=2500)

    return 0


#Calculate overlap between two distributions
def histogram_intersection(DLL_data_1, DLL_data_2, bin_no=200, x_range=None):
   
   hist1, bins = np.histogram(DLL_data_1, bins=bin_no, range=x_range, density=True)
   hist2, _ = np.histogram(DLL_data_2, bins=bin_no, range=x_range, density=True)
   bins = np.diff(bins)
   sm = 0
   for i in range(len(bins)):
       sm += min(bins[i]*hist1[i], bins[i]*hist2[i])
   return sm


#Plot examples of DLL distributions
def plot_examples(generated_vars, var_name, bin_no=400, x_range = None, y_range = None):
    
    fig1, ax1 = plt.subplots()
    ax1.cla()
    
    title = 'GAN_generated_' + var_name + subset_text + '_trained_.eps'
    
    if y_range is not None:
        ax1.set_ylim(bottom = 0, top = y_range)
    
    if x_range is not None:
        ax1.set_xlim(x_range)
    
    ax1.set_xlabel(var_name)
    ax1.set_ylabel("Number of events")
    ax1.hist(generated_vars, bins=bin_no, range=x_range)
    
    fig1.savefig("./plots/" + set_text + "_" + title, format='eps', dpi=2500)

    return 0


#Use imported generator to simulate DLL vales
def generate_data(x_test, generator, alt_model, data_no):
    
    noise = np.random.normal(0, 1, size=[data_no, noise_dim])
    phys_data = x_test[:, np.add(input_phys_index,DLLs_dim)]    
#    phys_data = x_test[:, DLLs_dim:DLLs_dim + input_phys_dim]
    gen_input = np.concatenate((noise, phys_data), axis=1)

    t1 = time.time()
    
    #Generate fake data (DLLs only)
    if RNN:

        noise_RNN, _ = create_dataset(noise, seq_length)
        phys_data_RNN, _ = create_dataset(phys_data, seq_length)
        gen_input_RNN = np.concatenate((noise_RNN, phys_data_RNN), axis=2)

        #Generate fake data (DLLs only)
        generated_vars = generator.predict(gen_input_RNN, batch_size=apparent_batch_size)
        generated_vars = np.concatenate((generated_vars[0,:-1,:], generated_vars[:,-1,:]))

    elif alt_model:
        generated_vars = generator.predict([noise, phys_data])

    else:
        #Generate fake data (DLLs only)
        generated_vars = generator.predict(gen_input)

    t2 = time.time()
    
    print("Time to generate ", generated_vars.shape[0], " tracks: ", (t2-t1))
    
    return generated_vars


#Shift data (real and generated) back to proper distribution
def data_shift(x_test, div_num, shift, generated_vars, particle_source):
    
    for i in range(x_test.shape[1]):
        x_test[:,i] = np.multiply(x_test[:,i], div_num[i])
        x_test[:,i] = np.add(x_test[:,i], shift[i])

    for i in range(generated_vars.shape[1]):        
    
        generated_vars[:,i] = np.multiply(generated_vars[:,i], div_num[i])
        generated_vars[:,i] = np.add(generated_vars[:,i], shift[i])
            
        if i<DLLs_dim:
            plot_examples(generated_vars[:,i], particle_source + '_DLL'+ DLLs[i])
        else:
            plot_examples(generated_vars[:,i], physical_vars[i-DLLs_dim])

    return generated_vars


# =============================================================================
# Import data
# =============================================================================

print("Importing data...")
#Get the training and testing data
x_test_1, shift_1, div_num_1 = get_x_data(DLLs, ref_particle, physical_vars, particle_source_1, examples, unused_mask_k, unused_mask_loc_k)
x_test_2, shift_2, div_num_2 = get_x_data(DLLs, ref_particle, physical_vars, particle_source_2, examples, unused_mask_p, unused_mask_loc_p)
print("Data imported")

######################################################################################################################################################################################

# =============================================================================
# Generate data
# =============================================================================

print("Generating data...")

#KAON data
#data_batch_1 = x_test_1[np.random.randint(0, x_test_1.shape[0], size=examples)]

if gen_av:
    if concat:
        generated_vars_k_1 = generate_data(x_test_1[:examples//2], generator_k, alt_model_k, examples//2)
        generated_vars_k_2 = generate_data(x_test_1[examples//2:], generator_k_2, alt_model_k, examples//2)
    else:
        generated_vars_k_1 = generate_data(x_test_1, generator_k, alt_model_k, examples)
        generated_vars_k_2 = generate_data(x_test_1, generator_k_2, alt_model_k, examples)
else:
    generated_vars_k = generate_data(x_test_1, generator_k, alt_model_k, examples)

######################################################################################################################################################################################

#PION data
#data_batch_2 = x_test_2[np.random.randint(0, x_test_2.shape[0], size=examples)]

if gen_av:
    if concat:
        generated_vars_p_1 = generate_data(x_test_2[:examples//2], generator_p, alt_model_p, examples//2)
        generated_vars_p_2 = generate_data(x_test_2[examples//2:], generator_p_2, alt_model_p, examples//2)
    else:
        generated_vars_p_1 = generate_data(x_test_2, generator_p, alt_model_p, examples)
        generated_vars_p_2 = generate_data(x_test_2, generator_p_2, alt_model_p, examples)
else:
    generated_vars_p = generate_data(x_test_2, generator_p, alt_model_p, examples)

print("Data generated")

######################################################################################################################################################################################

# =============================================================================
# Unnorm andplot initial data
# =============================================================================

print("Plotting data...")

#Shift data to correct dist and plot examples of graphs 
if gen_av:
    
    dummy_arr = np.zeros((1,1))
    
    generated_vars_k_1 = data_shift(x_test_1, div_num_1, shift_1, generated_vars_k_1, particle_source_1)
    generated_vars_k_2 = data_shift(dummy_arr, div_num_1, shift_1, generated_vars_k_2, particle_source_1)    
    generated_vars_p_1 = data_shift(x_test_2, div_num_2, shift_2, generated_vars_p_1, particle_source_2)
    generated_vars_p_2 = data_shift(dummy_arr, div_num_2, shift_2, generated_vars_p_2, particle_source_2)
    
    if concat:
        generated_vars_k = np.concatenate((generated_vars_k_1, generated_vars_k_2))
        generated_vars_p = np.concatenate((generated_vars_p_1, generated_vars_p_2))
    else:    
    
        generated_vars_k = np.add(generated_vars_k_1, generated_vars_k_2)
        generated_vars_k = np.divide(generated_vars_k, 2)
        
        generated_vars_p = np.add(generated_vars_p_1, generated_vars_p_2)
        generated_vars_p = np.divide(generated_vars_p, 2)
    
else:
    generated_vars_k = data_shift(x_test_1, div_num_1, shift_1, generated_vars_k, particle_source_1)
    generated_vars_p = data_shift(x_test_2, div_num_2, shift_2, generated_vars_p, particle_source_2)

# =============================================================================
# All other plotting functions e.g. scatter graphs, efficiency plots etc
# =============================================================================

max_var_index = 10000
#Scatter plot between DLLd and DLLk for KAON and PION data

#DLLk_k against DLLp_k (gen and real separate)
col_scatt(generated_vars_k[:,2], generated_vars_k[:,3], particle_source_1, DLLs[2], ref_particle, particle_source_1,  DLLs[3], ref_particle, max_var_index, 'gen', x_range=[-100,100], y_range=[-80,80], zero_lines=1, save_index=10000, size=0.5)
col_scatt(x_test_1[:,2], x_test_1[:,3], particle_source_1, DLLs[2], ref_particle, particle_source_1,  DLLs[3], ref_particle, max_var_index, 'real', x_range=[-100,100], y_range=[-80,80], zero_lines=1, save_index=10000, size=0.5)

#DLLk_p against DLLp_p (gen and real separate)
col_scatt(generated_vars_p[:,2], generated_vars_p[:,3], particle_source_2, DLLs[2], ref_particle, particle_source_2,  DLLs[3], ref_particle, max_var_index, 'gen', x_range=[-100,100], y_range=[-80,80], zero_lines=1, save_index=10000, size=0.5)
col_scatt(x_test_2[:,2], x_test_2[:,3], particle_source_2, DLLs[2], ref_particle, particle_source_2,  DLLs[3], ref_particle, max_var_index, 'real', x_range=[-100,100], y_range=[-80,80], zero_lines=1, save_index=10000, size=0.5)

#Scatter plot between DLLk for KAON and PION data (generated and original compared)
col_scatt(generated_vars_k[:,2], x_test_1[:,2], particle_source_1, DLLs[2], ref_particle, particle_source_1,  DLLs[2], ref_particle, max_var_index, 'real_gen', x_range=[-100,100], y_range=[-80,80], zero_lines=1, save_index=10000, size=0.5)
col_scatt(generated_vars_p[:,2], x_test_2[:,2], particle_source_2, DLLs[2], ref_particle, particle_source_2,  DLLs[2], ref_particle, max_var_index, 'real_gen', x_range=[-100,100], y_range=[-80,80], zero_lines=1, save_index=10000, size=0.5)

#Scatter plot between TrackP for KAON data if P generated (generated and original)
#if generate_P:
#    col_scatt(generated_vars_k[:,6], x_test_1[:,6], "TrackP", "TrackP", max_var_index, x_range=[0,100000], y_range=[0,100000], zero_lines=1, save_index=10000, size=0.5)


######################################################################################################################################################################################

for i in range(phys_dim):
    if physical_vars[i] == 'TrackP':
        p_index = DLLs_dim + i

    if physical_vars[i] == 'TrackPt':
        pt_index = DLLs_dim + i
        
    if physical_vars[i] == 'NumLongTracks':
        num_tracks_index = DLLs_dim + i

    if physical_vars[i] == 'NumPVs':
        PV_index = DLLs_dim + i

    if physical_vars[i] == 'RICH1EntryDist0':
        R1_En_0_index = DLLs_dim + i

    if physical_vars[i] == 'RICH1ExitDist0':
        R1_Ex_0_index = DLLs_dim + i

    if physical_vars[i] == 'RICH2EntryDist0':
        R2_En_0_index = DLLs_dim + i

    if physical_vars[i] == 'RICH2ExitDist0':
        R2_Ex_0_index = DLLs_dim + i

    if physical_vars[i] == 'RICH1EntryDist1':
        R1_En_1_index = DLLs_dim + i

    if physical_vars[i] == 'RICH1ExitDist1':
        R1_Ex_1_index = DLLs_dim + i

    if physical_vars[i] == 'RICH2EntryDist1':
        R2_En_1_index = DLLs_dim + i

    if physical_vars[i] == 'RICH2ExitDist1':
        R2_Ex_1_index = DLLs_dim + i

    if physical_vars[i] == 'RICH1EntryDist2':
        R1_En_2_index = DLLs_dim + i

    if physical_vars[i] == 'RICH1ExitDist2':
        R1_Ex_2_index = DLLs_dim + i

    if physical_vars[i] == 'RICH2EntryDist2':
        R2_En_2_index = DLLs_dim + i

    if physical_vars[i] == 'RICH2ExitDist2':
        R2_Ex_2_index = DLLs_dim + i

    if physical_vars[i] == 'RICH1ConeNum':
        R1_cone_index = DLLs_dim + i

    if physical_vars[i] == 'RICH2ConeNum':
        R2_cone_index = DLLs_dim + i



p_bins_no = 100 #Number of momentum bins
p_max = 100.0 #Maximum track momentum
uni_bins = 0 #Uniform bin sizes
exp_bins = 0 #Exponentially increasing bin sizes (if neither uni or exp, linear increas)
exponent = 2 #Exponent for logspace. Doesn't change anything currently as overspecified?

#Calculate efficiency using generated DLLs and generated momementa
if generate_P:
    eff_mom_calc(p_bins_no, p_max, uni_bins, exp_bins, exponent, generated_vars_k[:,2], generated_vars_k[:,6], generated_vars_p[:,2], generated_vars_p[:,6], 'k', ref_particle, particle_source_1, particle_source_2, 'gen_P')

#Calculate efficiency using generated DLLs and real momementa... Doesn't work when with models which generate P/Pt.
eff_mom_calc(p_bins_no, p_max, uni_bins, exp_bins, exponent, generated_vars_k[:,2], x_test_1[:, p_index], generated_vars_p[:,2], x_test_2[:, p_index], 'k', ref_particle, particle_source_1, particle_source_2, 'real_P')
eff_mom_calc(p_bins_no, p_max, uni_bins, exp_bins, exponent, x_test_1[:,2], x_test_1[:, p_index], x_test_2[:,2], x_test_2[:, p_index], 'k', ref_particle, particle_source_1, particle_source_2, 'real_DLL-P')

######################################################################################################################################################################################

misid_bin_no = 4
DLL_lim = 15
DLL_no = 21

######################################################################################################################################################################################

#'NumLongTracks'
phys_var_range = [0,400]
x_range = [0.1, 1]

#Generated DLLk, real no of tracks:
id_misid_eff(generated_vars_k[:,2], generated_vars_p[:,2], x_test_1[:, num_tracks_index], x_test_2[:, num_tracks_index], misid_bin_no, DLL_lim, DLL_no, phys_var_range, 'NumLongTracks', x_range, 'gen_DLL', True)

#Real DLLk, real no of tracks:
id_misid_eff(x_test_1[:,2], x_test_2[:,2], x_test_1[:, num_tracks_index], x_test_2[:, num_tracks_index], misid_bin_no, DLL_lim, DLL_no, phys_var_range, 'NumLongTracks', x_range, 'real_DLL', True)

######################################################################################################################################################################################

#'NumPVs'

phys_var_range = [1,5]
x_range = [0.5, 1]

#Generated DLLk, real no of tracks:
id_misid_eff(generated_vars_k[:,2], generated_vars_p[:,2], x_test_1[:, PV_index], x_test_2[:, PV_index], misid_bin_no, DLL_lim, DLL_no, phys_var_range, 'NumPVs', x_range, 'gen_DLL', False)

#Real DLLk, real no of tracks:
id_misid_eff(x_test_1[:,2], x_test_2[:,2], x_test_1[:, PV_index], x_test_2[:, PV_index], misid_bin_no, DLL_lim, DLL_no, phys_var_range, 'NumPVs', x_range, 'real_DLL', False)

######################################################################################################################################################################################

#'RICH1EntryDist0'
#Increase in KID as dist increases
misid_bin_no = 4
phys_var_range = [0,40]
x_range = [0.2, 1]
id_misid_eff(generated_vars_k[:,2], generated_vars_p[:,2], x_test_1[:, R1_En_0_index], x_test_2[:, R1_En_0_index], misid_bin_no, DLL_lim, DLL_no, phys_var_range, 'RICH1EntryDist0', x_range, 'gen_DLL', True)
id_misid_eff(x_test_1[:,2], x_test_2[:,2], x_test_1[:, R1_En_0_index], x_test_2[:, R1_En_0_index], misid_bin_no, DLL_lim, DLL_no, phys_var_range, 'RICH1EntryDist0', x_range, 'real_DLL', True)

#Increases with distance
#'RICH1ExitDist0'
misid_bin_no = 4
phys_var_range = [0,100]
x_range = [0.0, 1]
id_misid_eff(generated_vars_k[:,2], generated_vars_p[:,2], x_test_1[:, R1_Ex_0_index], x_test_2[:, R1_Ex_0_index], misid_bin_no, DLL_lim, DLL_no, phys_var_range, 'RICH1ExitDist0', x_range, 'gen_DLL', True)
id_misid_eff(x_test_1[:,2], x_test_2[:,2], x_test_1[:, R1_Ex_0_index], x_test_2[:, R1_Ex_0_index], misid_bin_no, DLL_lim, DLL_no, phys_var_range, 'RICH1ExitDist0', x_range, 'real_DLL', True)

#'RICH2EntryDist0'
#Increase as distance increases
misid_bin_no = 4
phys_var_range = [0,400]
x_range = [0.2, 1]
id_misid_eff(generated_vars_k[:,2], generated_vars_p[:,2], x_test_1[:, R2_En_0_index], x_test_2[:, R2_En_0_index], misid_bin_no, DLL_lim, DLL_no, phys_var_range, 'RICH2EntryDist0', x_range, 'gen_DLL', True)
id_misid_eff(x_test_1[:,2], x_test_2[:,2], x_test_1[:, R2_En_0_index], x_test_2[:, R2_En_0_index], misid_bin_no, DLL_lim, DLL_no, phys_var_range, 'RICH2EntryDist0', x_range, 'real_DLL', True)


#'RICH2ExitDist0'
#Increase as distance increases
misid_bin_no = 4
phys_var_range = [0,800]
x_range = [0.2, 1]
id_misid_eff(generated_vars_k[:,2], generated_vars_p[:,2], x_test_1[:, R2_Ex_0_index], x_test_2[:, R2_Ex_0_index],misid_bin_no, DLL_lim, DLL_no, phys_var_range, 'RICH2ExitDist0', x_range, 'gen_DLL', True)
id_misid_eff(x_test_1[:,2], x_test_2[:,2], x_test_1[:, R2_Ex_0_index], x_test_2[:, R2_Ex_0_index],misid_bin_no, DLL_lim, DLL_no, phys_var_range, 'RICH2ExitDist0', x_range, 'real_DLL', True)

#'RICH1EntryDist1'
#Increases with dist
misid_bin_no = 4
phys_var_range = [0,40]
x_range = [0.1, 1]
id_misid_eff(generated_vars_k[:,2], generated_vars_p[:,2], x_test_1[:, R1_En_1_index], x_test_2[:, R1_En_1_index], misid_bin_no, DLL_lim, DLL_no, phys_var_range, 'RICH1EntryDist1', x_range, 'gen_DLL', True)
id_misid_eff(x_test_1[:,2], x_test_2[:,2], x_test_1[:, R1_En_1_index], x_test_2[:, R1_En_1_index], misid_bin_no, DLL_lim, DLL_no, phys_var_range, 'RICH1EntryDist1', x_range, 'real_DLL', True)

#'RICH1ExitDist1'
#Increase with distance
misid_bin_no = 4
phys_var_range = [0,100]
x_range = [0.1, 1]
id_misid_eff(generated_vars_k[:,2], generated_vars_p[:,2], x_test_1[:, R1_Ex_1_index], x_test_2[:, R1_Ex_1_index], misid_bin_no, DLL_lim, DLL_no, phys_var_range, 'RICH1ExitDist1', x_range, 'gen_DLL', True)
id_misid_eff(x_test_1[:,2], x_test_2[:,2], x_test_1[:, R1_Ex_1_index], x_test_2[:, R1_Ex_1_index], misid_bin_no, DLL_lim, DLL_no, phys_var_range, 'RICH1ExitDist1', x_range, 'real_DLL', True)

#'RICH2EntryDist1'
#Increase as distance increases
misid_bin_no = 4
phys_var_range = [0,400]
x_range = [0.1, 1]
id_misid_eff(generated_vars_k[:,2], generated_vars_p[:,2], x_test_1[:, R2_En_1_index], x_test_2[:, R2_En_1_index], misid_bin_no, DLL_lim, DLL_no, phys_var_range, 'RICH2EntryDist1', x_range, 'gen_DLL', True)
id_misid_eff(x_test_1[:,2], x_test_2[:,2], x_test_1[:, R2_En_1_index], x_test_2[:, R2_En_1_index], misid_bin_no, DLL_lim, DLL_no, phys_var_range, 'RICH2EntryDist1', x_range, 'real_DLL', True)


#'RICH2ExitDist1'
#Increase as distance increases
misid_bin_no = 4
phys_var_range = [0,800]
x_range = [0.1, 1]
id_misid_eff(generated_vars_k[:,2], generated_vars_p[:,2], x_test_1[:, R2_Ex_1_index], x_test_2[:, R2_Ex_1_index], misid_bin_no, DLL_lim, DLL_no, phys_var_range, 'RICH2ExitDist1', x_range, 'gen_DLL', True)
id_misid_eff(x_test_1[:,2], x_test_2[:,2], x_test_1[:, R2_Ex_1_index], x_test_2[:, R2_Ex_1_index], misid_bin_no, DLL_lim, DLL_no, phys_var_range, 'RICH2ExitDist1', x_range, 'real_DLL', True)



#'RICH1EntryDist2'
#Incrases, last two together
misid_bin_no = 4
phys_var_range = [0,100]
x_range = [0.1, 1]
id_misid_eff(generated_vars_k[:,2], generated_vars_p[:,2], x_test_1[:, R1_En_2_index], x_test_2[:, R1_En_2_index], misid_bin_no, DLL_lim, DLL_no, phys_var_range, 'RICH1EntryDist2', x_range, 'gen_DLL', True)
id_misid_eff(x_test_1[:,2], x_test_2[:,2], x_test_1[:, R1_En_2_index], x_test_2[:, R1_En_2_index], misid_bin_no, DLL_lim, DLL_no, phys_var_range, 'RICH1EntryDist2', x_range, 'real_DLL', True)

#'RICH1ExitDist2'
#Increases with distance
misid_bin_no = 4
phys_var_range = [0,100]
x_range = [0.1, 1]
id_misid_eff(generated_vars_k[:,2], generated_vars_p[:,2], x_test_1[:, R1_Ex_2_index], x_test_2[:, R1_Ex_2_index], misid_bin_no, DLL_lim, DLL_no, phys_var_range, 'RICH1ExitDist2', x_range, 'gen_DLL', True)
id_misid_eff(x_test_1[:,2], x_test_2[:,2], x_test_1[:, R1_Ex_2_index], x_test_2[:, R1_Ex_2_index], misid_bin_no, DLL_lim, DLL_no, phys_var_range, 'RICH1ExitDist2', x_range, 'real_DLL', True)


#'RICH2EntryDist2'
#Increase as distance increases
misid_bin_no = 4
phys_var_range = [0,800]
x_range = [0.1, 1]
id_misid_eff(generated_vars_k[:,2], generated_vars_p[:,2], x_test_1[:, R2_En_2_index], x_test_2[:, R2_En_2_index], misid_bin_no, DLL_lim, DLL_no, phys_var_range, 'RICH2EntryDist2', x_range, 'gen_DLL', True)
id_misid_eff(x_test_1[:,2], x_test_2[:,2], x_test_1[:, R2_En_2_index], x_test_2[:, R2_En_2_index], misid_bin_no, DLL_lim, DLL_no, phys_var_range, 'RICH2EntryDist2', x_range, 'real_DLL', True)


#'RICH2ExitDist2'
#Increase as distance increases
misid_bin_no = 4
phys_var_range = [0,800]
x_range = [0.1, 1]
id_misid_eff(generated_vars_k[:,2], generated_vars_p[:,2], x_test_1[:, R2_Ex_2_index], x_test_2[:, R2_Ex_2_index], misid_bin_no, DLL_lim, DLL_no, phys_var_range, 'RICH2ExitDist2', x_range, 'gen_DLL', True)
id_misid_eff(x_test_1[:,2], x_test_2[:,2], x_test_1[:, R2_Ex_2_index], x_test_2[:, R2_Ex_2_index], misid_bin_no, DLL_lim, DLL_no, phys_var_range, 'RICH2ExitDist2', x_range, 'real_DLL', True)

######################################################################################################################################################################################

#Increase in KID as num decreases
#'RICH1ConeNum'
misid_bin_no = 4
phys_var_range = [0,24]
x_range = [0.0, 1]
id_misid_eff(generated_vars_k[:,2], generated_vars_p[:,2], x_test_1[:, R1_cone_index], x_test_2[:, R1_cone_index], misid_bin_no, DLL_lim, DLL_no, phys_var_range, 'RICH1ConeNum', x_range, 'gen_DLL', True)
id_misid_eff(x_test_1[:,2], x_test_2[:,2], x_test_1[:, R1_cone_index], x_test_2[:, R1_cone_index], misid_bin_no, DLL_lim, DLL_no, phys_var_range, 'RICH1ConeNum', x_range, 'real_DLL', True)


#Increase in KID as num decreases
#'RICH2ConeNum'
misid_bin_no = 4
phys_var_range = [0,4]
x_range = [0.1, 1]
id_misid_eff(generated_vars_k[:,2], generated_vars_p[:,2], x_test_1[:, R2_cone_index], x_test_2[:, R2_cone_index], misid_bin_no, DLL_lim, DLL_no, phys_var_range, 'RICH2ConeNum', x_range, 'gen_DLL', True)
id_misid_eff(x_test_1[:,2], x_test_2[:,2], x_test_1[:, R2_cone_index], x_test_2[:, R2_cone_index], misid_bin_no, DLL_lim, DLL_no, phys_var_range, 'RICH2ConeNum', x_range, 'real_DLL', True)

# =============================================================================
# Plot final indvidual histograms for generated DLLs
# =============================================================================

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
#plot_gen_hist(generated_vars_p[:,6], physical_vars[p_index], particle_source_2, 750, [0,70000], 12000)
#plot_gen_hist(generated_vars_p[:,7], physical_vars[pt_index], particle_source_2, 750, [0,3500], 6000)
#plot_gen_hist(x_test_2[:,6], physical_vars[p_index], particle_source_2, 750, [0,70000], 7000)
#plot_gen_hist(x_test_2[:,7], physical_vars[pt_index], particle_source_2, 750, [0,3500], 5000)

######################################################################################################################################################################################

#plot_DLL_hist(x_test_1[:,2], DLLs[2], ref_particle, particle_source_1, 750, [-40, 80], 0.07)
#plot_DLL_hist(generated_vars_k[:,2], DLLs[2], ref_particle, particle_source_1, 750, [-40, 80], 0.07)
#
#plot_two_DLL_hist(generated_vars_k[:,2], x_test_1[:,2], DLLs[2], ref_particle, particle_source_1, 'upper right', 250, [-30, 80], 0.07)
#plot_hist_ratio(generated_vars_k[:,2], x_test_1[:,2], DLLs[2], ref_particle, particle_source_1, 100, [-30, 80], 2)
#hist_overlap_1 = histogram_intersection(generated_vars_k[:,2], x_test_1[:,2], 750, [-40, 80])
#
#plot_DLL_hist(x_test_2[:,2], DLLs[2], ref_particle, particle_source_2, 750, [-60, 20], 0.09)
#plot_DLL_hist(generated_vars_p[:,2], DLLs[2], ref_particle, particle_source_2, 750, [-60, 20], 0.09)
#
#plot_two_DLL_hist(generated_vars_p[:,2], x_test_2[:,2], DLLs[2], ref_particle, particle_source_2, 'upper left', 750, [-60, 20], 0.09)
#plot_hist_ratio(generated_vars_p[:,2], x_test_2[:,2], DLLs[2], ref_particle, particle_source_2, 100, [-60, 20], 2)
#hist_overlap_2 = histogram_intersection(generated_vars_p[:,2], x_test_2[:,2], 750, [-60, 20])
#
#print("Overlaps:", hist_overlap_1, hist_overlap_2)

# =============================================================================
# Histograms comparing real/generated data, and ratios
# =============================================================================

# =============================================================================
# e:
# =============================================================================

#plot_four_DLL_hist(generated_vars_k[:,0], x_test_1[:,0], generated_vars_p[:,0], x_test_2[:,0], DLLs[0], ref_particle, particle_source_2, 'upper left', 750, [-80, 20], 0.16)
#plot_hist_ratios(generated_vars_k[:,0], x_test_1[:,0], generated_vars_p[:,0], x_test_2[:,0], DLLs[0], ref_particle, particle_source_1, 100, [-80, 20], 2)
hist_overlap_1 = histogram_intersection(generated_vars_k[:,0], x_test_1[:,0], 750, [-80, 20])
hist_overlap_2 = histogram_intersection(generated_vars_p[:,0], x_test_2[:,0], 750, [-80, 20])

print("Overlaps for DLL(" + DLLs[0] + "-" + ref_particle + "):")
print("Kaon:", hist_overlap_1)
print("Pion:", hist_overlap_2)

########################################################

# =============================================================================
# mu:
# =============================================================================

#plot_four_DLL_hist(generated_vars_k[:,1], x_test_1[:,1], generated_vars_p[:,1], x_test_2[:,1], DLLs[1], ref_particle, particle_source_2, 'upper left', 750, [-50, 20], 0.3)
#plot_hist_ratios(generated_vars_k[:,1], x_test_1[:,1], generated_vars_p[:,1], x_test_2[:,1], DLLs[1], ref_particle, particle_source_1, 100, [-50, 20], 2)
hist_overlap_3 = histogram_intersection(generated_vars_k[:,1], x_test_1[:,1], 750, [-50, 20])
hist_overlap_4 = histogram_intersection(generated_vars_p[:,1], x_test_2[:,1], 750, [-50, 20])

print("Overlaps for DLL(" + DLLs[1] + "-" + ref_particle + "):")
print("Kaon:", hist_overlap_3)
print("Pion:", hist_overlap_4)


# =============================================================================
# k:
# =============================================================================

#plot_four_DLL_hist(generated_vars_k[:,2], x_test_1[:,2], generated_vars_p[:,2], x_test_2[:,2], DLLs[2], ref_particle, particle_source_2, 'upper left', 750, [-60, 80], 0.1)
#plot_hist_ratios(generated_vars_k[:,2], x_test_1[:,2], generated_vars_p[:,2], x_test_2[:,2], DLLs[2], ref_particle, particle_source_1, 100, [-60, 80], 2)
hist_overlap_5 = histogram_intersection(generated_vars_k[:,2], x_test_1[:,2], 750, [-60, 80])
hist_overlap_6 = histogram_intersection(generated_vars_p[:,2], x_test_2[:,2], 750, [-60, 80])

print("Overlaps for DLL(" + DLLs[2] + "-" + ref_particle + "):")
print("Kaon:", hist_overlap_5)
print("Pion:", hist_overlap_6)

########################################################

# =============================================================================
# p:
# =============================================================================

#plot_four_DLL_hist(generated_vars_k[:,3], x_test_1[:,3], generated_vars_p[:,3], x_test_2[:,3], DLLs[3], ref_particle, particle_source_2, 'upper left', 750, [-60, 60], 0.1)
#plot_hist_ratios(generated_vars_k[:,3], x_test_1[:,3], generated_vars_p[:,3], x_test_2[:,3], DLLs[3], ref_particle, particle_source_1, 100, [-60, 60], 2)
hist_overlap_7 = histogram_intersection(generated_vars_k[:,3], x_test_1[:,3], 750, [-60, 60])
hist_overlap_8 = histogram_intersection(generated_vars_p[:,3], x_test_2[:,3], 750, [-60, 60])

print("Overlaps for DLL(" + DLLs[3] + "-" + ref_particle + "):")
print("Kaon:", hist_overlap_7)
print("Pion:", hist_overlap_8)

########################################################

# =============================================================================
# d:
# =============================================================================

#plot_four_DLL_hist(generated_vars_k[:,4], x_test_1[:,4], generated_vars_p[:,4], x_test_2[:,4], DLLs[4], ref_particle, particle_source_2, 'upper left', 750, [-60, 60], 0.1)
#plot_hist_ratios(generated_vars_k[:,4], x_test_1[:,4], generated_vars_p[:,4], x_test_2[:,4], DLLs[4], ref_particle, particle_source_1, 100, [-60, 60], 2)
hist_overlap_9 = histogram_intersection(generated_vars_k[:,4], x_test_1[:,4], 750, [-60, 60])
hist_overlap_10 = histogram_intersection(generated_vars_p[:,4], x_test_2[:,4], 750, [-60, 60])

print("Overlaps for DLL(" + DLLs[4] + "-" + ref_particle + "):")
print("Kaon:", hist_overlap_9)
print("Pion:", hist_overlap_10)

# =============================================================================
# bt:
# =============================================================================

#plot_four_DLL_hist(generated_vars_k[:,5], x_test_1[:,5], generated_vars_p[:,5], x_test_2[:,5], DLLs[5], ref_particle, particle_source_2, 'upper left', 750, [-60, 60], 0.1)
#plot_hist_ratios(generated_vars_k[:,5], x_test_1[:,5], generated_vars_p[:,5], x_test_2[:,5], DLLs[5], ref_particle, particle_source_1, 100, [-60, 60], 2)
hist_overlap_11 = histogram_intersection(generated_vars_k[:,5], x_test_1[:,5], 750, [-60, 60])
hist_overlap_12 = histogram_intersection(generated_vars_p[:,5], x_test_2[:,5], 750, [-60, 60])

print("Overlaps for DLL(" + DLLs[5] + "-" + ref_particle + "):")
print("Kaon:", hist_overlap_11)
print("Pion:", hist_overlap_12)

########################################################

# =============================================================================
# Combined ratio and histograms:
# =============================================================================

plot_hist_and_ratios(generated_vars_k[:,0], x_test_1[:,0], generated_vars_p[:,0], x_test_2[:,0], DLLs[0], ref_particle, particle_source_2, 'upper left', 750, 100, [-80, 20], 0.16, 2)
plot_hist_and_ratios(generated_vars_k[:,1], x_test_1[:,1], generated_vars_p[:,1], x_test_2[:,1], DLLs[1], ref_particle, particle_source_2, 'upper left', 750, 100, [-50, 20], 0.3, 2)
plot_hist_and_ratios(generated_vars_k[:,2], x_test_1[:,2], generated_vars_p[:,2], x_test_2[:,2], DLLs[2], ref_particle, particle_source_2, 'upper left', 750, 100, [-60, 80], 0.1, 2)
plot_hist_and_ratios(generated_vars_k[:,3], x_test_1[:,3], generated_vars_p[:,3], x_test_2[:,3], DLLs[3], ref_particle, particle_source_2, 'upper left', 750, 100, [-60, 60], 0.1, 2)
plot_hist_and_ratios(generated_vars_k[:,4], x_test_1[:,4], generated_vars_p[:,4], x_test_2[:,4], DLLs[4], ref_particle, particle_source_2, 'upper left', 750, 100, [-60, 60], 0.1, 2)
plot_hist_and_ratios(generated_vars_k[:,5], x_test_1[:,5], generated_vars_p[:,5], x_test_2[:,5], DLLs[5], ref_particle, particle_source_2, 'upper left', 750, 100, [-60, 60], 0.1, 2)

########################################################

print("Data plotted")

#Save overlaps as text
hist_overlaps = [hist_overlap_1, hist_overlap_2, hist_overlap_3, hist_overlap_4, hist_overlap_5, hist_overlap_6, hist_overlap_7, hist_overlap_8, hist_overlap_9, hist_overlap_10, hist_overlap_11, hist_overlap_12] 
with open('overlaps_' + set_text + '.txt', 'w') as f:
    print(hist_overlaps, file=f)


# =============================================================================
# Correlation coefficients
# =============================================================================

print(np.corrcoef(x_test_1[:,2], x_test_1[:,3])[1,0])
#print(np.corrcoef(x_test_1[1000000:,2], x_test_1[1000000:,3])[1,0])
#print(np.corrcoef(x_test_1[:1000000,2], x_test_1[:1000000,3])[1,0])

print(np.corrcoef(generated_vars_k[:,2], generated_vars_k[:,3])[1,0])
#print(np.corrcoef(generated_vars_k_1[:,2], generated_vars_k_1[:,3])[1,0])
#print(np.corrcoef(generated_vars_k_2[:,2], generated_vars_k_2[:,3])[1,0])

print(np.corrcoef(x_test_2[:,2], x_test_2[:,3])[1,0])
#print(np.corrcoef(x_test_2[1000000:,2], x_test_2[1000000:,3])[1,0])
#print(np.corrcoef(x_test_2[:1000000,2], x_test_2[:1000000,3])[1,0])

print(np.corrcoef(generated_vars_p[:,2], generated_vars_p[:,3])[1,0])
#print(np.corrcoef(generated_vars_p_1[:,2], generated_vars_p_1[:,3])[1,0])
#print(np.corrcoef(generated_vars_p_2[:,2], generated_vars_p_2[:,3])[1,0])


print(np.corrcoef(generated_vars_k[:,2], x_test_1[:,2])[1,0])
#print(np.corrcoef(generated_vars_k_1[:,2], x_test_1[:1000000,2])[1,0])
#print(np.corrcoef(generated_vars_k_2[:,2], x_test_1[1000000:,2])[1,0])

print(np.corrcoef(generated_vars_p[:,2], x_test_2[:,2])[1,0])
#print(np.corrcoef(generated_vars_p_1[:,2], x_test_2[:1000000,2])[1,0])
#print(np.corrcoef(generated_vars_p_2[:,2], x_test_2[1000000:,2])[1,0])


#Measure total run time for script and save
t_final = time.time()
runtime = t_final - t_init
print("Total run time = ", runtime)

with open('runtime_' + set_text + '.txt', 'w') as f:
    print(runtime, file=f)
    
