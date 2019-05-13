#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#Script to perform analysis of KAON/PION data
#Plots DLL distributions, correlations, ID/mis-ID efficiencies etc.

#Assumes data files for kaon and pion tracks (mod refers to additonal variables added):
# '../../data/mod-PID-train-data-KAONS.hdf'
# '../../data/mod-PID-train-data-PIONS.hdf'

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from scipy.stats import gaussian_kde
import math
import time
#from sklearn.preprocessing import QuantileTransformer

#Time total run
t_init = time.time()

#help(data_kaon) 
#help(data_kaon.values)

plt.rcParams['agg.path.chunksize'] = 10000 #Needed for plotting lots of data?

#Import data from kaons and pions
datafile_kaon = '../data/mod-PID-train-data-KAONS.hdf' 
data_kaon = pd.read_hdf(datafile_kaon, 'KAONS') 
print(data_kaon.columns)

datafile_pion = '../data/mod-PID-train-data-PIONS.hdf' 
data_pion = pd.read_hdf(datafile_pion, 'PIONS') 
print(data_pion.columns)

subset=False
sub_var = 'RICH2EntryDist0'
sub_min = None
sub_max = 30

if subset:
    subset_text = '_' + sub_var + '_' + str(sub_min) + '-' + str(sub_max)
else:
    subset_text = ''

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

print("Data imported")

###############################################################################

#Basic data manipulation e.g. selecting columns of data and changing DLLs

#Get column from kaon or pion data
#Input: Variable to extract, particle source (KAON or PION)
#Returns: pandas column with variable of interest
def get_data(var_type, particle_source):
    
    if(particle_source == 'KAON'):
        data_loc = data_kaon
    elif(particle_source == 'PION'):
        data_loc = data_pion
    else:
        print("Please select either kaon or pion as particle source")

    data = data_loc.loc[:, var_type]

    return data


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


# =============================================================================
# General plotting functions
# =============================================================================

#Make plots of one variable only to see general distribution, up to max_index
def one_var_plots(max_index, var1, var1_text):

    fig1, ax1 = plt.subplots()
    ax1.cla()
    ax1.plot(var1[0:max_index])
    ax1.set_ylabel(var1_text)
    fig1.savefig(var1_text + ".eps", format='eps', dpi=1000)

#Make plots of two variables up to max_index_current
#Don't save if max_index_current less than max_index_ever
def two_var_plots(max_index_ever, max_index_current, var1, var2, var1_text, var2_text, size):

    fig1, ax1 = plt.subplots()
    ax1.cla()
    ax1.scatter(var1[0:max_index_current], var2[0:max_index_current], s=size)
    ax1.set_xlabel(var1_text)
    ax1.set_ylabel(var2_text)
    if(max_index_current >= max_index_ever):
        fig1.savefig(var1_text + "_" + var2_text + ".eps", format='eps', dpi=1000)


#Make scatter plot of correlations between two variables (e.g. DLLs)
def ord_scatt(var1, var2, var1_text, var2_text, max_var_index, x_range=None, y_range=None, zero_lines=0, save_index=0, size=1):
    
    x = var1[0:max_var_index]
    y = var2[0:max_var_index]
        
    #Ordinary scatter plot
    fig1, ax1 = plt.subplots()
    ax1.cla()
    ax1.scatter(x, y, s=size)
    ax1.set_xlabel(var1_text)
    ax1.set_ylabel(var2_text)
    ax1.set_xlim(x_range)
    ax1.set_ylim(y_range)
    
    if(zero_lines):
        ax1.axhline(lw=1.0, color='k',ls='--')
        ax1.axvline(lw=1.0, color='k',ls='--')
    
    if(max_var_index >= save_index):
        fig1.savefig(var1_text + "_" + var2_text + ".eps", format='eps', dpi=1000)
    

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
    ax1.set_xlabel(var1_text, fontsize=13)
    ax1.set_ylabel(var2_text, fontsize=13)
    ax1.set_xlim(x_range)
    ax1.set_ylim(y_range)
    
    if(zero_lines):
        ax1.axhline(lw=1.0, color='k',ls='--')
        ax1.axvline(lw=1.0, color='k',ls='--')
    
    title = var1_text + "_" + var2_text + "_colour" + subset_text + ".eps"
    
    if(max_var_index >= save_index):
        fig1.savefig(title, format='eps', dpi=1000)
    
 

#Make histogram of correlations between two variables (e.g. DLLs)
def hist_2D(max_var_index, var1, var2, var1_text, var2_text, save_index=0):
    
    x = var1[0:max_var_index]
    y = var2[0:max_var_index]
    
    #Histogram. Doesn't really work
    fig1, ax1 = plt.subplots()
    ax1.cla()
    ax1.hist2d(x, y, bins=(100, 100), cmap=plt.cm.jet)
    #ax1.set_xlim(-100,100)
    #ax1.set_ylim(-60,60)
    ax1.axhline(lw=1.0, color='k',ls='--')
    ax1.axvline(lw=1.0, color='k',ls='--')
    if(max_var_index >= save_index):
        fig1.savefig(var1_text + "_" + var2_text + "_hist.eps", format='eps', dpi=1000)



#Plot general variables (via calling above plotting functions)
def plot_vars():
    
    #Get data:
    DLLe_k = get_data('RichDLLe', 'KAON')
    DLLmu_k = get_data('RichDLLmu', 'KAON')
    DLLk_k = get_data('RichDLLk', 'KAON')
    DLLp_k = get_data('RichDLLp', 'KAON')
    DLLd_k = get_data('RichDLLd', 'KAON')
    DLLbt_k = get_data('RichDLLbt', 'KAON')
    TrackP_k = get_data('TrackP', 'KAON')
    TrackPt_k = get_data('TrackPt', 'KAON')
    TrackChi2PerDof_k = get_data('TrackChi2PerDof', 'KAON')

    DLLd_p = get_data('RichDLLd', 'PION')
    DLLk_p = get_data('RichDLLk', 'PION')
    DLLp_p = get_data('RichDLLp', 'PION')

    #Make plots of correltions between two DLLs
    max_var_index = 10000
    
    ord_scatt(DLLk_k, DLLp_k, "DLLk_k", "DLLp_k", max_var_index, x_range=[-100,100], y_range=[-80,80], zero_lines=1, save_index=10000, size=0.5)
    
    #Using these currently.
    col_scatt(DLLk_k, DLLp_k, "DLLk_k", "DLLp_k", max_var_index, x_range=[-100,100], y_range=[-80,80], zero_lines=1, save_index=10000, size=0.5)
    col_scatt(DLLk_p, DLLp_p, "DLLk_p", "DLLp_p", max_var_index, x_range=[-100,100], y_range=[-80,80], zero_lines=1, save_index=10000, size=0.5)
    
    hist_2D(max_var_index, DLLk_k, DLLp_k, "DLLk_k", "DLLp_k", save_index=10000)

    col_scatt(DLLk_k, DLLd_k, "DLLk_k", "DLLd_k", max_var_index, x_range=[-100,100], y_range=[-80,80], zero_lines=1, save_index=10000, size=0.5)
    col_scatt(DLLk_p, DLLd_p, "DLLk_p", "DLLd_p", max_var_index, x_range=[-100,100], y_range=[-80,80], zero_lines=1, save_index=10000, size=0.5)

    max_var_index = 100000

    ord_scatt(DLLk_k, TrackP_k, "DLLk_k", "TrackP", max_var_index, size=0.5)
    ord_scatt(DLLk_k, TrackPt_k, "DLLk_k", "TrackPt", max_var_index, size=0.5)
    
    #Plotting one or two general variables 
    
#    max_index_0 = 100000 #Highest index number (so don't save over better version)
#    max_index_1 = 100000 #Plot P up this index number
#    size = 1
#    two_var_plots(0, max_index_1, DLLk_k, TrackChi2PerDof_k, "DLLk_k", "TrackChi2PerDof_k", size)
#    two_var_plots(max_index_0, max_index_1, DLLk_k, TrackP_k, "DLLk_k", "TrackP_k", size)
#    two_var_plots(max_index_0, max_index_1, DLLk_k, TrackPt_k, "DLLk_k", "TrackPt_k", size)

#    max_index_2 = 1000 #Maximum index of variable
#    one_var_plots(max_index_2, DLLe_k, "DLLe_k")
#    one_var_plots(max_index_2, DLLmu_k, "DLLmu_k")
#    one_var_plots(max_index_2, DLLk_k, "DLLk_k")
#    one_var_plots(max_index_2, DLLp_k, "DLLp_k")
#    one_var_plots(max_index_2, DLLd_k, "DLLd_k")
#    one_var_plots(max_index_2, DLLbt_k, "DLLbt_k")
#    one_var_plots(max_index_2, TrackP_k, "TrackP_k")
#    one_var_plots(max_index_2, TrackPt_k, "TrackPt_k")
#    one_var_plots(max_index_2, TrackChi2PerDof_k, "TrackChi2PerDof_k")


#Plot efficiency against momentum
def eff_mom_plot(p_points, source1_eff_0, source1_eff_5, source2_eff_0, source2_eff_5, DLL_part_1, DLL_part_2, particle_source_1, particle_source_2, p_max):
 
    title = DLL_part_1 + "_" + DLL_part_2 + "_" + particle_source_1 + "_" + particle_source_2 + "_" + str(p_max) + subset_text + ".eps"
    
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
    
    s1_0 = ax1.scatter(p_points, source1_eff_0, s = 5, marker = 'o', facecolors = 'none', edgecolors = 'r')
    s1_5 = ax1.scatter(p_points, source1_eff_5, s = 5, marker = 'o', color = 'r')
    s2_0 = ax1.scatter(p_points, source2_eff_0, s = 5, marker = 's', facecolors = 'none', edgecolors = 'k')
    s2_5 = ax1.scatter(p_points, source2_eff_5, s = 5, marker = 's', color = 'k')    
    ax1.legend((s1_0, s1_5, s2_0, s2_5), (process_1_text + ', ' + DLL_text + ' > 0', process_1_text + ', ' + DLL_text + ' > 5', process_2_text + ', ' + DLL_text + ' > 0', process_2_text + ', ' + DLL_text + ' > 5'), loc='upper right', ncol=2, fontsize=11)
    fig1.savefig(title, format='eps', dpi=1000)


###############################################################################
#Calculations
###############################################################################
        
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


#Calculate efficiency of particle identification for general DLL
def calc_eff(bins_no, DLL, DLL_lim, data_no, bounds, data, bin_var):
    
    particle_no = np.zeros(bins_no, dtype=float)
    tot_no = np.zeros(bins_no, dtype=float)

    for i in range(0, bins_no):

        DLL_arr = np.zeros(data_no, dtype=float)
        
        #Create bins (boolean depending on whether bin variavle lies in range or not)
        bins = (data[bin_var] >= bounds[i]) & (data[bin_var] < bounds[i+1])
    
        DLL_arr = np.multiply(bins, DLL) #Set DLL to 0 if not in variable range
        DLL_arr = np.subtract(DLL_arr, DLL_lim) #DLL Subtract limit e.g. 0 or 5 
        DLL_arr = np.clip(DLL_arr, 0, None) #Set all values less than 0 to 0
    
        particle_no[i] = np.count_nonzero(DLL_arr) #Count non-zero values
        tot_no[i] = np.sum(bins) #Sum events in momentum bin
        
#        print("Range: ", bounds[i], "-", bounds[i+1], "Total:", tot_no[i], " Particles:", particle_no[i])

        #Efficiency: divide number of events within DLL limit by total number
        eff = np.divide(particle_no, tot_no, out=np.zeros_like(particle_no), where=tot_no!=0)

    return eff


#Calculate and plot efficiency for K-> K and pi -> K
def eff_mom_calc(p_bins_no, p_max, uni_bins, exp_bins, exponent, DLL_part_1, DLL_part_2, particle_source_1, particle_source_2):

    #Calculate the momentum bin centres and bounds
    p_points, p_bounds = calc_p(p_bins_no, p_max, uni_bins,exp_bins, exponent)

    if(particle_source_1 == 'KAON'):
        data_1 = data_kaon
    elif(particle_source_1 == 'PION'):
        data_1 = data_pion

    if(particle_source_2 == 'KAON'):
        data_2 = data_kaon
    elif(particle_source_2 == 'PION'):
        data_2 = data_pion
    else:
        print("Please select either KAON or PION data")
    
    #Get data for DLLs including changing if the DLL is not x-pi
    if(DLL_part_2 == 'pi'):
        DLL1 = get_data('RichDLL' + DLL_part_1, particle_source_1)
        DLL2 = get_data('RichDLL' + DLL_part_1, particle_source_2)
    else:
        DLL1_1 = get_data('RichDLL' + DLL_part_1, particle_source_1)
        DLL1_2 = get_data('RichDLL' + DLL_part_2, particle_source_1)
        DLL1 = change_DLL(DLL1_1, DLL1_2)
        
        DLL2_1 = get_data('RichDLL' + DLL_part_1, particle_source_2)
        DLL2_2 = get_data('RichDLL' + DLL_part_2, particle_source_2)
        DLL2 = change_DLL(DLL2_1, DLL2_2)
            
    #Number of data points
    data_no_1 = len(DLL1)
    data_no_2 = len(DLL2)
    
    #Calculate the particle identification efficiencies for different DLL limits
        
    source1_eff_0 = calc_eff(p_bins_no, DLL1, 0, data_no_1, p_bounds, data_1, 'TrackP')
    source1_eff_5 = calc_eff(p_bins_no, DLL1, 5, data_no_1, p_bounds, data_1, 'TrackP')
    source2_eff_0 = calc_eff(p_bins_no, DLL2, 0, data_no_2, p_bounds, data_2, 'TrackP')
    source2_eff_5 = calc_eff(p_bins_no, DLL2, 5, data_no_2, p_bounds, data_2, 'TrackP')
    
    eff_mom_plot(p_points, source1_eff_0, source1_eff_5, source2_eff_0, source2_eff_5, DLL_part_1, DLL_part_2, particle_source_1, particle_source_2, p_max)


#Calculate and produce plots for Kaon ID efficiency and PION mis-ID efficiency
#Currently written for four different track numbers (0-400), but can very easily be changed to number of PVs

#'k', 'pi', 'KAON', 'PION', 'NumPVs', misid_bin_no, DLL_lim, DLL_no, phys_var_range
def id_misid_eff(DLL_particle, ref_particle, particle_source_1, particle_source_2, var_name, bins_no, DLL_lim, DLL_no, phys_var_range, x_range, var_range):

    if(particle_source_1 == 'KAON'):
        data_1 = data_kaon
    elif(particle_source_1 == 'PION'):
        data_1 = data_pion
    else:
        print("Please select either KAON or PION data")
        
    if(particle_source_2 == 'KAON'):
        data_2 = data_kaon
    elif(particle_source_2 == 'PION'):
        data_2 = data_pion
    else:
        print("Please select either KAON or PION data")
    
    #Get data for DLLs including changing if the DLL is not x-pi
    if(ref_particle == 'pi'):
        DLL1 = get_data('RichDLL' + DLL_particle, particle_source_1)
        DLL2 = get_data('RichDLL' + DLL_particle, particle_source_2)
    else:
        DLL1_1 = get_data('RichDLL' + DLL_particle, particle_source_1)
        DLL1_2 = get_data('RichDLL' + ref_particle, particle_source_1)
        DLL1 = change_DLL(DLL1_1, DLL1_2)
        
        DLL2_1 = get_data('RichDLL' + DLL_particle, particle_source_2)
        DLL2_2 = get_data('RichDLL' + ref_particle, particle_source_2)
        DLL2 = change_DLL(DLL2_1, DLL2_2)

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

    #Number of data points
    data_no_1 = len(DLL1)
    data_no_2 = len(DLL2)
    
    source_1_eff_av = np.zeros([bins_no, DLL_no-1])
    source_2_eff_av = np.zeros([bins_no, DLL_no-1])
    
    for i in range(bins_no):
        bounds=full_bounds[i:i+2]
        for j in range(0, DLL_no - 1):
            source_1_eff = calc_eff(1, DLL1, DLL_lims[j], data_no_1, bounds, data_1, var_name)
            source_1_eff_av[i, j] = np.average(source_1_eff)
            source_2_eff = calc_eff(1, DLL2, DLL_lims[j], data_no_2, bounds, data_2, var_name)
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
        
    fig_title =  "kID_pMID_eff_" + var_name + subset_text + ".eps"   
    ax1.legend(title=plot_title, loc='upper left', fontsize=11)
    fig1.savefig(fig_title, format='eps', dpi=1000)


###############################################################################
    
def plot_gen_hist(var, particle_source, bin_no='auto', x_range=None, y_range=None):
    
    var_data = get_data(var, particle_source)
    
    if var == 'TrackP':
        x_label_text = 'Momentum (GeV/c)'

    if var == 'TrackPt':
        x_label_text = 'Transverse Momentum (GeV/c)'

    title = var + "_" + particle_source + "_hist.eps"
    
    fig1, ax1 = plt.subplots()
    ax1.cla()
    
    if y_range is not None:
        ax1.set_ylim(bottom=0, top=y_range)
    
    if x_range is not None:
        ax1.set_xlim(x_range)
    
    ax1.set_xlabel(x_label_text)
    ax1.set_ylabel("Density of events")
        
    ax1.hist(var_data, bins=bin_no, range=x_range)    
    
    fig1.savefig(title, format='eps', dpi=2500)
    
    
def plot_DLL_hist(DLL_part_1, DLL_part_2, particle_source, bin_no='auto', x_range=None, y_range=None):
        
    #Get data for DLLs including changing if the DLL is not x-pi
    if(DLL_part_2 == 'pi'):
        DLL = get_data('RichDLL' + DLL_part_1, particle_source)
    else:
        DLL_1 = get_data('RichDLL' + DLL_part_1, particle_source)
        DLL_2 = get_data('RichDLL' + DLL_part_2, particle_source)
        DLL = change_DLL(DLL_1, DLL_2)   
    
    title = "DLL" + DLL_part_1 + "-" + DLL_part_2 + "_" + particle_source + "_hist" + subset_text + ".eps"
                
    if(DLL_part_1 == 'pi'):
        DLL_part_1 = r'$\pi$'
    elif(DLL_part_1 == 'k'):
        DLL_part_1 = 'K'
      
    if(DLL_part_2 == 'pi'):
        DLL_part_2 = r'$\pi$'
    elif(DLL_part_2 == 'k'):
        DLL_part_2 = 'K)'
    
    DLL_text = r'$\Delta LL ($' + DLL_part_1 + '-' + DLL_part_2 + ')'
        
    fig1, ax1 = plt.subplots()
    ax1.cla()
    
    if y_range is not None:
        ax1.set_ylim(bottom=0, top=y_range)

    if x_range is not None:
        ax1.set_xlim(x_range)
    
    ax1.set_xlabel(DLL_text)
    ax1.set_ylabel("Density of events")
    
    ax1.hist(DLL, bins=bin_no, range=x_range, density=True)
    
    fig1.savefig(title, format='eps', dpi=2500)

     
#    DLL_hist =  np.histogram(DLL,bins=bin_no,range=x_range)
#    DLL_reshaped = np.reshape(np.array(DLL), (-1,1))
#    qt = QuantileTransformer(n_quantiles=20000, output_distribution='normal')
#    DLL1_norm = qt.fit_transform(DLL_reshaped).squeeze()
#
#    fig2, ax2 = plt.subplots()
#    ax2.cla()
#
#    ax2.set_xlabel(DLL_text)
#    ax2.set_ylabel("Number of events")
#
#    if x_range is not None:
#        ax2.set_xlim(x_range)
#        
#    ax2.hist(DLL1_norm, bins=bin_no)
    

 ###############################################################################

def DLL_batch(DLL_part_1, DLL_part_2, particle_source_1, bin_no, x_min, x_max, y_max, batch_size = 10000):    
            
    #Get data for DLLs including changing if the DLL is not x-pi
    if(DLL_part_2 == 'pi'):
        DLL = get_data('RichDLL' + DLL_part_1, particle_source_1)
    else:
        DLL_1 = get_data('RichDLL' + DLL_part_1, particle_source_1)
        DLL_2 = get_data('RichDLL' + DLL_part_2, particle_source_1)
        DLL = change_DLL(DLL_1, DLL_2)   
    
    title = "DLL" + DLL_part_1 + "-" + DLL_part_2 + "_" + particle_source_1 + "_batch_hist.eps"
    
    if(DLL_part_1 == 'pi'):
        DLL_part_1 = r'$\pi$'
    elif(DLL_part_1 == 'k'):
        DLL_part_1 = 'K'
      
    if(DLL_part_2 == 'pi'):
        DLL_part_2 = r'$\pi$'
    elif(DLL_part_2 == 'k'):
        DLL_part_2 = 'K)'

    DLL_batch = DLL[np.random.randint(0, DLL.shape[0], size=batch_size)]      
    
    DLL_text = r'$\Delta LL ($' + DLL_part_1 + '-' + DLL_part_2 + ')'
        
    fig1, ax1 = plt.subplots()
    ax1.cla()
    ax1.set_ylim([0, y_max])
    ax1.set_xlabel(DLL_text)
    ax1.set_ylabel("Number of events")
    
    ax1.hist(DLL_batch, bins=bin_no, range=[x_min,x_max])
    
    fig1.savefig(title, format='eps', dpi=2500)
    


p_bins_no = 100 #Number of momentum bins
p_max = 100.0 #Maximum track momentum
uni_bins = 0 #Uniform bin sizes
exp_bins = 0 #Exponentially increasing bin sizes (if neither uni or exp, linear increas)
exponent = 2 #Exponent for logspace. Doesn't change anything currently as overspecified?

#Calculate and plot efficiency for K-> K and pi -> K
#eff_mom_calc(p_bins_no, p_max, uni_bins, exp_bins, exponent, 'k', 'pi', 'KAON', 'PION')

#Plot other varibles e.g. individual DLLs or correlations
#plot_vars()

#######################################################################################################################################################################

misid_bin_no = 4
DLL_lim = 15
DLL_no = 21

#'NumLongTracks'
phys_var_range = [0,400]
x_range = [0.2, 1]
#id_misid_eff('k', 'pi', 'KAON', 'PION', 'NumLongTracks', misid_bin_no, DLL_lim, DLL_no, phys_var_range, x_range, True)


#'NumPVs'
phys_var_range = [1,5]
x_range = [0.5, 1]
#id_misid_eff('k', 'pi', 'KAON', 'PION', 'NumPVs', misid_bin_no, DLL_lim, DLL_no, phys_var_range, x_range, False)

#######################################################################################################################################################################

#'RICH1EntryDist0'

#Increase in KID as dist increases
misid_bin_no = 4
phys_var_range = [0,40]
x_range = [0.2, 1]
id_misid_eff('k', 'pi', 'KAON', 'PION', 'RICH1EntryDist0', misid_bin_no, DLL_lim, DLL_no, phys_var_range, x_range, True)

##Shows increase initially, then all merge together
#misid_bin_no = 4
#phys_var_range = [0,100]
#x_range = [0.3, 1]
#id_misid_eff('k', 'pi', 'KAON', 'PION', 'RICH1EntryDist0', misid_bin_no, DLL_lim, DLL_no, phys_var_range, x_range, True)

##Unclear - increase initially then decrease  
#misid_bin_no = 4
#phys_var_range = [0,400]
#x_range = [0.3, 1]
#id_misid_eff('k', 'pi', 'KAON', 'PION', 'RICH1EntryDist0', misid_bin_no, DLL_lim, DLL_no, phys_var_range, x_range, True)

#######################################################################################################################################################################

#'RICH1EntryDist1'

#Increases with dist
misid_bin_no = 4
phys_var_range = [0,40]
x_range = [0.1, 1]
id_misid_eff('k', 'pi', 'KAON', 'PION', 'RICH1EntryDist1', misid_bin_no, DLL_lim, DLL_no, phys_var_range, x_range, True)

##Increases then merges together
#misid_bin_no = 4
#phys_var_range = [0,100]
#x_range = [0.2, 1]
#id_misid_eff('k', 'pi', 'KAON', 'PION', 'RICH1EntryDist1', misid_bin_no, DLL_lim, DLL_no, phys_var_range, x_range, True)
#
##Increase initially then decreases again
#misid_bin_no = 4
#phys_var_range = [0,400]
#x_range = [0.2, 1]
#id_misid_eff('k', 'pi', 'KAON', 'PION', 'RICH1EntryDist1', misid_bin_no, DLL_lim, DLL_no, phys_var_range, x_range, True)

#######################################################################################################################################################################

#'RICH1EntryDist2'

#Increases with dist
misid_bin_no = 4
phys_var_range = [0,40]
x_range = [0.1, 1]
id_misid_eff('k', 'pi', 'KAON', 'PION', 'RICH1EntryDist2', misid_bin_no, DLL_lim, DLL_no, phys_var_range, x_range, True)

##Incrases, last two together
#misid_bin_no = 4
#phys_var_range = [0,100]
#x_range = [0.2, 1]
#id_misid_eff('k', 'pi', 'KAON', 'PION', 'RICH1EntryDist2', misid_bin_no, DLL_lim, DLL_no, phys_var_range, x_range, True)
#
##Increase first then decrease
#misid_bin_no = 4
#phys_var_range = [0,400]
#x_range = [0.2, 1]
#id_misid_eff('k', 'pi', 'KAON', 'PION', 'RICH1EntryDist2', misid_bin_no, DLL_lim, DLL_no, phys_var_range, x_range, True)

#######################################################################################################################################################################

#'RICH1ExitDist0'

##Hard to tell - KID mostly increases with distance?
#misid_bin_no = 4
#phys_var_range = [0,40]
#x_range = [0.3, 1]
#id_misid_eff('k', 'pi', 'KAON', 'PION', 'RICH1ExitDist0', misid_bin_no, DLL_lim, DLL_no, phys_var_range, x_range, True)
#
##Increases with distance
misid_bin_no = 4
phys_var_range = [0,100]
x_range = [0.2, 1]
id_misid_eff('k', 'pi', 'KAON', 'PION', 'RICH1ExitDist0', misid_bin_no, DLL_lim, DLL_no, phys_var_range, x_range, True)

##Increase initially then somewhat decreases
#misid_bin_no = 4
#phys_var_range = [0,400]
#x_range = [0.3, 1]
#id_misid_eff('k', 'pi', 'KAON', 'PION', 'RICH1ExitDist0', misid_bin_no, DLL_lim, DLL_no, phys_var_range, x_range, True)

#######################################################################################################################################################################

#'RICH1ExitDis1'

##Unclear, all together
#misid_bin_no = 4
#phys_var_range = [0,40]
#x_range = [0.3, 1]
#id_misid_eff('k', 'pi', 'KAON', 'PION', 'RICH1ExitDist1', misid_bin_no, DLL_lim, DLL_no, phys_var_range, x_range, True)

#Increase with distance
misid_bin_no = 4
phys_var_range = [0,100]
x_range = [0.1, 1]
id_misid_eff('k', 'pi', 'KAON', 'PION', 'RICH1ExitDist1', misid_bin_no, DLL_lim, DLL_no, phys_var_range, x_range, True)

##Increase initially then somwwhat decreases back
#misid_bin_no = 4
#phys_var_range = [0,400]
#x_range = [0.3, 1]
#id_misid_eff('k', 'pi', 'KAON', 'PION', 'RICH1ExitDist1', misid_bin_no, DLL_lim, DLL_no, phys_var_range, x_range, True)

#######################################################################################################################################################################

#'RICH1ExitDist2'

##Unclear
#misid_bin_no = 4
#phys_var_range = [0,40]
#x_range = [0.3, 1]
#id_misid_eff('k', 'pi', 'KAON', 'PION', 'RICH1ExitDist2', misid_bin_no, DLL_lim, DLL_no, phys_var_range, x_range, True)

#Increases with distance
misid_bin_no = 4
phys_var_range = [0,100]
x_range = [0, 1]
id_misid_eff('k', 'pi', 'KAON', 'PION', 'RICH1ExitDist2', misid_bin_no, DLL_lim, DLL_no, phys_var_range, x_range, True)

##Increase initially then decrease somewhat
#misid_bin_no = 4
#phys_var_range = [0,400]
#x_range = [0.3, 1]
#id_misid_eff('k', 'pi', 'KAON', 'PION', 'RICH1ExitDist2', misid_bin_no, DLL_lim, DLL_no, phys_var_range, x_range, True)

######################################################################################################################################################################

#'RICH2EntryDist0'

#Increase as distance increases
misid_bin_no = 4
phys_var_range = [0,400]
x_range = [0.2, 1]
id_misid_eff('k', 'pi', 'KAON', 'PION', 'RICH2EntryDist0', misid_bin_no, DLL_lim, DLL_no, phys_var_range, x_range, True)

##Increase as distance increases, last two together
#misid_bin_no = 4
#phys_var_range = [0,800]
#x_range = [0.1, 1]
#id_misid_eff('k', 'pi', 'KAON', 'PION', 'RICH2EntryDist0', misid_bin_no, DLL_lim, DLL_no, phys_var_range, x_range, True)
#
##Increase as distance increases initially, then decrease
#misid_bin_no = 4
#phys_var_range = [0,1600]
#x_range = [0.1, 1]
#id_misid_eff('k', 'pi', 'KAON', 'PION', 'RICH2EntryDist0', misid_bin_no, DLL_lim, DLL_no, phys_var_range, x_range, True)

#######################################################################################################################################################################

#'RICH2EntryDist1'

##Increase as distance increases
misid_bin_no = 4
phys_var_range = [0,400]
x_range = [0.1, 1]
id_misid_eff('k', 'pi', 'KAON', 'PION', 'RICH2EntryDist1', misid_bin_no, DLL_lim, DLL_no, phys_var_range, x_range, True)

#Increase as distance increases
#misid_bin_no = 4
#phys_var_range = [0,800]
#x_range = [0.2, 1]
#id_misid_eff('k', 'pi', 'KAON', 'PION', 'RICH2EntryDist1', misid_bin_no, DLL_lim, DLL_no, phys_var_range, x_range, True)

##Increase as distance increases initially, then decrease
#misid_bin_no = 4
#phys_var_range = [0,1600]
#x_range = [0.1, 1]
#id_misid_eff('k', 'pi', 'KAON', 'PION', 'RICH2EntryDist1', misid_bin_no, DLL_lim, DLL_no, phys_var_range, x_range, True)

#######################################################################################################################################################################

#'RICH2EntryDist2'

##Increase as distance increases
#misid_bin_no = 4
#phys_var_range = [0,400]
#x_range = [0.1, 1]
#id_misid_eff('k', 'pi', 'KAON', 'PION', 'RICH2EntryDist2', misid_bin_no, DLL_lim, DLL_no, phys_var_range, x_range, True)

#Increase as distance increases
misid_bin_no = 4
phys_var_range = [0,800]
x_range = [0.1, 1]
id_misid_eff('k', 'pi', 'KAON', 'PION', 'RICH2EntryDist2', misid_bin_no, DLL_lim, DLL_no, phys_var_range, x_range, True)

##Increase as distance increases, overlapping at end
#misid_bin_no = 4
#phys_var_range = [0,1600]
#x_range = [0.1, 1]
#id_misid_eff('k', 'pi', 'KAON', 'PION', 'RICH2EntryDist2', misid_bin_no, DLL_lim, DLL_no, phys_var_range, x_range, True)

#######################################################################################################################################################################

#'RICH2ExitDist0'

##Increase as distance increases
#misid_bin_no = 4
#phys_var_range = [0,400]
#x_range = [0.1, 1]
#id_misid_eff('k', 'pi', 'KAON', 'PION', 'RICH2ExitDist0', misid_bin_no, DLL_lim, DLL_no, phys_var_range, x_range, True)

#Increase as distance increases
misid_bin_no = 4
phys_var_range = [0,800]
x_range = [0.2, 1]
id_misid_eff('k', 'pi', 'KAON', 'PION', 'RICH2ExitDist0', misid_bin_no, DLL_lim, DLL_no, phys_var_range, x_range, True)

##'RICH2ExitDist0'
##Increase as distance increases initially, then on top
#misid_bin_no = 4
#phys_var_range = [0,1600]
#x_range = [0.1, 1]
#id_misid_eff('k', 'pi', 'KAON', 'PION', 'RICH2ExitDist0', misid_bin_no, DLL_lim, DLL_no, phys_var_range, x_range, True)

#######################################################################################################################################################################

#'RICH2ExitDist1'

##Increase as distance increases, a bit unclear
#misid_bin_no = 4
#phys_var_range = [0,400]
#x_range = [0.1, 1]
#id_misid_eff('k', 'pi', 'KAON', 'PION', 'RICH2ExitDist1', misid_bin_no, DLL_lim, DLL_no, phys_var_range, x_range, True)

#Increase as distance increases
misid_bin_no = 4
phys_var_range = [0,800]
x_range = [0.1, 1]
id_misid_eff('k', 'pi', 'KAON', 'PION', 'RICH2ExitDist1', misid_bin_no, DLL_lim, DLL_no, phys_var_range, x_range, True)

##Increase as distance increases, last two on top of each other ish
#misid_bin_no = 4
#phys_var_range = [0,1600]
#x_range = [0.1, 1]
#id_misid_eff('k', 'pi', 'KAON', 'PION', 'RICH2ExitDist1', misid_bin_no, DLL_lim, DLL_no, phys_var_range, x_range, True)

#######################################################################################################################################################################

#'RICH2ExitDist2'

##Increase as distance increases probably?
#misid_bin_no = 4
#phys_var_range = [0,400]
#x_range = [0.1, 1]
#id_misid_eff('k', 'pi', 'KAON', 'PION', 'RICH2ExitDist2', misid_bin_no, DLL_lim, DLL_no, phys_var_range, x_range, True)

#Increase as distance increases
misid_bin_no = 4
phys_var_range = [0,800]
x_range = [0.1, 1]
id_misid_eff('k', 'pi', 'KAON', 'PION', 'RICH2ExitDist2', misid_bin_no, DLL_lim, DLL_no, phys_var_range, x_range, True)

##Increase as distance increases, last two on top of each other
#misid_bin_no = 4
#phys_var_range = [0,1600]
#x_range = [0.1, 1]
#id_misid_eff('k', 'pi', 'KAON', 'PION', 'RICH2ExitDist2', misid_bin_no, DLL_lim, DLL_no, phys_var_range, x_range, True)


#######################################################################################################################################################################

#Increase in KID as num decreases
#'RICH1ConeNum'
misid_bin_no = 4
phys_var_range = [0,24]
x_range = [0, 1]
id_misid_eff('k', 'pi', 'KAON', 'PION', 'RICH1ConeNum', misid_bin_no, DLL_lim, DLL_no, phys_var_range, x_range, True)

#Increase in KID as num decreases
#'RICH2ConeNum'
misid_bin_no = 4
phys_var_range = [0,4]
x_range = [0.1, 1]
id_misid_eff('k', 'pi', 'KAON', 'PION', 'RICH2ConeNum', misid_bin_no, DLL_lim, DLL_no, phys_var_range, x_range, True)

#######################################################################################################################################################################

#Plot histograms of DLLs
#Args: DLL_part_1, DLL_part_2, particle_source, bin_no=200, x_range=None, y_range=None

#plot_DLL_hist('k', 'pi', 'PION', 325, -60, 20, 250000)

#Mostly smooth but narrow spikes at DLL=0. Density = False
#plot_DLL_hist('e', 'pi', 'KAON', 500, [-40, 20], 180000)
#plot_DLL_hist('mu', 'pi', 'KAON', 500, [-20, 15], 200000)
#plot_DLL_hist('k', 'pi', 'KAON', 750, [-40, 80], 80000)
#plot_DLL_hist('p', 'pi', 'KAON', 600, [-40, 60], 80000)
#plot_DLL_hist('d', 'pi', 'KAON', 600, [-40, 60], 80000)
#plot_DLL_hist('bt', 'pi', 'KAON', 600, [-40, 60], 80000)
#
#plot_DLL_hist('e', 'pi', 'PION', 500, [-80, 20], 120000)
#plot_DLL_hist('mu', 'pi', 'PION', 500, [-50, 20], 200000)
#plot_DLL_hist('k', 'pi', 'PION', 750, [-60, 20], 125000)
#plot_DLL_hist('p', 'pi', 'PION', 600, [-60, 40], 200000)
#plot_DLL_hist('d', 'pi', 'PION', 600, [-60, 40], 200000)
#plot_DLL_hist('bt', 'pi', 'PION', 600, [-60, 40], 200000)


#######################################################################################################################################################################
#Mostly these in use

#New data: Mostly smooth but narrow spikes at DLL=0
plot_DLL_hist('e', 'pi', 'KAON', 500, [-40, 20], 0.14)
plot_DLL_hist('mu', 'pi', 'KAON', 500, [-20, 15], 0.3)
plot_DLL_hist('k', 'pi', 'KAON', 750, [-40, 80], 0.05)
plot_DLL_hist('p', 'pi', 'KAON', 600, [-40, 60], 0.05)
plot_DLL_hist('d', 'pi', 'KAON', 600, [-40, 60], 0.05)
plot_DLL_hist('bt', 'pi', 'KAON', 600, [-40, 60], 0.05)

plot_DLL_hist('e', 'pi', 'PION', 500, [-80, 20], 0.06)
plot_DLL_hist('mu', 'pi', 'PION', 500, [-50, 20], 0.14)
plot_DLL_hist('k', 'pi', 'PION', 750, [-60, 20], 0.1)
plot_DLL_hist('p', 'pi', 'PION', 600, [-60, 40], 0.1)
plot_DLL_hist('d', 'pi', 'PION', 600, [-60, 40], 0.1)
plot_DLL_hist('bt', 'pi', 'PION', 600, [-60, 40], 0.1)

#######################################################################################################################################################################
#NOT USING

#Old data: Mostly smooth but narrow spikes at DLL=0
#plot_DLL_hist('e', 'pi', 'KAON', 500, [-40, 20], 0.14)
#plot_DLL_hist('mu', 'pi', 'KAON', 500, [-20, 15], 0.3)
#plot_DLL_hist('k', 'pi', 'KAON', 750, [-40, 80], 0.05)
#plot_DLL_hist('p', 'pi', 'KAON', 600, [-40, 60], 0.05)
#plot_DLL_hist('d', 'pi', 'KAON', 600, [-40, 60], 0.05)
#plot_DLL_hist('bt', 'pi', 'KAON', 600, [-40, 60], 0.05)
#
#plot_DLL_hist('e', 'pi', 'PION', 500, [-80, 20], 0.06)
#plot_DLL_hist('mu', 'pi', 'PION', 500, [-50, 20], 0.14)
#plot_DLL_hist('k', 'pi', 'PION', 750, [-60, 20], 0.12)
#plot_DLL_hist('p', 'pi', 'PION', 600, [-60, 40], 0.12)
#plot_DLL_hist('d', 'pi', 'PION', 600, [-60, 40], 0.12)
#plot_DLL_hist('bt', 'pi', 'PION', 600, [-60, 40], 0.12)

########################################################################################################

#Plot histogram of TrackP
#Args: var, particle_source, bin_no=200, x_range=None, y_range=None

#plot_gen_hist('TrackP', 'KAON', 750, [0,100000], 120000)
#plot_gen_hist('TrackPt', 'KAON', 750, [0,5000], 100000)

#plot_gen_hist('TrackP', 'PION', 750, [0,70000], 125000)
#plot_gen_hist('TrackPt', 'PION', 750, [0,3500], 100000)

########################################################################################################

#Measure total run time for script
t_final = time.time()
print("Total run time = ", t_final - t_init)
