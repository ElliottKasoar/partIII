#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 26 00:39:06 2018

@author: Elliott
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from scipy.stats import gaussian_kde
import math
import time

#Time total run
t_init = time.time()

#help(data_kaon) 
#help(data_kaon.values) 

plt.rcParams['agg.path.chunksize'] = 10000 #Needed for plotting lots of data?

#Import data from kaons and pions
datafile_kaon = '../Data/PID-train-data-KAONS.hdf' 
data_kaon = pd.read_hdf(datafile_kaon, 'KAONS' ) 
print(data_kaon.columns)

datafile_pion = '../Data/PID-train-data-PIONS.hdf' 
data_pion = pd.read_hdf(datafile_pion, 'PIONS' ) 
print(data_pion.columns)


###############################################################################
#Basic data manipulation e.g. selecting columns of data and changing DLLs


#Get column from kaon or pion data
def get_data(var_type, particle_source):
    
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


###############################################################################
#General plotting functions
###############################################################################

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

#Make plots of correlations between two DLLs: scatter, scatter w/ colour and histogram
def DLL_corr_plots(max_index, DLL1, DLL2, DLL1_text, DLL2_text):
    
    x = DLL1[0:max_index]
    y = DLL2[0:max_index]
    
    x_label = "DLL" + DLL1_text
    y_label = "DLL" + DLL2_text
    
    #Ordinary scatter plot
    fig1, ax1 = plt.subplots()
    ax1.cla()
    ax1.scatter(x,y, s=1)
    ax1.set_xlabel(x_label)
    ax1.set_ylabel(y_label)
    ax1.set_xlim(-150,150)
    ax1.set_ylim(-80,80)
    ax1.axhline(lw=1.0, color='k',ls='--')
    ax1.axvline(lw=1.0, color='k',ls='--')
    if(max_index >= 1000000):
        fig1.savefig(x_label + "_" + y_label + ".eps", format='eps', dpi=1000)
    
    # Calculate the point density
    xy = np.vstack([x,y])
    z = gaussian_kde(xy)(xy)
    
    # Sort the points by density, so that the densest points are plotted last
    idx = z.argsort()
    x, y, z = x[idx], y[idx], z[idx]
    
    #Scatter plot with Gaussian colour scale
    fig2, ax2 = plt.subplots()
    ax2.cla()
    ax2.scatter(x, y, c=z, s=1, edgecolor='')
    ax2.set_xlabel(x_label)
    ax2.set_ylabel(y_label)
    ax2.set_xlim(-100,100)
    ax2.set_ylim(-60,60)
    ax2.axhline(lw=1.0, color='k',ls='--')
    ax2.axvline(lw=1.0, color='k',ls='--')
    if(max_index >= 100000):
        fig2.savefig(x_label + "_" + y_label + "_colour.eps", format='eps', dpi=1000)
    
    #Histogram. Doesn't really work
    fig3, ax3 = plt.subplots()
    ax3.cla()
    ax3.hist2d(x, y, bins=(100, 100), cmap=plt.cm.jet)
    #ax3.set_xlim(-100,100)
    #ax3.set_ylim(-60,60)
    ax3.axhline(lw=1.0, color='k',ls='--')
    ax3.axvline(lw=1.0, color='k',ls='--')
    if(max_index >= 10000):
        fig3.savefig(x_label + "_" + y_label + "_hist.eps", format='eps', dpi=1000)


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

    #Make plots of correltions between two DLLs
    max_index = 10000
    DLL_corr_plots(max_index, DLLk_k, DLLp_k, "k_k", "p_k")
    DLL_corr_plots(max_index, DLLp_k, DLLk_k, "p_k", "k_k")
    
    #Plotting one or two general variables 
    
    max_index_0 = 100000 #Highest index number (so don't save over better version)
    max_index_1 = 100000 #Plot P up this index number
    size = 1
    two_var_plots(0, max_index_1, DLLk_k, TrackChi2PerDof_k, "DLLk_k", "TrackChi2PerDof_k", size)
    two_var_plots(max_index_0, max_index_1, DLLk_k, TrackP_k, "DLLk_k", "TrackP_k", size)
    
    max_index_2 = 1000 #Maximum index of variable
    one_var_plots(max_index_2, DLLe_k, "DLLe_k")
    one_var_plots(max_index_2, DLLmu_k, "DLLmu_k")
    one_var_plots(max_index_2, DLLk_k, "DLLk_k")
    one_var_plots(max_index_2, DLLp_k, "DLLp_k")
    one_var_plots(max_index_2, DLLd_k, "DLLd_k")
    one_var_plots(max_index_2, DLLbt_k, "DLLbt_k")
    one_var_plots(max_index_2, TrackP_k, "TrackP_k")
    one_var_plots(max_index_2, TrackPt_k, "TrackPt_k")
    one_var_plots(max_index_2, TrackChi2PerDof_k, "TrackChi2PerDof_k")


#Plot efficiency against momentum
def eff_mom_plot(p_points, source1_eff_0, source1_eff_5, source2_eff_0, source2_eff_5, DLL_part_1, DLL_part_2, particle_source_1, particle_source_2):
 
    title = DLL_part_1 + "_" + DLL_part_2 + "_" + particle_source_1 + "_" + particle_source_2
    
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
    ax1.set_ylim(0,1.2)
    ax1.set_xlabel('Momentum (GeV/c)')
    ax1.set_ylabel('Efficiency')
    ax1.yaxis.set_minor_locator(AutoMinorLocator(4))
    
    s1_0 = ax1.scatter(p_points, source1_eff_0, s = 5, marker = 'o', facecolors = 'none', edgecolors = 'r')
    s1_5 = ax1.scatter(p_points, source1_eff_5, s = 5, marker = 'o', color = 'r')
    s2_0 = ax1.scatter(p_points, source2_eff_0, s = 5, marker = 's', facecolors = 'none', edgecolors = 'k')
    s2_5 = ax1.scatter(p_points, source2_eff_5, s = 5, marker = 's', color = 'k')    
    ax1.legend((s1_0, s1_5, s2_0, s2_5), (process_1_text + ', ' + DLL_text + ' > 0)', process_1_text + ', ' + DLL_text + ' > 5)', process_2_text + ', ' + DLL_text + ' > 0', process_2_text + ', ' + DLL_text + ' > 5)'), loc='upper right', ncol=2, fontsize=8)
    fig1.savefig(title + ".eps", format='eps', dpi=1000)


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
            p_bounds[j+1] = (j+1) * p_max * 10e3 / p_bins_no
    
    #Expontential bin spacing
    elif(exp_bins):
        p_bounds = np.logspace(0, math.log(p_max,exponent), num = p_bins_no, base=exponent)        
        p_bounds = np.insert(p_bounds,0,0)
        p_bounds = np.multiply(p_bounds,10e3)

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

        p_bounds = np.multiply(p_bounds,10e3)

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
    
    eff_mom_plot(p_points, source1_eff_0, source1_eff_5, source2_eff_0, source2_eff_5, DLL_part_1, DLL_part_2, particle_source_1, particle_source_2)


def id_misid_eff(bins_no, DLL_lim, DLL_no, DLL_part_1, DLL_part_2, particle_source_1, particle_source_2):

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

    full_bounds = np.linspace(0, 400, num = bins_no + 1)
    DLL_lims = np.linspace(0, DLL_lim, DLL_no)

    #Number of data points
    data_no_1 = len(DLL1)
    data_no_2 = len(DLL2)
    
    source1_eff_av = np.zeros([bins_no, DLL_no-1])
    source2_eff_av = np.zeros([bins_no, DLL_no-1])
    
    for i in range(0, bins_no):
        bounds=full_bounds[i:i+2]
        for j in range(0, DLL_no - 1):
            source1_eff = calc_eff(1, DLL1, DLL_lims[j], data_no_1, bounds, data_1, 'NumLongTracks')
            source1_eff_av[i, j] = np.average(source1_eff)
            source2_eff = calc_eff(1, DLL2, DLL_lims[j], data_no_2, bounds, data_2, 'NumLongTracks')
            source2_eff_av[i, j] = np.average(source2_eff)
        
    fig1, ax1 = plt.subplots()
    ax1.cla()
    ax1.set_xlim(0.2, 1)
    ax1.set_xlabel('Kaon ID Efficiency')
    ax1.set_ylabel('Pion Mis-ID Efficiency')
#    ax1.yaxis.set_minor_locator(AutoMinorLocator(4))
    ax1.semilogy(source1_eff_av[0,:], source2_eff_av[0,:], 'yo-', markersize=4)
    ax1.semilogy(source1_eff_av[1,:], source2_eff_av[1,:], 'rs-', markersize=4)
    ax1.semilogy(source1_eff_av[2,:], source2_eff_av[2,:], 'b^-', markersize=4)
    ax1.semilogy(source1_eff_av[3,:], source2_eff_av[3,:], 'gv-', markersize=4)

    print(source1_eff_av[0,:], source2_eff_av[0,:])

###############################################################################

p_bins_no = 50 #Number of momentum bins
p_max = 50.0 #Maximum track momentum
uni_bins = 0 #Uniform bin sizes
exp_bins = 0 #Exponentially increasing bin sizes (if neither uni or exp, linear increas)
exponent = 2 #Exponent for logspace. Doesn't change anything currently as overspecified?

#Calculate and plot efficiency for K-> K and pi -> K
#eff_mom_calc(p_bins_no, p_max, uni_bins, exp_bins, exponent, 'k', 'pi', 'KAON', 'PION')

#Plot other varibles e.g. individual DLLs or correlations
#plot_vars()

track_bins_no = 4
DLL_lim = 15
DLL_no = 21
id_misid_eff(track_bins_no, DLL_lim, DLL_no, 'k', 'pi', 'KAON', 'PION')


#Measure total run time for script
t_final = time.time()
print("Total run time = ", t_final - t_init)