#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 26 16:01:27 2021

@author: xuefeng
"""

from optimization_functions import simf_no_embolism, eller, optimize_gs_drydown, gs_min, gcmaxf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#%%

# Optimization parameters
rerun = 1                               # rerun optimization, can set to 0 if file has already been saved
file_name = 'Cnet_dynfb_vs_instant.csv'
s0_arr = np.arange(0.3, 1.01, 0.1)      # range of intitial soil moisture conditions
dura_arr = np.arange(10, 151, 20)       # range of dry-down durations

# Plotting parameters 
col = 'firebrick'           # color for dynamic feedback optimization 
col2 = '#1f77b4'            # color for instantaneous optimization
n_contours = 5              # number of contours
mumol = 10**6               # umol/mol, converstion factor

#%%
''' OPTIMIZATION ''' 

if rerun: 
    Cnet_eller = np.zeros((len(s0_arr), len(dura_arr)))
    Cnet_dynfb = np.zeros_like(Cnet_eller)
    for i, s0 in enumerate(s0_arr):
        for j, dura in enumerate(dura_arr):
            print('Init soil moisture s0:', round(s0,1), 'Dry down duration (days):, ', dura)
            
            # Compares averaged net gain instead of cumulative net gain, in umol m-2 s-1
            _, _, _, net_gain1, _ = simf_no_embolism(eller, s0=s0, duration=dura, output_option='All')
            Cnet_eller[i,j] = np.mean(net_gain1)
            
            a, b, c = optimize_gs_drydown(simf_no_embolism, s0=s0, duration=dura)['params'].values()
            gs_dynfb = lambda ps: min(a*np.exp(-(ps/c)**b)-gs_min, gcmaxf(ps))
            _, _, _, net_gain2, _ = simf_no_embolism(gs_dynfb, s0=s0, duration=dura, output_option='All')
            Cnet_dynfb[i,j] = np.mean(net_gain2)  

    # Create dataframe as repository of results            
    df = pd.DataFrame([[s0_arr[i], dura_arr[j], Cnet_dynfb[i,j], Cnet_eller[i,j]] for i in range(len(s0_arr)) for j in range(len(dura_arr))],
                      columns=['s0', 'duration', 'Cnet_dynfb', 'Cnet_eller'] )
    df.to_csv(file_name)
    
else: 
    df = pd.read_csv(file_name) 

#%% 
''' EXTRACTING RESULTS '''

X,Y = np.meshgrid(dura_arr, s0_arr)
z = np.zeros((len(s0_arr), len(dura_arr)))
for i,s0 in enumerate(s0_arr):
    for j,dura in enumerate(dura_arr):
        
        # net carbon gain in units of mol m-2 s-1
        Cnet_dynfb = df[(df.s0==s0) & (df.duration==dura)]['Cnet_dynfb'].values[0] 
        Cnet_eller = df[(df.s0==s0) & (df.duration==dura)]['Cnet_eller'].values[0] 
        # difference between dynamic feedback and instantaneous optimization
        z[i,j] = (Cnet_dynfb - Cnet_eller) * mumol


#%% 
''' PLOTTING ''' 

plt.figure(figsize=(11,3.5))

ax = plt.subplot(131)
plt.plot(s0_arr, df[df.duration==30]['Cnet_eller'] * mumol, '--', color=col2, label='Instantaneous')
plt.plot(s0_arr, df[df.duration==30]['Cnet_dynfb'] * mumol, color=col, label='Dynamic feedback')
plt.xlabel('Initial relative soil moisture (-) ')
plt.ylabel('Mean net carbon gain rate ($\mathrm{\mu mol\ m^{-2} s^{-1}}$)')
ax.text(0.05, 0.95, 'a', transform=ax.transAxes, fontsize=14, fontweight='bold', va='top')
plt.legend()

ax = plt.subplot(132)
plt.plot(dura_arr, df[np.isclose(df.s0.values, 0.6)]['Cnet_eller'] * mumol, '--', color=col2, label='Instantaneous')
plt.plot(dura_arr, df[np.isclose(df.s0.values, 0.6)]['Cnet_dynfb'] * mumol, color=col, label='Dynamic feedback')
plt.xlabel(' Dry down duration (days) ')
plt.ylabel('Mean net carbon gain rate ($\mathrm{\mu mol\ m^{-2} s^{-1}}$)')
ax.text(0.05, 0.95, 'b', transform=ax.transAxes, fontsize=14, fontweight='bold', va='top')

ax = plt.subplot(133)
CS = ax.contour(X,Y,z, n_contours, cmap='binary')
plt.ylabel( 'Initial relative soil moisture (-)')
plt.xlabel( 'Dry down duration (days)')
plt.title('$\mathrm{\Delta}$ Mean net carbon gain rate ($\mathrm{\mu mol\ m^{-2} s^{-1}}$)', fontsize=10)
plt.clabel(CS)
ax.text(0.05, 0.95, 'c', transform=ax.transAxes, fontsize=14, fontweight='bold', va='top')

plt.tight_layout()
plt.savefig('./Figures/Figure_2.png', dpi=300)
plt.show()

