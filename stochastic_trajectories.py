#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 11 15:59:57 2022

@author: xuefeng
"""

from optimization_functions import simf_full_refill, optimize_gs_drydown, eller, gcmaxf, gs_min
from optimization_functions import n, Z, diy, Mmm
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


''' WHY ARE THE EXPECTED GAINS SO DIFFERENT??? 
COMPARE OPTIMIZED PARAMETERS FOR LONG-TERM VS. SINGLE DRY DOWN? '''

#%% 
# Simulation parameters 
s0 = 0.3            # -, initial relative soil moisture
duration = 30       # days, total duration of simulation
mumol = 10**6       # umol/mol, converstion factor
tincre = 0.1 

# Figure parameters 
c2 = 'firebrick'                    # line color for dynamic feedback optimization
t_arr = np.arange(duration)         # initialize time array for plotting 

# climate and plant parameters
freq = 0.15          # mean frequency of rainfall 
MAP = 1000          # mm, mean annual precipitation
gamma = (n*Z) / (MAP*Mmm / (freq*diy))
df = pd.read_csv('../expected_Cnet_dynfb.csv')  
a, b, c = [df[(df.freq==freq) &(df.MAP==MAP)][i].values[0] for i in ['a', 'b', 'c']]

#%%
''' SET UP STOCHASTIC DRY DOWNS '''

# simulate rainfall 
size = int(duration/tincre)
depthExp = np.random.exponential(1/gamma * np.ones(size)) 
freqUnif = np.random.random(size=size)
    
depths = np.zeros(size)
# the occurence of rainfall in any independent interval is lam*dt
yesrain = freqUnif<np.tile(freq,size)*tincre
depths[yesrain] = depthExp[yesrain] # rain falls according to prob within an increment

# cut into consecutive dry downs 
irain = np.where(depths>0)[0]; irain2 = [0] + irain.tolist()
duras = np.diff(irain, prepend=0); duras2 = duras.tolist() + [size - irain[-1]]

s_arr = np.zeros((2, size))
B_arr = np.zeros_like(s_arr)
g_arr = np.zeros_like(s_arr)
E_arr = np.zeros_like(s_arr)
slast = 0

for i, (ira, d) in enumerate(zip(irain2, duras2)):
    
    if i==0: 
        s0 = s0
        istart = 0
    else: 
        s0 = min(1, slast+depths[ira])
        istart = ira 
    iend = istart + d
    
    # simulation results using eller's instantaneous gs function 
    s1, ps1, gc1, net_gain1, E1 = simf_full_refill(eller, s0=s0, duration=d, output_option='All')
    slast = s1[-1]
    
    # designate values from long-term optimization 
    gs_dynfb = lambda ps, p50, pL, pk, k_func: min(a*np.exp(-(ps/c)**b)-gs_min, gcmaxf(ps))
    s2, ps2, gc2, net_gain2, E2 = simf_full_refill(gs_dynfb, s0=s0, duration=d, output_option='All')
    slast = s2[-1]
    
    # save
    s_arr[0, istart:iend], s_arr[1, istart:iend] = s1, s2 
    B_arr[0, istart:iend], B_arr[1, istart:iend] = net_gain1, net_gain2
    g_arr[0, istart:iend], g_arr[1, istart:iend] = gc1, gc2
    E_arr[0, istart:iend], E_arr[1, istart:iend] = E1, E2
    
#%%
''' PLOTTING '''
fig = plt.figure(figsize=(6.5,6))

ax1 = fig.add_subplot(311)
ax2 = ax1.twinx()
ax1.plot(np.arange(0,duration,tincre), s_arr[0], '--', label='Instantaneous')
ax1.plot(np.arange(0,duration,tincre), s_arr[1], c=c2, lw=1.5, label='Dynamic feedback')
ax1.hlines(0, 0, duration, color='lightgrey', lw=1)
ax2.vlines((irain)*tincre, 0*np.ones(len(irain)), (depths[depths>0])*np.ones(len(irain)) * (n*Z) / Mmm, lw=6, color='lightgrey', alpha=0.5)
ax1.set_ylabel('Relative soil moisture')
ax2.set_ylabel('Rainfall depths (mm)')
ax1.legend()
ax1.text(0.05, 0.9, 'a', transform=ax1.transAxes, fontsize=14, fontweight='bold', va='top')


ax = plt.subplot(3,1,2)
plt.plot(np.arange(0,duration,tincre), g_arr[0], '--', label='Instantaneous')
plt.plot(np.arange(0,duration,tincre), g_arr[1], c=c2, lw=1.5, label='Dynamic feedback')
plt.ylabel('Stomatal conductance \n (mol s$^{-1}$ m$^{2}$)')
ax.text(0.05, 0.9, 'b', transform=ax.transAxes, fontsize=14, fontweight='bold', va='top')

ax = plt.subplot(3,1,3)
plt.plot(np.arange(0,duration,tincre), B_arr[0]*mumol, '--', label='Instantaneous')
plt.plot(np.arange(0,duration,tincre), B_arr[1]*mumol, c=c2, lw=1.5, label='Dynamic feedback')
plt.ylabel('Net carbon gain \n ($\mu$mol s$^{-1}$ m$^{2}$)')
plt.xlabel('Days')
ax.text(0.05, 0.9, 'c', transform=ax.transAxes, fontsize=14, fontweight='bold', va='top')

plt.tight_layout()
plt.show()
    
    