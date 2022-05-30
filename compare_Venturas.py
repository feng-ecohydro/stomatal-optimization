#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 22 16:50:12 2021

@author: xuefeng
"""

import pandas as pd
import matplotlib.pyplot as plt
from optimization_functions import simf_full_refill, optimize_gs_drydown, eller, gcmaxf, gs_min
import numpy as np

#%%
''' SET UP''' 

# Simulation parameters 
s0 = 0.3           # -, initial relative soil moisture
duration = 30      # days, duration of simulation
mumol = 10**6       # umol/mol, converstion factor
mmol = 10**3        # mmol/mol, converstion factor

# Figure parameters 
c2 = 'firebrick'                    # line color for dynamic feedback optimization
t_arr = np.arange(duration)         # initialize time array for plotting 
p_arr = np.arange(-5, 0, 0.01)      # initialize water potential array for plotting

#%% 
''' OPTIMIZATION ''' 

# simulation results using eller's instantaneous gs function 
s1, ps1, gc1, net_gain1, E1 = simf_full_refill(eller, s0=s0, duration=duration, output_option='All')

# simulation using stomatal conductance function based on long-term, dynamic feedback optimization 
a, b, c = optimize_gs_drydown(simf_full_refill, s0, duration)['params'].values()
gs_dynfb = lambda ps, p50, pL, pk, k_func: min(a*np.exp(-(ps/c)**b)-gs_min, gcmaxf(ps))
s2, ps2, gc2, net_gain2, E2 = simf_full_refill(gs_dynfb, s0=s0, duration=duration, output_option='All')

# Figure setup
net_gain1 = net_gain1*mumol    # convert net carbon gain rate to umol m-2 s-1
net_gain2 = net_gain2*mumol    # net_gainf(gs_dynfb(p_arr)) * mumol
gc1 = gc1*mmol
gc2 = gc2*mmol
E1 = E1*mmol
E2 = E2*mmol

#%%
''' DATA COMPARISON '''
data = pd.read_excel('Venturas2018.xlsx', sheet_name='Measurements') 
Treatment = data['Treatment']
tr = 'SDr'

fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(12, 3))

for i, (target, unit, (model1, model2)) in enumerate(zip(['Anet', 'Gw', 'E'], 
                                                         ['umol s-1 m2', 'mmol s-1 m2', 'mmol s-1 m-2'],
                                                         [(net_gain1, net_gain2), (gc1, gc2), (E1, E2)])):
    ax = plt.subplot(1,4,i+1)
    x = -data[Treatment==tr]['Ppd_Mean']
    y = data[Treatment==tr][target+'_Mean']
    
    plt.errorbar(x, y, yerr = data[Treatment==tr][target+'_CI'], fmt='o', label=tr)   
    plt.plot(ps1, model1, '--', label='Instantaneous')
    plt.plot(ps2, model2, c=c2, lw=1.5, label='Dynamic feedback')
    
    plt.xlabel('Soil water potential (MPa)')
    plt.ylabel(target + ' ('+ unit+')')
    plt.xlim(-3,0)

plt.subplot(1,4,4)
target = 'Anet'; start_day = 171
meas_days = data[Treatment==tr]['Day'] 
meas_Anet = data[Treatment==tr][target+'_Mean']
day_incre = np.zeros(len(meas_days), dtype=int)
net_gains = np.zeros((2, len(meas_days)))
for i in range(len(meas_days)):
    day_incre[i] = meas_days.iloc[i] - start_day
    if day_incre[i] > len(net_gain1): 
        break 
    net_gains[0, i] = net_gain1[day_incre[i]]
    net_gains[1, i] = net_gain2[day_incre[i]]

plt.errorbar(meas_days - start_day, meas_Anet, yerr = data[Treatment==tr][target+'_CI'], fmt='o', label=tr)
plt.plot(day_incre, net_gains[0], 'o', label='Instantaneous')
plt.plot(day_incre, net_gains[1], 'x', c=c2, lw=1.5, label='Dynamic feedback')

plt.xlim(0, duration*1.1)
plt.xlabel('Days')
plt.ylabel('Anet (umol s-1 m2)')

plt.tight_layout()
plt.legend()
plt.show()

#%% 
''' PLOTTING ''' 

fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(10, 3))
ax = plt.subplot(131)
plt.plot(ps1, net_gain1, '--', label='Instantaneous')
plt.plot(ps2, net_gain2, c=c2, lw=1.5, label='Dynamic feedback')
plt.plot(ps1, np.zeros(len(ps1)), lw=0.5, c='lightgrey')
plt.xlim(-5,0)
plt.ylabel('Net carbon gain rate \n ($\mathrm{\mu mol\ m^{-2} s^{-1}}$)')
plt.xlabel('Soil water potential (MPa) ')
ax.text(0.85, 0.95, 'a', transform=ax.transAxes, fontsize=16, fontweight='bold', va='top')
plt.legend()

ax = plt.subplot(132)
plt.plot(t_arr, net_gain1, '--')
plt.plot(t_arr, net_gain2, c=c2, lw=1.5)
plt.ylabel( 'Net carbon gain rate \n ($\mathrm{\mu mol\ m^{-2} s^{-1}}$)' )
plt.xlabel('Days')
ax.text(0.85, 0.95, 'b', transform=ax.transAxes, fontsize=16, fontweight='bold', va='top')

ax = plt.subplot(133)
plt.plot(t_arr, s1, '--')
plt.plot(t_arr, s2, c=c2, lw=1.5)
plt.xlabel('Days')
plt.ylabel('Relative soil moisture (-) ')
ax.text(0.85, 0.95, 'c', transform=ax.transAxes, fontsize=16, fontweight='bold', va='top')

plt.tight_layout()
plt.show()

#%% 
''' COMBINE '''

fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(10, 3))

ax = plt.subplot(131)
plt.plot(ps1, net_gain1, '--', label='Instantaneous')
plt.plot(ps2, net_gain2, c=c2, lw=1.5, label='Dynamic feedback')

target = 'Anet'
unit = '$\mathrm{\mu mol\ m^{-2} s^{-1}}$'
x = -data[Treatment==tr]['Ppd_Mean']
y = data[Treatment==tr][target+'_Mean']
plt.errorbar(x, y, yerr = data[Treatment==tr][target+'_CI'], fmt='o', color='lightgrey', zorder=1)  

plt.xlabel('Soil water potential (MPa)')
plt.ylabel('Net carbon gain' + ' ('+ unit+')')
plt.xlim(-3,0)
plt.ylim(-3,40)
ax.text(0.85, 0.95, 'a', transform=ax.transAxes, fontsize=16, fontweight='bold', va='top')
plt.legend()

ax = plt.subplot(132)
plt.plot(t_arr, net_gain1, '--')
plt.plot(t_arr, net_gain2, c=c2, lw=1.5)

target = 'Anet'
unit = '$\mathrm{\mu mol\ m^{-2} s^{-1}}$'
start_day = 171
meas_days = data[Treatment==tr]['Day'] 
meas_Anet = data[Treatment==tr][target+'_Mean']
day_incre = np.zeros(len(meas_days), dtype=int)
net_gains = np.zeros((2, len(meas_days)))
for i in range(len(meas_days)):
    day_incre[i] = meas_days.iloc[i] - start_day
    if day_incre[i] > len(net_gain1): 
        break 
    net_gains[0, i] = net_gain1[day_incre[i]]
    net_gains[1, i] = net_gain2[day_incre[i]]
plt.errorbar(meas_days - start_day, meas_Anet, yerr = data[Treatment==tr][target+'_CI'], fmt='o', label=tr, color='lightgray', zorder=1)
plt.xlim(-1, duration-1)
plt.ylim(-3,40)
plt.ylabel( 'Net carbon gain rate \n ($\mathrm{\mu mol\ m^{-2} s^{-1}}$)' )
plt.xlabel('Days')
ax.text(0.85, 0.95, 'b', transform=ax.transAxes, fontsize=16, fontweight='bold', va='top')

ax = plt.subplot(133)
plt.plot(t_arr, s1, '--', label='Instantaneous')
plt.plot(t_arr, s2, c=c2, lw=1.5, label='Dynamic feedback')
plt.xlabel('Days')
plt.ylabel('Relative soil moisture (-) ')
plt.xlim(-1, duration-1)
plt.ylim(0.05,0.31)
ax.text(0.85, 0.95, 'c', transform=ax.transAxes, fontsize=16, fontweight='bold', va='top')

plt.tight_layout()
plt.show()
