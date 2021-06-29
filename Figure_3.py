#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 24 18:58:46 2021

@author: xuefeng
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from optimization_functions import optimize_gs_multiprocessing_dynfb, net_carbon_multiprocessing_eller
import time 

#%% 
# optimization parameters 
file_name = '../expected_Cnet_eller_vs_dynfb.csv'

# if rerunning optimization from scratch
rerun = True

# range of rainfall frequencies and MAP used for optimization 
freq_arr = np.arange(0.05, 0.46, 0.05) 
MAP_arr = np.arange(500, 2001, 100)

# plotting parameters 
col = 'firebrick'           # color for dynamic feedback optimization 
col2 = '#1f77b4'            # color for instantaneous optimization
n_contours = 5              # number of contours
f1 = 0.1            # rainfall frequency used as test case (thin lines)
f2 = 0.45           # rainfall frequency used as test case (thick lines)

#%%
''' OPTIMIZATION AND SIMULATION '''
if __name__ == '__main__':
    
    if rerun: 
        # set up climate param array for multiprocessing 
        climate_params = [[x, y] for y in MAP_arr for x in freq_arr ]
        
        # # expected Cnet using eller's optimial stomatal conductance 
        # print('Start simulation with Eller instantaneous gs')
        # start = time.time()
        # result_eller = net_carbon_multiprocessing_eller(climate_params)
        # end = time.time()
        # print(result_eller, end-start, 'seconds')
        
        # # write tp csv file
        # df = pd.DataFrame(climate_params, columns=['freq', 'MAP'])
        # df_eller = pd.DataFrame( [r for r in result_eller], columns=['expected_Cnet_eller'] )
        # df_all = df.join(df_eller)
        # df_all.to_csv(file_name)
        
        # for dynamic feedback optimization 
        print('Start dynamic feedback optimization for gs')
        start = time.time()
        result_dynfb = optimize_gs_multiprocessing_dynfb(climate_params)
        end = time.time()
        print(result_dynfb, end-start, 'seconds')
    
        # write tp csv file
        df = pd.read_csv(file_name) 
        df2 = pd.DataFrame([[r['target'], r['params']['a'], r['params']['b'], r['params']['c']] 
                            for r in result_dynfb], columns=['expected_Cnet_dynfb', 'a', 'b', 'c'])
        df_all = df.join(df2)
        df_all.to_csv(file_name)
    
    else: 
        df = pd.read_csv(file_name)  
        freq_arr = np.unique(df.freq)
        MAP_arr = np.unique(df.MAP)
    
        
    #%% 
    ''' EXTRACTING RESULTS '''  
    
    X,Y = np.meshgrid(MAP_arr, freq_arr)
    z = np.zeros((len(freq_arr), len(MAP_arr)))
    for i,f in enumerate(freq_arr):
        for j,m in enumerate(MAP_arr):
            B_opt = df[(df.freq==f) & (df.MAP==m)]['expected_Cnet_dynfb'].values[0]
            B_eller = df[(df.freq==f) & (df.MAP==m)]['expected_Cnet_eller'].values[0]
            z[i,j] = B_opt - B_eller
    
    #%% 
    ''' PLOTTING ''' 
    
    plt.figure(figsize=(9,4))
    ax = plt.subplot(121)
    # eller
    plt.plot(np.sort(df.MAP[df.freq==f1]), np.sort(df.expected_Cnet[df.freq==f1]), '--', c=col, lw=0.5)
    plt.plot(np.sort(df.MAP[df.freq==f2]), np.sort(df.expected_Cnet[df.freq==f2]), '--', c=col, label='Instantaneous')
    # long-term 
    plt.plot(np.sort(df2.MAP[df2.freq==f1]), np.sort(df2.expected_Cnet[df2.freq==f1]), c=col2, lw=0.5)
    plt.plot(np.sort(df2.MAP[df2.freq==f2]), np.sort(df2.expected_Cnet[df2.freq==f2]), c=col2, label='Dynamic feedback')
    plt.xlim(500,2000)
    plt.legend()
    plt.xlabel('Mean annual precipitation (mm)')
    plt.ylabel( 'Net carbon gain rate ($\mathrm{\mu mol\ m^{-2} s^{-1}}$)' )
    ax.text(0.9, 0.95, 'a', transform=ax.transAxes, fontsize=14, fontweight='bold', va='top')
    
    ax = plt.subplot(122)
    CS = ax.contour(X,Y,z, 7, cmap='binary')
    plt.xlabel('Mean annual precipitation (mm)')
    plt.ylabel( 'Rainfall frequency (1/day)')
    plt.title('Difference in net carbon gain rate ($\mathrm{\mu mol\ m^{-2} s^{-1}}$)', fontsize=10)
    plt.clabel(CS)
    ax.text(0.9, 0.95, 'b', transform=ax.transAxes, fontsize=14, fontweight='bold', va='top')
    
    plt.tight_layout()
    plt.show()
