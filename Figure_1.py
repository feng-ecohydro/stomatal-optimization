from optimization_functions import simf_full_refill, optimize_gs_drydown, eller, gcmaxf, gs_min
import numpy as np
import matplotlib.pyplot as plt

#%% 

# Simulation parameters 
s0 = 0.3            # -, initial relative soil moisture
duration = 100      # days, duration of simulation
mumol = 10**6       # umol/mol, converstion factor

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
gs_dynfb = lambda ps: min(a*np.exp(-(ps/c)**b)-gs_min, gcmaxf(ps))
s2, ps2, gc2, net_gain2, E2 = simf_full_refill(gs_dynfb, s0=s0, duration=duration, output_option='All')

#%% 
''' PLOTTING ''' 

# Figure setup
net_gain1 = net_gain1*mumol    # convert net carbon gain rate to umol m-2 s-1
net_gain2 = net_gain2*mumol    # net_gainf(gs_dynfb(p_arr)) * mumol

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
plt.savefig('./Figures/Figure_1.png', dpi=300)
plt.show()

 
