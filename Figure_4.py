import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from optimization_functions import gs_min, eller, gcmaxf, simf_partial_refill, optimize_gs_drydown

#%% 

# if rerunning optimization from scratch
rerun = True
column_names = ['net_gain', 'PLC', 's', 'pl']

# optimization parameters 
s0 = 0.5            # initial soil moisture
duration = 100      # days, dry down duration 
pk = 0.5            # percent recovery 
p50_1 = -1            # plant water potential at 50% loss of max conductivity 
p50_2 = -2            # plant water potential at 50% loss of max conductivity 

# plotting parameters 
col = 'firebrick'           # color for dynamic feedback optimization 
col2 = '#1f77b4'            # color for instantaneous optimization
mumol = 10**6               # umol/mol, converstion factor

#%% 
''' SIMULATIONS '''
if rerun: 
    
    def eller_save(p50): 
    # dry down using eller's instantaneous optimal gs function 
        res = simf_partial_refill(eller, s0=s0, duration=duration, p50=p50, pk=pk, output_option='All')
        df = pd.DataFrame(data=np.array([res[0], res[2], res[3], res[4]]).T, columns=column_names)
        df.to_csv(str(p50)+'eller.csv')
        return df
    
    def dynfb_save(p50):
        # dynamic feedback optimization 
        a,b,c= optimize_gs_drydown(simf_partial_refill, s0=s0, duration=duration, p50=p50, pk=pk)['params'].values()
        # using results to redefine stomatal conductance function 
        gs_dynfb = lambda ps, p50, pL, pk, k_func: min(a*np.exp(-(ps/c)**b)-gs_min, gcmaxf(ps))
        res = simf_partial_refill(gs_dynfb, s0=s0, duration=duration, p50=p50, pk=pk, output_option='All')
        
        # saving to dataframe
        df = pd.DataFrame(data=np.array([res[0], res[2], res[3], res[4]]).T, columns=column_names)
        df.to_csv(str(p50)+'dynfb.csv')
        return df

    df1e = eller_save(p50_1)
    df2e = eller_save(p50_2)
    df1d = dynfb_save(p50_1)
    df2d = dynfb_save(p50_2)
    
else: 
    # import from existing files 
    df1e = pd.read_csv(str(p50_1)+'eller.csv', names=column_names, skiprows=1)
    df2e = pd.read_csv(str(p50_2)+'eller.csv', names=column_names, skiprows=1)
    df1d = pd.read_csv(str(p50_1)+'dynfb.csv', names=column_names, skiprows=1)
    df2d = pd.read_csv(str(p50_2)+'dynfb.csv', names=column_names, skiprows=1)

#%% 
''' PLOTTING '''

fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(10, 3))
ax = plt.subplot(131)
var = 'net_gain'
plt.plot(df1e[var] * mumol, '--', c=col2, lw=0.5)
plt.plot(df2e[var] * mumol, '--', c=col2)
plt.plot(df1d[var] * mumol, c=col, lw=0.5)
plt.plot(df2d[var] * mumol, c=col)
plt.ylabel( 'Net carbon gain rate \n ($\mathrm{\mu mol\ m^{-2} s^{-1}}$)' )
plt.xlabel('Days')
ax.text(0.85, 0.95, 'a', transform=ax.transAxes,
      fontsize=16, fontweight='bold', va='top')

ax = plt.subplot(132)
var = 's'
plt.plot(df1e[var], '--', c=col2, lw=0.5)
plt.plot(df2e[var], '--', c=col2, label='Instantaneous')
plt.plot(df1d[var], c=col, lw=0.5)
plt.plot(df2d[var], c=col, label='Dynamic feedback')
plt.legend()
plt.xlabel('Days')
plt.ylabel('Relative soil moisture (-) ')
ax.text(0.85, 0.95, 'b', transform=ax.transAxes,
      fontsize=16, fontweight='bold', va='top')

ax = plt.subplot(133)
var = 'PLC'
plt.plot(df1e[var], '--', c=col2, lw=0.5)
plt.plot(df2e[var], '--', c=col2, label='Instantaneous')
plt.plot(df1d[var], c=col, lw=0.5)
plt.plot(df2d[var], c=col, label='Dynamic feedback')
plt.xlabel('Days')
plt.ylabel('Percent loss in conductivity (%)')
ax.text(0.05, 0.95, 'c', transform=ax.transAxes,
      fontsize=16, fontweight='bold', va='top')

plt.tight_layout()
plt.savefig('./Figures/Figure_4.png', dpi=300)
plt.show()
