#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 22 14:27:54 2021

@author: Xue Feng, Yaojie Lu

References: 
    Eller et al. (2018), Philosophical Transactions of the Royal Society B, 373.1760: 20170315
    Christoffersen et al. (2016),  Geoscientific Model Development 9.11: 4227-4255
    Clark et al. (2011), Geosci. Model Dev. Discuss 4: 641-688
    Laio et al. (2001), Advances in Water Resources, 24:707-723
    Lu et el. (2020), New Phytologist, 225 (3), 1206-1217
    
"""

import numpy as np
import warnings
warnings.filterwarnings('ignore')
from scipy import optimize
from bayes_opt import BayesianOptimization
import scipy.integrate as integrate
import multiprocessing 

#%% 
''' CONSTANTS & PARAMETERS ''' 

# Stomatal optimization parameters (for dynamic feedback approach)
# Based on prescribed stomatal conductance function a*np.exp(-(ps/c)**b)

a_bounds = (0.02, 1)    # a bounds, mol m-2 s-1
b_bounds = (0.1, 10)    # b bounds
c_bounds = (-1, -0.01)  # c bounds
init_points = 10        # number of initial random explorations
n_iter_single = 100         # number of iterations for single dry down scenario 
n_iter_stochastic = 50      # number of iterations for stochastic rainfall scenario 

# Numerical implementation adjustments

err_abs = 1e-05     # absolute error tolerance for quadrature during integration, default is 1.49e-08
err_rel = err_abs   # relative error tolerance for quadrature during integration, default is 1.49e-08
gs_min = 1e-05      # mol m-2 s-1, minimum cut off for stomatal conductance function
p_min = -1e4        # MPa, lower bound for water potential during minimization

# Conversion constants 

l = 1.8*10**-5      # m3/mol of water
dt = 8*60*60        # sec/day, photosynthetically active period during the day
LAI = 1             # m2 leaf area / m2 ground area, used to scale ET between plant and soil moisture models
diy = 365           # days, number of days in year
Mmm = 0.001         # m/mm, meters in mm

# Soil moisture and water retention parameters
# Defaul parameter values for loamy sand, from Laio et al. (2001), Table 1

n = 0.43            # -, soil porosity
Z = 0.5             # m, effective rooting depth 
pe = -0.17*10**(-3) # MPa, soil water potential near saturation
beta = 4.38         # -, nonlinear parameter in soil water retention curve

# Vulnerability curve parameters

kmax = 0.01         # mol m-2 s-1 MPa-1, maximum hydraulic conductivity
p50= -1             # MPa, plant water potential at 50% loss in hydraulic conductivity
pk = 1              # percent recovery of embolism xylem conductivity, after Lu et al. (2020) - defaults to complete recovery

# Atmospheric constants

a_ratio = 1.6       # water to air ratio
D = 0.01            # mol/mol, VPD
Patm = 101325       # Pa, atmospheric pressure
ca= 40              # ppm, atmospheric CO2 concentration 

# Photosynthesis parameters - default values from Eller et al. (2018), Table 1

IPAR = 0.002        # mol m-2 s-1, incident photosynthetically active radiation
Tc = 20             # C, air temperature
Oa = 21000          # mol mol-1, air oxygen concentration
Vcmax25 = 0.0001    # mol m-2 s-1, maximum Rubisco carboxylation rate at 25C
Tup = 50            # C, high temperature photosynthesis range
Tlw = 10            # C, low temperature photosynthesis range
alfa = 0.1          # mol mol-1, quantum efficiency
wPAR = 0.15         # -, leaf scattering coefficient
Tref = 25           # C, reference leaf temperature 

# Photosynthesis parameters - default values from Clark et al. (2011) and Eller et al. (2018)

Q10leaf = 2         # Q10 factor for leaf 
Q10rs = 0.57        # Q10 factor for CO2 compensation point (Gamma)
Q10Kc = 2.1         # Q10 factor for Michaelis-Menton constant for CO2
Q10Ko = 1.2         # Q10 factor for Michaelis-Menton constant for O2

#%%
''' HELPER FUNCTIONS (PHOTOSYNTHESIS, EVAPOTRANSPIRATION, NET CARBON GAIN) ''' 

# Soil water potentials 

def psf(s):
    # soil water potential, MPa
    # soil water retention curve: s to soil water potential 
    
    return pe*s**(-beta)

def sf(ps):
    # relative soil moisture, - 
    # soil water retention curve: soil water potential to s 
    
    return (ps/pe)**(-1/beta)

# Plant water potentials

def PLCf(p, p50=p50):
    # native percent loss in conductivity
    
    return 1-kf(p, p50)/kmax
 
def kf(p, p50=p50, pL=0, pk=1): 
    # Xylem hydraulic conductivity, mol m-2 s-1 MPa-1
    # relationship between a and P50 based on Christoffersen et al. (2016)
    # pL, pk argument is only to keep consistent formatting with kfm function, never used 
    
    slope = 65.15*(-p50)**(-1.25) # slope of xylem vulnerability curve at P50
    a = -4*slope/100*p50
    return kmax/(1+(p/p50)**a)

def kfm(p, p50=p50, pL=0, pk=1): 
    # modified xylem hydraulic conductivity function to account for xylem embolism and recovery
    # pL is the lowest water potential yet experienced. 
    # if there is complete recovery, then return kf(p) always - set pL to high value (e.g., 0 or ps)
    
    if p>pL: # recovery occurs (when pk=1, PLCfm = PLCf(p); when pk=0, PLCfm = PLCf(pL) )
        PLCfm = PLCf(pL,p50)- pk*( PLCf(pL,p50)-PLCf(p, p50) )
    else:  
        PLCfm = PLCf(p,p50)
    return kmax*(1 - PLCfm)

# Photosynthesis and assimilation 

def Acif(ci): 
    # assimilation, mol m-2 s-1
    # Collatz et al (1991) C3 photosynthesis model, following Clark et al (2011)
    
    # Vcmax temperature response (mol m-2 s-1)
    Vcmax = Vcmax25*(Q10leaf**(0.1*(Tc-Tref)))/((1+np.exp(0.3*(Tc-Tup)))*\
                                              (1+np.exp(0.3*(Tlw-Tc))))
    
    # Photocompensation point (Pa)
    photocomp = Oa/(2*2600*(Q10rs**(0.1*(Tc-Tref))))
    
    # Kc & Ko - Michaelis-Menton constants for CO2 and O2
    Kc = 30*(Q10Kc**(0.1*(Tc-Tref)))
    Ko = 30000*(Q10Ko**(0.1*(Tc-Tref)))
    
    # Assimilation
    Ac = Vcmax*((ci-photocomp)/(ci+Kc*(1+Oa/Ko)))
    Al = alfa*(1-wPAR)*IPAR*((ci-photocomp)/(ci+2*photocomp))
    Ae = 0.5*Vcmax

    return min(Ac, Al, Ae)

def Agcf(gc, ci):
    # assimilation, mol m-2 s-1
    
    return gc*(ca-ci)/Patm

def Af(gc):
    # find equilibrium internal CO2 concentration to match assimilation (Acif and Agcf)
    
    f1 = lambda ci: Acif(ci)-Agcf(gc, ci)
    ci = optimize.brentq(f1, 0, ca)                                     
    return Acif(ci)

# Evapotranspiration 

def Egcf(gc):
    # Evapotranspiration (demand), mol m-2 leaf s-1
    
    return a_ratio*gc*D

def Epf(p, ps, p50=p50, pL=0, pk=1, k_func=kf): # mol m-2 leaf s-1
    # Evapotranspiration (supply), mol m-2 leaf s-1
    # ps: soil water potential
    # p: plant water potential
    # k_func: generic conductivity function (kf or kfm); if kfm is picked, an additional pL input is required
    # pL: minimum plant water potential yet experienced 
    # pk: percent of recovery 
    
    return (ps-p) * k_func((ps+p)/2, p50, pL, pk)

# Whole plant carbon-water integration 

def pminf(ps, p50=p50, pL=0, pk=1, k_func=kf):
    # find plant water potential that yields maximum water supply 
    # pL, pk, and k_func passed onto ET supply function 
    
    f1 = lambda p: -Epf(p, ps, p50, pL, pk, k_func)
    p = optimize.minimize_scalar(f1, bounds=(p_min, ps), method='bounded').x 
    return p

def gcmaxf(ps, p50=p50, pL=0, pk=1, k_func=kf):
    # find maximum stomatal conductance at given soil water potential (ps) 
    # and pL (if considering permaneng xylem embolism)
    # pL, pk, and k_func passed onto pminf and Epf functions
    
    p = pminf(ps, p50, pL, pk, k_func)
    gc = Epf(p, ps, p50, pL, pk, k_func) / (a_ratio*D)
    return gc

def pf(gc, ps, p50=p50, pL=0, pk=1, k_func=kf):
    # find water potential that equilibriates plant water supply and demand (Egsf and Epf)
    # brentq is a root finding function -- the solution is bracketed between pmin and ps
    # calculation of pmin is necessary to find the "critical" water potential. 
    # pL, pk, and k_func passed onto gcmaxf, pminf, Epf
    
    pmin = pminf(ps, p50, pL, pk, k_func)
    if gc == gcmaxf(ps, p50, pL, pk, k_func):
        return pmin
    elif pmin < ps: 
        f1 = lambda p: Epf(p, ps, p50, pL, pk, k_func) - Egcf(gc)
        try: 
            p = optimize.brentq(f1, pmin, ps)
        except ValueError: # demand outstrips supply 
            p = pmin
        return p
    else: 
        return ps

def net_gainf(gc, ps, p50=p50, pL=0, pk=1, k_func=kf):
    # net carbon gain function based on equilibrated water potential (ps) -- needed by cost function 
    # and (gc) -- needed by the assimilation function
    # pL, pk, and k_func passed to pf to calculate cost 
    
    p = pf(gc, ps, p50, pL, pk, k_func) 
    k_cost = k_func((p+ps)/2, p50, pL, pk)/kmax
    A = Af(gc)
    return A*k_cost


def eller(ps, p50=p50, pL=0, pk=1, k_func=kf):
    # find optimal stomatal conductance with highest net gain, based on Eller et al. (2018)
    # pL, pk, and k_func passed onto gcmaxf, net_gainf
    # ps assumed to be greater than pL
    
    gcmax = gcmaxf(ps, p50, pL, pk, k_func)
    # if ps <= pL:
    #     return 0
    if gcmax > 0:
        f1 = lambda gc: -net_gainf(gc, ps, p50, pL, pk, k_func) 
        gc = optimize.minimize_scalar(f1, bounds=(0, gcmax), method='bounded').x
        return gc
    return 0


#%% 
''' SIMULATION & OPTIMIZATION FUNCTIONS - SINGLE DRY DOWN SCENARIO''' 

def simf_full_refill(gs_func, s0, duration, output_option, p50=p50, pk=pk):
    # simulates trajectories of soil moisture, stomatal conductance, net carbon gain, and ET over time
    # gs_func is the functional form of stomatal conductance with respect to soil water potential 
    # gs_func accepts either "eller" or a pre-defined function
    # output_options are 'Cnet' (for objective function) or 'All' (for simulations)
    # p50 and pk are dummy inputs - default values set at beginning

    s = np.zeros(duration+1)        # relative soil moisture
    net_gain = np.zeros(duration)   # net carbon gain
    ps = np.zeros(duration)         # soil water potential
    gc = np.zeros(duration)         # stomatal conductance
    E = np.zeros(duration)          # transpiration 
    
    # initial condition
    s[0] = s0
    
    for i in range(duration):
        ps[i] = psf(s[i])
        gc[i] = gs_func(ps[i])
        net_gain[i] = net_gainf(gc[i], ps[i]) 
        E[i] = Egcf(gc[i])*dt*l*LAI / (n*Z)      
        s[i+1] = max(0, s[i]-E[i])
        
    if output_option=='Cnet': # Cumulative net carbon gain, mol m-2 s-1
        output = sum(net_gain)              
    if output_option=='All':  # Trajectories over time, mol m-2 s-1
        output = s[:duration], ps, gc, net_gain, E    
    return output

def simf_partial_refill(gs_func, s0, duration, p50, pk, output_option): 
    # output_options are 'Cnet' (for objective function) or 'All' (for simulations)
    # in simulation pL is the lowest p experienced by plants 
    # for lt, sL is the prescribed tolerance p
    # gs_func can take in eller or pre-defined stomatal function
    # gs_func needs ps, pL, pk, kfm as input arguments 
    
    # initialize variables 
    s = np.zeros(duration+1)            # relative soil moisture
    p = np.zeros(duration)              # plant water potential
    net_gain = np.zeros(duration)       # net carbon gain
    gc = np.zeros(duration)             # stomatal conductance
    E = np.zeros(duration)              # transpiration
    # additional variables for simulating trajectories
    PLC = np.zeros(duration)            # percent loss of conductivity
    pL_t = np.zeros(duration+1)           # minimum plant water potential
    
    # initial conditions
    s[0] = s0
    pL = 0
    
    for i in range(duration):
        ps = psf(s[i])
        pL = pL_t[i]
        
        # resolve internal states
        gc[i] = gs_func(ps, p50=p50, pL=0, pk=1, k_func=kf) 
        net_gain[i] = net_gainf(gc[i], ps, p50, pL, pk, kfm)
        
        # resolve soil water balance 
        E[i] = Egcf(gc[i])*dt*l*LAI / (n*Z)
        s[i+1] = max(0, s[i]-E[i])
        
        # additional for simulating embolism trajectories (not actually needed for Cnet)
        p[i] = pf(gc[i], ps, p50, pL, pk, kfm)
        pL_t[i+1] = min(pL, (p[i]+ps)/2)
        PLC[i] = PLCf(pL_t[i])
        
    if output_option=='Cnet': # Cumulative net carbon gain, mol m-2 s-1
        output = sum(net_gain)          
    if output_option=='All':  # Trajectories over time, mol m-2 s-1
        output = net_gain, p, PLC, s[:duration], pL_t[:duration] 
    return output

def optimize_gs_drydown(sim_func, s0, duration, pk=pk, p50=p50, 
             a_bounds=a_bounds, b_bounds=b_bounds, c_bounds=c_bounds, 
             init_points=init_points, n_ter=n_iter_single):
    # bayesian optimization that looks for the optimized parameters 
    # for the stomatal conductance function a*np.exp(-(ps/c)**b)
    # that yields the maximum net carbon gain over a set initial condition s0 and duration. 
    # sim_func: (simf_embolism or simf_no_embolism) the trajectory used for optimization
    # pk and p50 added during cases with partial recovery 
    
    # objective function 
    def objf(a, b, c):
        # prescribed stomatal conductance function 
        # p50, pL, pk, k_func are dummy variables to keep consistency with Eller's gs function
        gcpsf = lambda ps, p50, pL, pk, k_func: min(a*np.exp(-(ps/c)**b)-gs_min, gcmaxf(ps))
        res = sim_func(gcpsf, s0=s0, duration=duration, p50=p50, pk=pk, output_option='Cnet')
        return res

    # parameter space
    pbounds = {'a': a_bounds, 'b': b_bounds, 'c': c_bounds}
    
    # bayesian optimization
    optimizer = BayesianOptimization(
        f=objf,
        pbounds=pbounds,
        verbose=1,
        random_state=1
        )
    
    # reiterate
    optimizer.maximize(
        init_points=init_points,
        n_iter=n_iter_single,
        )
    
    # return dictionary with keys "target" and "params" 
    return optimizer.max

#%%
''' SIMULATION & OPTIMIZATION FUNCTIONS - STOCHASTIC RAINFALL SCENARIO 
    "dynfb": dynamic feedback approach
    "eller": instantaneous approach based on Eller et al. (2018)
    ''' 

def soil_moisture_pdf(gs_func, gamma, freq, s_min): 
    # define pdf, now knowing stomatal conductance function and rainfall conditions 
    # gamma (-):  rooting depth / mean rainfall depth 
    # freq (day-1): mean rainfall frequency  
    # soil moisture pdf after Rodriguez-Iturbe et al. (1999)
    
    rLf = lambda s: 1/ (Egcf(gs_func(psf(s)))*l*dt*LAI/n/Z)
    integralf = lambda s: integrate.quad(rLf, s, 1, epsabs=err_abs, epsrel=err_rel)[0] 
    pdf_no_c = lambda s: rLf(s)*np.exp(-gamma*s-freq*integralf(s))
    cinverse = integrate.quad(pdf_no_c, s_min, 1, epsabs=err_abs, epsrel=err_rel)[0]   
    pdf = lambda s: pdf_no_c(s) / cinverse
    return pdf
    
def expected_gain(pdf, gs_func, s_min): 
    # calculate net carbon gain based on carbon cost & benefit function
    # gs_func a placeholder for either a prescribed functional form, or the eller's optimal function 
    # expected gain in units of mol m-2 s-1
    
    expected_gain_integrand = lambda s: net_gainf(gs_func(psf(s)), psf(s))*pdf(s)
    res = integrate.quad(expected_gain_integrand, s_min, 1, epsabs=err_abs, epsrel=err_rel)[0]
    return res 

def objective_function_dynfb(a, b, c, gamma, freq):
    
    # find lower bounds for ps and s
    ps_min = c*((-np.log(gs_min/a))**(1/b)) # minimum p for getting 1% of max gc
    s_min=sf(ps_min)
    
    # define stomatal conductance function  
    gcpsf = lambda ps: min(a*np.exp(-(ps/c)**b)-gs_min, gcmaxf(ps))
    
    # define pdf, now knowing stomatal conductance function
    pdf = soil_moisture_pdf(gcpsf, gamma, freq, s_min)
    
    # find expected carbon gain given stomatal conductance and soil moisture pdf
    res = expected_gain(pdf, gcpsf, s_min)
    if np.isnan(res): res = 0
    return res

def objective_function_eller(climate_param):
    # directly calculates expected net carbon gain 
    # based on eller's optimal stomatal conductance function 
    
    # convert climate parameters to soil moisture pdf parameters (freq, gamma)
    freq, MAP = climate_param;
    gamma = (n*Z) / (MAP*Mmm / (freq*diy))
    print(freq, MAP, gamma)
    
    # find lower bound for soil moisture 
    s_min=sf(p_min)
    
    # define pdf using Eller's optimal stomatal conductance function
    pdf = soil_moisture_pdf(eller, gamma, freq, s_min)
    
     # find expected carbon gain given stomatal conductance and soil moisture pdf
    res = expected_gain(pdf, eller, s_min)
    if np.isnan(res): res = 0
    return res

def optimize_gs_stochastic(climate_param):
    # find correponding a,b,c parameters for the gs function given rainfall frequency and MAP
    
    # convert climate parameters to soil moisture pdf parameters (freq, gamma)
    freq, MAP = climate_param; 
    gamma = (n*Z) / (MAP*Mmm / (freq*diy))
    print(freq, MAP, gamma)
    
    # define objective function 
    objf_wrapper = lambda a,b,c: objective_function_dynfb(a,b,c, gamma, freq)
    
    # parameter space
    pbounds = {'a': a_bounds, 'b': b_bounds, 'c': c_bounds}
    
    # bayesian optimization
    optimizer = BayesianOptimization(
        f=objf_wrapper,
        pbounds=pbounds,
        verbose=0,
        random_state=1,
        )
    
    # # reiterate
    optimizer.maximize(
        init_points=init_points,
        n_iter=n_iter_stochastic,
        )
    
    return optimizer.max

def optimize_gs_multiprocessing_dynfb(climate_params):
    # climate_params takes in a tuple of (freq, MAP)
    # freq (day-1): mean rainfall frequency 
    # MAP (mm): mean annual precipitation 
    
    pool = multiprocessing.Pool()
    results = pool.map(optimize_gs_stochastic, climate_params)
    return results

def net_carbon_multiprocessing_eller(climate_params): 
    # climate_params takes in a tuple of (freq, MAP)
    # freq (day-1): mean rainfall frequency 
    # MAP (mm): mean annual precipitation 
    
    pool = multiprocessing.Pool()
    # eller function is already prescribed -- no need to "optimize"
    # i.e., eller's objective function already gives expected net carbon gain 
    results = pool.map(objective_function_eller, climate_params)
    return results 



