#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 15:04:29 2023

@author: afzal-admin
RetMIP paper Vandecrux et al. (2020) for Dye2 in 2016 from Samira data from 08/09/2016
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize
plt.rcParams.update({'font.size': 20})
plt.rcParams.update({'font.family': 'Serif'}) 
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
import pandas as pd
from datetime import datetime
from colliander_data_analysis import spun_up_profile_May2016_Colliander, Qnet_May2Sept2016_Samira

#Colors
brown  = [181/255 , 101/255, 29/255]
red    = [190/255 , 30/255 , 45/255]    
blue   = [ 30/255 ,144/255 , 255/255]
green  = [  0/255 , 166/255 ,  81/255]
orange = [247/255 , 148/255 ,  30/255]
purple = [102/255 ,  45/255 , 145/255]
brown  = [155/255 ,  118/255 ,  83/255]
tan    = [199/255 , 178/255 , 153/255]
gray   = [100/255 , 100/255 , 100/255]


#Importing results from RetMIP paper, 2020
RetMIP_init_rho = np.loadtxt('./Samira-data/RetMIP_Forcing_data_Dye2_16/RetMIP_initial_firn_density_Dye-2_16.tab',delimiter=';',skiprows=1)
RetMIP_init_depth = RetMIP_init_rho[:,0]
RetMIP_init_phi = 1 - RetMIP_init_rho[:,1]/917
RetMIP_init_lwc = np.loadtxt('./Samira-data/RetMIP_Forcing_data_Dye2_16/RetMIP_initial_firn_lwc_Dye-2_16.tab',delimiter=';',skiprows=1)
RetMIP_init_T = np.loadtxt('./Samira-data/RetMIP_Forcing_data_Dye2_16/RetMIP_initial_firn_temperature_Dye-2_16.tab',delimiter=';',skiprows=1)
#RetMIP_surf_q = np.loadtxt('./Samira-data/RetMIP_Forcing_data_Dye2_16/RetMIP_surface_forcing_Dye-2_16.tab',delimiter=';',skiprows=1)

df = pd.read_csv('./Samira-data/RetMIP_Forcing_data_Dye2_16/RetMIP_surface_forcing_Dye-2_16.tab',sep=';',skiprows=(0),header=(0))
RetMIP_dates = np.array(df['time'])
RetMIP_melt_mmweq     = np.array(df['melt_mmweq'])
RetMIP_Tsurf_K        = np.array(df['Tsurf_K'])    
RetMIP_acc_subl_mmweq = np.array(df['acc_subl_mmweq'])

#Original to relative time
abs_time = np.empty_like(RetMIP_dates)
date_time_origin = datetime.strptime(RetMIP_dates[0], '%d-%b-%Y %H:%M:%S')

for i in range(0,len(abs_time)):
    date_dummy = datetime.strptime(RetMIP_dates[i], '%d-%b-%Y %H:%M:%S')
    abs_time[i]  = (date_dummy - date_time_origin).total_seconds()

RetMIP_abs_time  = (np.array(abs_time)).astype(float) #time in seconds 

RetMIP_fit_porosity = interp1d(RetMIP_init_depth,RetMIP_init_phi, kind='cubic', fill_value='extrapolate')
RetMIP_fit_depth    = np.linspace(RetMIP_init_depth[0],RetMIP_init_depth[-1],1000)   
RetMIP_fit_temp     = interp1d(RetMIP_init_depth,RetMIP_init_T[:,1], kind='cubic', fill_value='extrapolate') 


#Flux computation

h  =14.8   #surface-air heat exchange coeff [Wm−2 K−1] (from Meyer and Hewitt, 2017; Cuffey and Paterson, 2010; van den Broeke et al., 2011)
Tm =273.16 #Melting temp in [K]
L  = 334000#Latent heat of fusion [m2 s−2]
rho_w = 1e3#denisty of water
day2s = 60*60*24
yr2s  = day2s*365.25

Q_t  = (np.array(h*(RetMIP_Tsurf_K- Tm))).astype(float)  #Turbulent flux [W/m^2]
Q_l  =  (rho_w * L * RetMIP_melt_mmweq * 1e-3 /(3 * 60 * 60)).astype(float)  #Latent flux [W/m^2] 
Q    = Q_t + Q_l #Net flux [W/m^2]
Q_f  = (rho_w * RetMIP_melt_mmweq * 1e-3 /(3 * 60 * 60)).astype(float)    #Rate of meltwater flux [kg m^-2 s-1]
acc  = RetMIP_acc_subl_mmweq.astype(float)  #Accumulation [mm w eq] in 3 hours
acc[acc  <0]  = 0  #Zeroing out negative accumulation
acc_rate = acc*1e-3 / (3 * 60 * 60)  #Accumulation rate [m water eq per second]


RetMIP_Q  = interp1d(RetMIP_abs_time/day2s, Q, kind='cubic', fill_value='interpolate', bounds_error=False)   #Time is in day flux in W/m^2
RetMIP_Qf = interp1d(RetMIP_abs_time/day2s, Q_f, kind='cubic', fill_value='interpolate', bounds_error=False)
RetMIP_acc_rate= interp1d(RetMIP_abs_time/day2s, acc_rate, kind='cubic', fill_value='extrapolate', bounds_error=False)

fig = plt.figure(figsize=(15,7.5) , dpi=100)
plot = plt.plot(RetMIP_abs_time/day2s, Q_t,'r-', label='Turbulent')
plot = plt.plot(RetMIP_abs_time/day2s, Q_l,'b-', label='Latent')
plt.plot(np.linspace(0,179,1790),RetMIP_Q(np.linspace(0,179,1790)),'k--')
plot = plt.plot(RetMIP_abs_time/day2s,  Q,'ko--', label='Total')
plt.legend()
plt.xlabel(r'$t$ [days]')
plt.ylabel(r'Q [W/m$^2$]')
#plt.clim(0.000000, 1.0000000)
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
plt.savefig(f'../Figures/Q_combined.pdf',bbox_inches='tight', dpi = 600)


fig = plt.figure(figsize=(15,7.5) , dpi=100)
plot = plt.plot(RetMIP_abs_time/day2s, Q_f,'b-')
plt.xlabel(r'$t$ [days]')
plt.ylabel(r'$Q_f [kg m^{-2} s^{-1}]$')
#plt.clim(0.000000, 1.0000000)
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
plt.savefig(f'../Figures/Qc_combined.pdf',bbox_inches='tight', dpi = 600)


##############################################################################################################################
#Samira's paper 2021
#Site A
Samira_SiteA_depth       = np.array([0.3, 0.6, 0.9, 1.4, 1.8, 2.1, 2.8, 3.7])
Samira_SiteA_init_T      = np.array([-7.3, -10.1, -12.7, -14.5, -15.1, -15.6, -15.7, -15.8])
Samira_SiteA_init_rho    = np.array([380, 380, 230, 510, 410, 460, 360, 520])
Samira_SiteA_init_phi    = 1-Samira_SiteA_init_rho/917  

Samira_SiteA_fit_porosity = interp1d(Samira_SiteA_depth,Samira_SiteA_init_phi, kind='cubic', fill_value='extrapolate')
Samira_SiteA_fit_depth    = np.linspace(Samira_SiteA_depth[0],Samira_SiteA_depth[-1],1000)   
Samira_SiteA_fit_temp     = interp1d(Samira_SiteA_depth,Samira_SiteA_init_T, kind='cubic', fill_value='extrapolate') 

#Site B
Samira_SiteB_depth       = np.array([0.1, 0.2, 0.4, 0.6, 0.9, 1.2, 1.4, 1.6])
Samira_SiteB_init_T      = np.array([-3.7, -4.2, -6.6, -9.2, -11.9, -13.1, -14.4, -14.9])
Samira_SiteB_init_rho    = np.array([280, 310, 280, 320, 640, 620, 560, 480])
Samira_SiteB_init_phi    = 1-Samira_SiteB_init_rho/917 

Samira_SiteB_fit_porosity = interp1d(Samira_SiteB_depth,Samira_SiteB_init_phi, kind='cubic', fill_value='extrapolate')
Samira_SiteB_fit_depth    = np.linspace(Samira_SiteB_depth[0],Samira_SiteB_depth[-1],1000)   
Samira_SiteB_fit_temp     = interp1d(Samira_SiteB_depth,Samira_SiteB_init_T, kind='cubic', fill_value='extrapolate') 



############################################################################################
#Samira sensor data at Site A right at the beginning

df = pd.read_excel('./Samira-data/TDRA_MaytoSept2016.xlsx',header=(1))
Samira_dates = df['Time']
data  = pd.read_csv('/Users/afzal-admin/Documents/Research/meltwater-percolation/Infiltration/JPL/Samira-data/Achilig_upGPR_data.csv')
dates = data['date'].apply(lambda x: datetime.strptime(x, "'%d-%b-%Y %H:%M:%S.%f'")) 
Samira_dates_index = np.where((Samira_dates>=dates[850]) & (Samira_dates<=dates[len(dates)-1]))
Samira_data_Temp_actual  =  np.array([np.array((df['degC.1'])[Samira_dates_index[0]])[0], np.array((df['degC.2'])[Samira_dates_index[0]])[0], \
                            np.array((df['degC.3'])[Samira_dates_index[0]])[0], np.array((df['degC.4'])[Samira_dates_index[0]])[0], \
                            np.array((df['degC.5'])[Samira_dates_index[0]])[0], np.array((df['degC.6'])[Samira_dates_index[0]])[0], \
                            np.array((df['degC.7'])[Samira_dates_index[0]])[0], np.array((df['degC.8'])[Samira_dates_index[0]])[0]])

Samira_data_depth_actual = np.array([0.3, 0.6, 0.9, 1.4, 1.8, 2.1, 2.8, 3.7])
Samira_data_actual_fit_temp= interp1d(Samira_data_depth_actual,Samira_data_Temp_actual, kind='cubic', fill_value='extrapolate') 

##############################################################################################################################
#Colliander unpublished
data     = np.genfromtxt('./Colliander_data/density_profile_greenland_DYE2_python.csv', delimiter=',')
Coll_depth    = data[:,0]  #Depth in m
Coll_depth[0] = 0.1
Coll_temp     = data[:,1]  #Temperature in deg C
Coll_rho_firn = data[:,2]  #Density of firn including ice kg/m^3
Coll_rho_bulk = data[:,3]  #Density including ice kg/m^3
Coll_ice      = data[:,4]  #Ice in m
Coll_porosity = 1-Coll_rho_bulk/917

Coll_fit_porosity = interp1d(Coll_depth,Coll_porosity, kind='cubic', fill_value='extrapolate')
Coll_fit_depth    = np.linspace(Coll_depth[0],Coll_depth[-1],1000)   
Coll_fit_temp     = interp1d(Coll_depth[0:],Coll_temp[0:], kind='cubic', fill_value='extrapolate') 

day,Coll_Qnet,Coll_Qnet_func             = Qnet_May2Sept2016_Samira()


##############################################################################################################################
#Rennermalm (2021)
dR = np.loadtxt('./Rennermalm/Dye2_2016.csv',skiprows=1,delimiter=',')

#Plotting 

fig, (ax1, ax2) = plt.subplots(1,2, sharey=True,figsize=(10,10))
plt.subplot(1,2,1)
#plt.plot(Coll_porosity,Coll_depth,'kx',label=r'Colliander$^+$ (2022)',markersize=10, markerfacecolor = 'none')
#plt.plot(Coll_fit_porosity(Coll_fit_depth),Coll_fit_depth,'k--',markersize=10)
plt.plot(RetMIP_init_phi,RetMIP_init_depth,'ro',label=r'Vandecrux$^+$ (2020)',markersize=10, markerfacecolor = 'none')
plt.plot(RetMIP_fit_porosity(RetMIP_fit_depth),RetMIP_fit_depth,'r--',markersize=10,color=red)
plt.plot(Samira_SiteA_init_phi,Samira_SiteA_depth,'bs',label=r'Samimi$^+$ (2021) Site-A',markersize=10, markerfacecolor = 'none')
#plt.plot(Samira_SiteA_fit_porosity(Samira_SiteA_fit_depth),Samira_SiteA_fit_depth,'b--',markersize=10)

plt.plot(Samira_SiteB_init_phi,Samira_SiteB_depth,'gd',label=r'Samimi$^+$ (2021) Site-B',markersize=10, markerfacecolor = 'none')
#plt.plot(Samira_SiteB_fit_porosity(Samira_SiteB_fit_depth),Samira_SiteB_fit_depth,'g--',markersize=10)
plt.plot(1-dR[:,5]/917,(dR[:,0]+dR[:,1])/2,'g.',label=r'Rennermalm$^+$ (2022)',markersize=10)

plt.xlabel(r'$\phi$')
plt.ylabel(r'Depth [m]')
#plt.ylim([20,0])
plt.tight_layout()
plt.legend(framealpha=0.5,loc=(1.04, 0))
plt.subplot(1,2,2)
plt.plot(Coll_temp,Coll_depth,'kx',label=r'Colliander$^+$ (2022)',markersize=10, markerfacecolor = 'none')
#plt.plot(Coll_fit_temp(Coll_fit_depth),Coll_fit_depth,'k--',markersize=10)
plt.plot(RetMIP_init_T[:,1],RetMIP_init_depth,'ro',label=r'Vandecrux$^+$ (2020)',markersize=10, markerfacecolor = 'none')
plt.plot(RetMIP_fit_temp(RetMIP_fit_depth),RetMIP_fit_depth,'r--',markersize=10)
plt.plot(Samira_SiteA_init_T,Samira_SiteA_depth,'bs',label=r'Samimi$^+$ (2021) Site-A',markersize=10, markerfacecolor = 'none')
#plt.plot(Samira_SiteA_fit_temp(Samira_SiteA_fit_depth),Samira_SiteA_fit_depth,'b--',markersize=10)

plt.plot(Samira_SiteB_init_T,Samira_SiteB_depth,'gd',label=r'Samimi$^+$ (2021) Site-B',markersize=10, markerfacecolor = 'none')
#plt.plot(Samira_SiteB_fit_temp(Samira_SiteB_fit_depth),Samira_SiteB_fit_depth,'g--',markersize=10)

#plt.plot(Samira_data_Temp_actual,Samira_data_depth_actual,'c^',label=r'Samimi$^+$ (2021) Site-A D',markersize=10, markerfacecolor = 'none')
#plt.plot(Samira_data_actual_fit_temp(Samira_SiteA_fit_depth),Samira_SiteA_fit_depth,'c--',markersize=10)

plt.xlabel(r'Temperature $[\circ C]$')
plt.ylim([5,0])
plt.tight_layout()
plt.savefig(f'../Figures/Combined_analysis_porosity_Temp.pdf',bbox_inches='tight', dpi = 600)


#Plotting 

fig, (ax1) = plt.subplots(1,1, sharey=True,figsize=(10,10))
plt.subplot(1,1,1)

# Create a Rectangle patch
import matplotlib.patches as patches
rect = patches.Rectangle((0.05, 0), 0.1,5, linewidth=1, edgecolor='none', facecolor=gray,alpha=0.2)

# Add the patch to the Axes
ax1.add_patch(rect)

#plt.plot(Coll_porosity,Coll_depth,'kx',label=r'Colliander$^+$ (2022)',markersize=10, markerfacecolor = 'none')
#plt.plot(Coll_fit_porosity(Coll_fit_depth),Coll_fit_depth,'k--',markersize=10)
plt.plot(RetMIP_init_phi,RetMIP_init_depth,'ro',label=r'Vandecrux$^+$ (2020)',markersize=10, markerfacecolor = 'none',color=red)
plt.plot(RetMIP_fit_porosity(RetMIP_fit_depth),RetMIP_fit_depth,'r--',markersize=10,color=red,label=r'Present (Fit)')
plt.plot(Samira_SiteA_init_phi,Samira_SiteA_depth,'bs',label=r'Samimi$^+$ (2021) Site-A',markersize=10, markerfacecolor = 'none',color=blue)
#plt.plot(Samira_SiteA_fit_porosity(Samira_SiteA_fit_depth),Samira_SiteA_fit_depth,'b--',markersize=10)

plt.plot(Samira_SiteB_init_phi,Samira_SiteB_depth,'gd',label=r'Samimi$^+$ (2021) Site-B',markersize=10, markerfacecolor = 'none',color=green)
#plt.plot(Samira_SiteB_fit_porosity(Samira_SiteB_fit_depth),Samira_SiteB_fit_depth,'g--',markersize=10)
plt.plot(1-dR[:,5]/917,(dR[:,0]+dR[:,1])/2,'g.',label=r'Rennermalm$^+$ (2022)',markersize=10,color=green)
#plt.axvline(0.05, ymin=0, ymax=5,alpha=0.2)

plt.xlabel(r'$\phi$')
plt.ylabel(r'Depth [m]')
#plt.ylim([20,0])
plt.tight_layout()
plt.legend(framealpha=0.5,loc=(1.04, 0))
plt.xlabel(r'Porosity $[-]$')
plt.ylim([5,0])
plt.xlim([0,0.75])
plt.tight_layout()
plt.savefig(f'../Figures/Combined_analysis_porosity.pdf',bbox_inches='tight', dpi = 600)


fig = plt.figure(figsize=(15,7.5) , dpi=100)
plot = plt.plot(RetMIP_abs_time/day2s, Q_t,'r-', label='Turbulent')
plot = plt.plot(RetMIP_abs_time/day2s, Q_l,'b-', label='Latent')
plt.plot(np.linspace(0,179,1790),RetMIP_Q(np.linspace(0,179,1790)),'k--')
plt.plot(np.linspace(0,120,1200),Coll_Qnet_func(np.linspace(0,120,1200)),'g-', label='Samimi$^+$ (2021)')
plot = plt.plot(RetMIP_abs_time/day2s,  Q,'k--', label='Total')
plt.legend()
plt.xlabel(r'$t$ [days]')
plt.ylabel(r'Q [W/m$^2$]')
#plt.clim(0.000000, 1.0000000)
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
plt.savefig(f'../Figures/Q_combined_all.pdf',bbox_inches='tight', dpi = 600)


fig = plt.figure(figsize=(15,7.5) , dpi=100)
plot = plt.plot(RetMIP_abs_time/day2s, acc_rate,'r-')
plt.xlabel(r'$t$ [days]')
plt.ylabel(r'Accumulation rate [m w. eq./second]')
#plt.clim(0.000000, 1.0000000)
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
plt.savefig(f'../Figures/Accumulation.pdf',bbox_inches='tight', dpi = 600)



