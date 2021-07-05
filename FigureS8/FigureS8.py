"""Script for plotting Figure S8 of the paper
"Sinking enhances the degradation of organic particle by marine bacteria"
Uria Alcolombri, François J. Peaudecerf, Vicente Fernandez, Lars Behrendt, Kang Soo Lee, Roman Stocker
Nature Geosciences (2021)

It shows how a uniform high degradation rate compares to the dynamic coupling
of flow speed and degradation implemented in our reference model.

See caption of Figure S8 in Extended Data for more details.

Author: Francois Peaudecerf
Creation: 08.10.2019

History of modification
08.10.2019: creation
26.11.2019: change of default parameter values and figure formatting
17.12.2019: further figure editing
30.06.2021: editing for publication on Github
"""
# from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib as matplotlib
import sys,os
sys.path.append(os.path.realpath('..'))
from scipy import stats
import utils
from params import *
from dynamic_laws import *

########## Preamble ##############

# width of figure in mm
mm_w = 150
# height of figure in mm
mm_h = 120
 
def mm2inch(value):
    return value/25.4
 
# definition of fontsize
fs = 10
matplotlib.rcParams.update({'font.size': fs})
 
# linewidth
lw = 1.0
 
# pad on the outside of the gridspec box
pad = 0.2
 
# extension for figures
ext = 'png'


##### Computing the vertical fluxes #####

# we compute the results for the flux on a range of z
print('Reference size R_0 = {:.2e} m'.format(R_0))
print('Minimum radius R_l = {:.2e} m'.format(R_l))
print('Maximum radius R_g = {:.2e} m'.format(R_g))
zfmax = 20*z_0
Nzf = 190
azf = np.linspace(z_0, zfmax, num=Nzf+1, endpoint=True)

FAm_nr_ref = []
FAm_nr = []
FAm_nr_refhigh = []

# we consider the rate of radius change applied to the largest particle at the initial depth as
# a uniform high rate to see how it compares to the more complex feedback
gamma_high = + gamma + gamma2*np.power(B,delta)*np.power(R_g,delta*(1.0 + omega))

print('#####################################################################################')
print('Computation of vertical flux of POC in ref, feedback, and ref high cases with depth z')
print('#####################################################################################')
for zf in azf:
    print('zf = {:.0f} m'.format(zf))
    FAm_nr.append(FA_z_m_flowbis(zf, z_0, B, omega, gamma, gamma2, E, chi, rho_dry, delta, C, R_l, R_g, beta,R_0))
    FAm_nr_ref.append(FA_z_m_flow_refbis(zf, z_0, B, omega, gamma, E, chi, rho_dry, delta, C, R_l, R_g, beta,R_0))
    # we need an extra long time range for reaching 2000m with the high uniform degradation rate
    FAm_nr_refhigh.append(FA_z_m_flow_refbis(zf, z_0, B, omega, gamma_high, E, chi, rho_dry, delta, C, R_l, R_g, beta,10*R_0))


FAm_nr = np.asarray(FAm_nr)
FAm_nr_ref = np.asarray(FAm_nr_ref)
FAm_nr_refhigh = np.asarray(FAm_nr_refhigh)

# We attempt a power law fit, following Martin et al and enforcing the
# initial value at z_0
slope, intercept, r_value, p_value, std_err = stats.linregress(np.log(azf/z_0),np.log(FAm_nr/FAm_nr[0]))
slope_ref, intercept_ref, r_value_ref, p_value_ref, std_err_ref = stats.linregress(np.log(azf/z_0),np.log(FAm_nr_ref/FAm_nr_ref[0]))
slope_refhigh, intercept_refhigh, r_value_refhigh, p_value_refhigh, std_err_refhigh = stats.linregress(np.log(azf/z_0),np.log(FAm_nr_refhigh/FAm_nr_ref[0]))

plot = 0
print('feedback case')
print("slope: %f    intercept: %f" % (slope, intercept))
print("r-squared: %f" % r_value**2)
print('reference case')
print("slope: %f    intercept: %f" % (slope_ref, intercept_ref))
print("r-squared: %f" % r_value_ref**2)
print('reference case high degradation')
print("slope: %f    intercept: %f" % (slope_refhigh, intercept_refhigh))
print("r-squared: %f" % r_value_refhigh**2)
if plot ==1:
    figtemp = plt.figure(figsize = (mm2inch(0.6*mm_w), mm2inch(0.5*mm_h))) 
    gstemp = gridspec.GridSpec(1,1)
    axtemp = figtemp.add_subplot(gstemp[0,0])

    axtemp.plot(np.log(FAm_nr/FAm_nr[0]),np.log(azf/z_0),'.r', label = 'model')
    axtemp.plot(intercept + slope*np.log(azf/z_0), np.log(azf/z_0), 'k--', label='fitted line')
    axtemp.plot(np.log(FAm_nr_ref/FAm_nr_ref[0]),np.log(azf/z_0),'.b', label = 'model ref')
    axtemp.plot(intercept_ref + slope_ref*np.log(azf/z_0), np.log(azf/z_0), 'g--', label='fitted line ref')
    axtemp.plot(np.log(FAm_nr_refhigh/FAm_nr_ref[0]),np.log(azf/z_0),'.g', label = 'model ref high')
    axtemp.plot(intercept_refhigh + slope_refhigh*np.log(azf/z_0), np.log(azf/z_0), 'm--', label='fitted line ref high')
    
    plt.legend()
    plt.show()
    plt.close()

# #### Plotting #####


fig = plt.figure(figsize = (mm2inch(0.6*mm_w), mm2inch(0.5*mm_h))) 
gs = gridspec.GridSpec(1,1)
ax = fig.add_subplot(gs[0,0])

ax.plot(FAm_nr/FAm_nr[0], azf, color = [0.4,0.4,1.0], linestyle = '-', marker = 'None',label='Coupled')

ax.plot(FAm_nr_ref/FAm_nr[0],  azf, color = [1.0,0.4,0.4], linestyle = '-', marker = 'None', label='Uncoupled')

ax.plot(FAm_nr_refhigh/FAm_nr[0],  azf, color = 'k', linestyle = '-.', marker = 'None', label='Uncoupled high')


# #####

ax.set_xlabel(r'Vertical flux $F(Z)/F(Z_0)$')
ax.set_ylabel(r'Depth $Z \;(\mathrm{m})$')

ax.invert_yaxis()
ax.xaxis.tick_top()
ax.xaxis.set_label_position('top') 

# Setting axes limits
ax.set_xlim([0,1.0])

plt.legend(loc = 0, fontsize = 7.5, edgecolor = 'none')

gs.tight_layout(fig, rect = [0.0, 0.0, 1.0, 1.0],  h_pad = 0.0, w_pad = 0.0, pad = pad)

# plt.show()
# plt.close()
# exit()
utils.save("FigureS8", ext=ext, close=True, verbose=True)




