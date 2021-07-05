"""Script for plotting Figure 4 of the paper
"Sinking enhances the degradation of organic particle by marine bacteria"
Uria Alcolombri, François J. Peaudecerf, Vicente Fernandez, Lars Behrendt, Kang Soo Lee, Roman Stocker
Nature Geosciences (2021)

It shows the comparison between a reference case with degradation rate without
flow and the feedback case where flow speed increases degradation.
On top are shown two fits as power laws.

See caption of Figure 4 in manuscript for more details

Author: Francois Peaudecerf
Creation: 08.10.2019

History of modification
08.10.2019: creation
26.11.2019: change of default parameter values and figure formatting
17.06.2020: save computed fluxes to file to allow quick replotting with re-running computation
30.06.2021: editing for publication on Github.
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


##### Preamble #####
 
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


##### Computation of the vertical fluxes with depth #####

# we compute the results for the flux on a range of z
print('Reference size R_0 = {:.2e} m'.format(R_0))
print('Minimum radius R_l = {:.2e} m'.format(R_l))
print('Maximum radius R_g = {:.2e} m'.format(R_g))
print('Initial depth z_0 = {:.0f} m'.format(z_0))
zfmax = 20*z_0
Nzf = 190
azf = np.linspace(z_0, zfmax, num = Nzf+1, endpoint=True)

FAm_ref = []
FAm = []

### load if files exist ###
if os.path.isfile('F_Fig4.npy') and os.path.isfile('F_ref_Fig4.npy'):
    FAm = np.load('F_Fig4.npy',allow_pickle = True)  
    print('Coupled flux loaded from file')
    print(' ')
    FAm_ref = np.load('F_ref_Fig4.npy',allow_pickle = True)  
    print('Reference flux loaded from file')
    print(' ')
else:
    print('##########################################################################')
    print('Computation of vertical flux of POC in ref and feedback cases with depth z')
    print('##########################################################################')
    for zf in azf:
        print('zf = {:.0f} m'.format(zf))
        FAm.append(FA_z_m_flowbis(zf, z_0, B, omega, gamma, gamma2, E, chi, rho_dry, delta, C, R_l, R_g, beta,R_0))
        FAm_ref.append(FA_z_m_flow_refbis(zf, z_0, B, omega, gamma, E, chi, rho_dry, delta, C, R_l, R_g, beta,R_0))

    FAm = np.asarray(FAm)
    FAm_ref = np.asarray(FAm_ref)

    ### save to file ####
    np.save('F_Fig4',FAm)
    np.save('F_ref_Fig4',FAm_ref)

# We attempt a power law fit of the mass case, following Martin et al and enforcing the initial value at z_0
slope, intercept, r_value, p_value, std_err = stats.linregress(np.log(azf/z_0),np.log(FAm/FAm[0]))
slope_ref, intercept_ref, r_value_ref, p_value_ref, std_err_ref = stats.linregress(np.log(azf/z_0),np.log(FAm_ref/FAm_ref[0]))

print('feedback case')
print("slope: %f    intercept: %f" % (slope, intercept))
print("r-squared: %f" % r_value**2)
print('reference case')
print("slope: %f    intercept: %f" % (slope_ref, intercept_ref))
print("r-squared: %f" % r_value_ref**2)


##### Plotting #####

fig = plt.figure(figsize = (mm2inch(0.6*mm_w), mm2inch(0.5*mm_h))) 
gs = gridspec.GridSpec(1,1)
ax = fig.add_subplot(gs[0,0])

ax.plot(FAm/FAm[0], azf, color = [0.4,0.4,1.0], linestyle = '-', marker = 'None', label = 'Coupled')
ax.plot(FAm_ref/FAm[0],  azf,color = [1.0,0.4,0.4], linestyle = '-', marker = 'None', label='Uncoupled')

ax.plot(np.exp(intercept)*np.power(azf/z_0,slope), azf, dashes =[6,6], color = [0,0,0.2],  marker = 'None' ,lw=lw, label='Coupled (fit)')
ax.plot(np.exp(intercept_ref)*np.power(azf/z_0,slope_ref), azf, dashes =[2,2],color = [0.2,0,0], linestyle = '-.', marker = 'None',lw=lw ,label = 'Uncoupled (fit)')

ax.set_xlabel(r'Vertical flux $F(Z)/F(Z_0)$')
ax.set_ylabel(r'Depth $Z \;(\mathrm{m})$')

ax.invert_yaxis()
ax.xaxis.tick_top()
ax.xaxis.set_label_position('top') 

ax.set_xlim([0,1.0])

plt.legend(loc=0, fontsize = 7.5, edgecolor = 'none')

gs.tight_layout(fig, rect = [0.0, 0.0, 1.0, 1.0],  h_pad = 0.0, w_pad = 0.0, pad = pad)

# plt.show()
# plt.close()
# exit()
utils.save("Figure4", ext=ext, close=True, verbose=True)

