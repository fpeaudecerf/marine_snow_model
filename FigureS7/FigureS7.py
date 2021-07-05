"""Script for plotting Figure S7 of the paper
"Sinking enhances the degradation of organic particle by marine bacteria"
Uria Alcolombri, François J. Peaudecerf, Vicente Fernandez, Lars Behrendt, Kang Soo Lee, Roman Stocker
Nature Geosciences (2021)

It shows the comparison between a reference case with degradation rate without
flow and the feedback case where flow speed increases degradation,
together with the same two curves when considering a fixed range of sizes
at every depth.

See caption of Figure S7 in Extended Data for more details

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

# Set of color blind friendly palette
orange = [230/255.0, 159/255.0, 0.0/255.0]
skyBlue = [86/255.0, 180/255.0, 233/255.0]
bluishGreen = [0/255.0, 158/255.0, 115.0/255.0]
yellow = [240/255.0, 228/255.0, 66.0/255.0]
blue = [0/255.0, 114/255.0, 178.0/255.0]
vermilion = [213/255.0, 94/255.0, 0.0/255.0]
reddishPurple = [204/255.0, 121/255.0, 167.0/255.0]


##### Computing the vertical fluxes and plotting #####

# we compute the results for the flux on a range of z
print('Reference size R_0 = {:.2e} m'.format(R_0))
print('Minimum radius R_l = {:.2e} m'.format(R_l))
print('Maximum radius R_g = {:.2e} m'.format(R_g))
zfmax = 20*z_0
Nzf = 190
azf = np.linspace(z_0, zfmax, num=Nzf+1, endpoint=True)

R_min_l = [20e-6]#,50e-6,125e-6]
labels = [r'Coupled (fixed lower size, $R_\mathrm{min} = 20\,\mu m$)']#,r'$R_\mathrm{min} = 50\mu m$',r'$R_\mathrm{min} = 125\mu m$']
labelsref = [r'Uncoupled (fixed lower size, $R_\mathrm{min} = 20\,\mu m$)']#,r'$R_\mathrm{min} = 50\mu m$',r'$R_\mathrm{min} = 125\mu m$']
colors = [orange]#, bluishGreen, reddishPurple]

fig = plt.figure(figsize = (mm2inch(0.6*mm_w), mm2inch(0.5*mm_h))) 
gs = gridspec.GridSpec(1,1)
ax = fig.add_subplot(gs[0,0])

### we first compute the reference case with varying cut-off size with depth
FAm_ref = []
FAm = []

print('#####################################################################################')
print('Computation of vertical flux of POC in ref, feedback, and ref high cases with depth z')
print('#####################################################################################')
for zf in azf:
    print('zf = {:.0f} m'.format(zf))
    FAm.append(FA_z_m_flowbis(zf, z_0, B, omega, gamma, gamma2, E, chi, rho_dry, delta, C, R_l, R_g, beta, R_0))
    FAm_ref.append(FA_z_m_flow_refbis(zf, z_0, B, omega, gamma, E, chi, rho_dry, delta, C, R_l, R_g, beta, R_0))

FAm = np.asarray(FAm)
FAm_ref = np.asarray(FAm_ref)

ax.plot(FAm/FAm[0], azf, color = [0.0,0.0,0.0], linestyle = '-', marker = 'None', label='Coupled (unbounded particle size)')
ax.plot(FAm_ref/FAm[0],  azf, color = [0.0,0.0,0.0], linestyle = '--', marker = 'None', label='Uncoupled (unbounded particle size)')

# now we consider cut-off in size
for i,R_min in enumerate(R_min_l):
    FAm_ref = []
    FAm = []
    print('R_min = {:.2e} m'.format(R_min))

    # first guess for a lower bound of search
    R_low = 0.9*R_min
    R0_minbis = R_min # we know that's the solution at z0
    R0_min_ref = R_min

    for zf in azf:
        print('zf = {:.0f} m'.format(zf))
        # we start with the feedback case
        if R0_minbis >= R_g:
            R0_minbis = R_g
        else:
            R0_minbis = R_0_minbis(zf, R_min, z_0, B, omega, gamma, gamma2, E, chi, rho_dry,delta, 1, R_low,R_0)

        #when we are basically considering no particles in the integral, no need to look further
        if R0_minbis >= R_g:
            R0_minbis = R_g

        # we can now compute the flux at this depth in the feedback case
        FAm.append(FA_z_m_flow_fixed_cutoffbis(zf, z_0, B, omega, gamma, gamma2, E, chi, rho_dry, delta, C, R_l, R_g, beta,R0_minbis,R_0))

        # we then continue with the reference case
        # when we are basically considering no particles in the integral, no need to look further
        if R0_min_ref >= R_g:
            R0_min_ref = R_g
        else:
            R0_min_ref = R_0_min_ref(zf, R_min, z_0, B, omega, gamma, E, chi, rho_dry,delta, 1, R_low,R_0)

        # when we are basically considering no particles in the integral, no need to look further
        if R0_min_ref >= R_g:
            R0_min_ref = R_g

        # we can now compute the flux at this depth in the feedback case
        FAm_ref.append(FA_z_m_flow_ref_fixed_cutoffbis(zf, z_0, B, omega, gamma, E, chi, rho_dry, delta, C, R_l, R_g, beta,R0_min_ref,R_0))
    
    FAm = np.asarray(FAm)
    FAm_ref = np.asarray(FAm_ref)

    # normalised option
    ax.plot(FAm/FAm[0], azf, color = colors[i], linestyle = '-', marker = 'None',label = labels[i])
    ax.plot(FAm_ref/FAm[0],  azf, color = colors[i], linestyle = '--', marker = 'None',label = labelsref[i])

ax.set_xlabel(r'Vertical flux $F(Z)/F(Z_0)$')
ax.set_ylabel(r'Depth $Z \;(\mathrm{m})$')

ax.invert_yaxis()
ax.xaxis.tick_top()
ax.xaxis.set_label_position('top') 

# Setting axes limits
ax.set_xlim([0,1.0])

plt.legend(loc = 0, fontsize = 5.0, edgecolor = 'none')

gs.tight_layout(fig, rect = [0.0, 0.0, 1.0, 1.0],  h_pad = 0.0, w_pad = 0.0, pad = pad)

# plt.show()
# plt.close()
# exit()
utils.save("FigureS7", ext=ext, close=True, verbose=True)
