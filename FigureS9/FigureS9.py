"""Script for plotting Figure S9 on the effect of different ranges of particles sizes,
and of different beta coefficients in particle size distributions on the transfer efficiency 
at 100m below euthrophic zone, part of the paper
"Sinking enhances the degradation of organic particle by marine bacteria"
Uria Alcolombri, François J. Peaudecerf, Vicente Fernandez, Lars Behrendt, Kang Soo Lee, Roman Stocker
Nature Geosciences (2021)

The main comparison is between a reference case with degradation rate without
flow and the feedback case where flow speed increases microbial degradation.

See caption of Figure S9 in Extended Data for more details.

This script also produces ancillary plots for each of the conditions studied, and saves the flux profiles to files.

Author: Francois Peaudecerf
Creation: 23.06.2020

History of modification
23.06.2020: creation from Figure 4 script.
04.07.2020: design into a sensitivity figure for range and beta
01.07.2021: editing for publication on Github
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
# ext = 'eps'

##### Computing the vertical fluxes #####

# we compute the results for the flux on a range of z
print('Computation for different size distributions')
print('Reference size R_0 = {:.2e} m'.format(R_0))
print('Minimum radius R_l = {:.2e} m'.format(R_l))
print('Maximum radius R_g = {:.2e} m'.format(R_g))
print('Initial depth z_0 = {:.0f} m'.format(z_0))
zfmax = 20*z_0
Nzf = 190
azf = np.linspace(z_0, zfmax, num=Nzf+1, endpoint=True)

# different exponents for the power law, without the minus sign
betas = np.linspace(2, 5, 7)
# two ranges of sizes, short and long 
R_ls = np.array([125.0,22.5])*1e-6 # in meters
R_gs = np.array([750.0,2500.0])*1e-6 # in meters
rangelabels = ['narrow','wide']
print('List of betas = ', betas)
print('min and max radii =',R_ls, R_gs)

### load if files exist ###
filename = 'T100s'
if os.path.isfile(filename + '.npy'):
    T100s = np.load(filename + '.npy',allow_pickle = True)  
    print('Transfer efficiencies at 100m loaded from file')
    print(' ')
else:

    T100s = np.zeros((betas.shape[0], R_ls.shape[0], 2)) # indexation in betas, range, coupled vs noncoupled

    ##### initialisation of temp variables
    FAm_temp_ref = []
    FAm_temp = []

    for i, beta in enumerate(betas):
        for j, label in enumerate(rangelabels):

            filename = "F_FigS9_beta_" + str(betas[i]) + "_range_" + rangelabels[j]
            filename2 = "FigS9_supplPlot_beta_" + str(betas[i]) + "_range_" + rangelabels[j]


            print('beta = ', beta)
            R_l = R_ls[j]
            R_g = R_gs[j]
            print('range of radii = {0:.2e} to {1:.2e} meters'.format(R_l, R_g))

            for zf in azf:
                print('zf = {:.0f} m'.format(zf))
                FAm_temp.append(FA_z_m_flowbis(zf, z_0, B, omega, gamma, gamma2, E, chi, rho_dry, delta, C, R_l, R_g, beta,R_0))
                FAm_temp_ref.append(FA_z_m_flow_refbis(zf, z_0, B, omega, gamma, E, chi, rho_dry, delta, C, R_l, R_g, beta,R_0))
            FAm_temp = np.asarray(FAm_temp)
            FAm_temp_ref = np.asarray(FAm_temp_ref)

            ### save to file ####
            np.save(filename, FAm_temp)
            np.save(filename + '_ref', FAm_temp_ref)

            ### Printing transfer efficiencies ###
            print('Computation of transfer efficiencies')
            T100s[i, j, 0] = FAm_temp[azf==200][0]/FAm_temp[0]
            T100s[i, j, 1] = FAm_temp_ref[azf==200][0]/FAm_temp_ref[0]

            print("T_100 coupled: %f" % T100s[i, j, 0])
            print("T_100 uncoupled: %f" % T100s[i, j, 1])
            print(FAm_temp[0], FAm_temp_ref[0])

            fig = plt.figure(figsize = (mm2inch(0.6*mm_w), mm2inch(0.5*mm_h))) 
            gs = gridspec.GridSpec(1,1)
            ax = fig.add_subplot(gs[0,0])

            ax.plot(FAm_temp/FAm_temp[0], azf, color = [0.4,0.4,1.0], linestyle = '-', marker = 'None', label = 'Coupled')
            ax.plot(FAm_temp_ref/FAm_temp[0],  azf,color = [1.0,0.4,0.4], linestyle = '-', marker = 'None', label='Uncoupled')

            ax.set_xlabel(r'Normalised dry mass flux $F(Z)/F(Z_0)$')
            ax.set_ylabel(r'Depth $Z \;(\mathrm{m})$')

            ax.invert_yaxis()
            ax.xaxis.tick_top()
            ax.xaxis.set_label_position('top') 
            ax.set_xlim([0,1.0])
            plt.legend(loc = 0, fontsize = 7.5, edgecolor = 'none')

            gs.tight_layout(fig, rect = [0.0, 0.0, 1.0, 1.0],  h_pad = 0.0, w_pad = 0.0, pad = pad) # pad corresponds to the padding around the subplot domain

            utils.save(filename2, ext=ext, close=True, verbose=True)

            FAm_temp_ref = []
            FAm_temp = []

    np.save('T100s', T100s)

# print(T100s[:,:,:])
attenuation = np.ones_like(T100s)-T100s
ratio = attenuation[:,:,0]/attenuation[:,:,1]
# print(ratio)

##### Plotting ######

fig = plt.figure(figsize = (mm2inch(0.6*mm_w), mm2inch(0.5*mm_h))) 
gs = gridspec.GridSpec(1,1)
ax = fig.add_subplot(gs[0,0])

ax.plot(- betas, 100*T100s[:,0,0], markeredgecolor = [0.3,0.3,1.0], markerfacecolor = 'none', linestyle = 'none', marker = 'o', label = 'Coupled, '+rangelabels[0] + ' size range')
ax.plot(- betas, 100*T100s[:,0,1], markeredgecolor = [1.0,0.3,0.3], markerfacecolor = 'none', linestyle = 'none', marker = 'o', label='Uncoupled, '+rangelabels[0] + ' size range')

ax.plot(- betas, 100*T100s[:,1,0], markeredgecolor = [0.3,0.3,1.0], markerfacecolor = [0.3,0.3,1.0], linestyle = 'none', marker = '+', label = 'Coupled, '+rangelabels[1] + ' size range')
ax.plot(- betas, 100*T100s[:,1,1], markeredgecolor = [1.0,0.3,0.3], markerfacecolor = 'none', linestyle = 'none', marker = '+', label='Uncoupled, '+rangelabels[1] + ' size range')

ax.set_xlabel(r'PSD exponent $\beta$')
ax.set_ylabel(r'$T_{100}$ in %')


# Setting axes limits
ax.set_xlim([-betas[-1]-0.5,-betas[0]+0.5])

plt.legend(loc = 0, fontsize = 5.0, edgecolor = 'none')

gs.tight_layout(fig, rect = [0.0, 0.0, 1.0, 1.0],  h_pad = 0.0, w_pad = 0.0, pad = pad)

# plt.show()
# plt.close()
# exit()
utils.save("FigureS9", ext = ext, close = True, verbose = True)




