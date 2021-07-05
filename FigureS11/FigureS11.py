"""Script for plotting FigureS11 showing the effect of varying viscosity with change of temperature
on the coupled and uncoupled vertical profile of carbon export, part of the paper
"Sinking enhances the degradation of organic particle by marine bacteria"
Uria Alcolombri, François J. Peaudecerf, Vicente Fernandez, Lars Behrendt, Kang Soo Lee, Roman Stocker
Nature Geosciences (2021)

It shows the comparison between a reference case with degradation rate without
flow and the feedback case where flow speed increases degradation.

Note that the temperature profile is hard coded in dynamic_laws_v2

See caption of Figure S11 in manuscript for more details.

Author: Francois Peaudecerf
Creation: 23.06.2020

History of modification
23.06.2020: creation from Figure 4 script
04.07.2021: editing for publication on Github
"""
# from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib as matplotlib
import sys,os
sys.path.append(os.path.realpath('..'))
import utils
from params_T import *
from dynamic_laws import *
from scipy import stats

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
ext = 'png'#'eps'


##### Computing the vertical fluxes #####

print('Computation for viscosity modulation by T variation with depth')
print('Reference size R_0 = {:.2e} m'.format(R_0))
print('Minimum radius R_l = {:.2e} m'.format(R_l))
print('Maximum radius R_g = {:.2e} m'.format(R_g))
print('Initial depth z_0 = {:.0f} m'.format(z_0))
zfmax = 20*z_0
Nzf = 190
azf = np.linspace(z_0, zfmax, num = Nzf + 1, endpoint = True)

FAm_ref = []
FAm = []
FAm_ref_T = []
FAm_T = []

### load if files exist ###
if os.path.isfile('F_FigS11_T.npy') and os.path.isfile('F_ref_FigS11_T.npy') and os.path.isfile('F_ref_FigS11_TT.npy') and os.path.isfile('F_FigS11_TT.npy'):
    FAm = np.load('F_FigS11_T.npy',allow_pickle = True)  
    print('Coupled flux loaded from file')
    print(' ')
    FAm_T = np.load('F_FigS11_TT.npy',allow_pickle = True)  
    print('Coupled flux with T variation loaded from file')
    print(' ')
    FAm_ref = np.load('F_ref_FigS11_T.npy',allow_pickle = True)  
    print('Reference flux loaded from file')
    print(' ')
    FAm_ref_T = np.load('F_ref_FigS11_TT.npy',allow_pickle = True)  
    print('Reference flux with T variation loaded from file')
    print(' ')
else:


    print('##########################################################################')
    print('Computation of vertical flux of POC in ref and feedback cases with depth z')
    print('##########################################################################')
    for zf in azf:
        print('zf = {:.0f} m'.format(zf))
        FAm.append(FA_z_m_flowbis(zf, z_0, B, omega, gamma, gamma2, E, chi, rho_dry, delta, C, R_l, R_g, beta,R_0))
        FAm_T.append(FA_z_m_flowbis_T(zf, z_0, B_prime, omega, gamma, gamma1, E, chi, rho_dry, D, delta, C, R_l, R_g, beta,R_0))
        FAm_ref.append(FA_z_m_flow_refbis(zf, z_0, B, omega, gamma, E, chi, rho_dry, delta, C, R_l, R_g, beta,R_0))
        FAm_ref_T.append(FA_z_m_flow_refbis_T(zf, z_0, B_prime, omega, gamma, E, chi, rho_dry, delta, C, R_l, R_g, beta,R_0))


    FAm = np.asarray(FAm)
    FAm_T = np.asarray(FAm_T)
    FAm_ref = np.asarray(FAm_ref)
    FAm_ref_T = np.asarray(FAm_ref_T)

    ### save to file ####
    np.save('F_FigS11_T',FAm)
    np.save('F_FigS11_TT',FAm_T)
    np.save('F_ref_FigS11_T',FAm_ref)
    np.save('F_ref_FigS11_TT',FAm_ref_T)

### Printing transfer efficiencies ###
print('Computation of transfer efficiencies')
T100 = FAm[azf==200][0]/FAm[0]
T100_T = FAm_T[azf==200][0]/FAm_T[0]
T100_ref = FAm_ref[azf==200][0]/FAm_ref[0]
T100_ref_T = FAm_ref_T[azf==200][0]/FAm_ref_T[0]


print("T_100 coupled: %f" % T100)
print("T_100 coupled T: %f" % T100_T)
print("T_100 uncoupled: %f" %T100_ref)
print("T_100 uncoupled T: %f" %T100_ref_T)


##### Plotting #####

fig = plt.figure(figsize = (mm2inch(0.6*mm_w), mm2inch(0.5*mm_h))) 
gs = gridspec.GridSpec(1,1)
ax = fig.add_subplot(gs[0,0])

ax.plot(FAm/FAm[0], azf, color = [0.4,0.4,1.0], linestyle = '-', marker = 'None', label = 'Coupled')
ax.plot(FAm_T/FAm_T[0], azf, color = [0.2,0.2,0.6], linestyle = '--', marker = 'None', label = 'Coupled with varying viscosity')
ax.plot(FAm_ref/FAm[0],  azf,color = [1.0,0.4,0.4], linestyle = '-', marker = 'None', label='Uncoupled')
ax.plot(FAm_ref_T/FAm_ref_T[0],  azf,color = [0.6,0.2,0.2], linestyle = '--', marker = 'None', label='Uncoupled with varying viscosity')

ax.set_xlabel(r'Vertical flux $F(Z)/F(Z_0)$')
ax.set_ylabel(r'Depth $Z \;(\mathrm{m})$')

ax.invert_yaxis()
ax.xaxis.tick_top()
ax.xaxis.set_label_position('top') 

ax.set_xlim([0,1.0])

plt.legend(loc = 0, fontsize = 4.5, edgecolor = 'none')

gs.tight_layout(fig, rect = [0.0, 0.0, 1.0, 1.0],  h_pad = 0.0, w_pad = 0.0, pad = pad)

# plt.show()
# plt.close()
# exit()
utils.save("FigureS11", ext=ext, close=True, verbose=True)




