"""Script for plotting Figure S10 on the effect of degradation slow-down with decreasing temperature 
on the observed coupling of flow and degradation, part of the paper
"Sinking enhances the degradation of organic particle by marine bacteria"
Uria Alcolombri, François J. Peaudecerf, Vicente Fernandez, Lars Behrendt, Kang Soo Lee, Roman Stocker
Nature Geosciences (2021)

It shows the comparison between a reference case with degradation rate without
flow and the feedback case where flow speed increases degradation.
It also shows the result of the model when considering a varying degradation rate with temperature,
both for a uniformly dampened degradation and more complex temperature profile.

Note that the temperature profile is coded in dynamic_laws_v2

See caption of Figure S10 in manuscript for more details.

Author: Francois Peaudecerf
Creation: 23.06.2020

History of modification
23.06.2020: creation from Figure 4 script.
03.07.2020: modification to include the Q10
07.07.2020: further edits of figure
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
from params_Q10 import *
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
ext = 'png'#'eps'

##### Computing the vertical fluxes #####

print('Temperature coefficient Q_10 = {:.1f}'.format(Q_10))

print('Computation with degradation modulation by T via a temperature coefficient Q10')
print('Reference size R_0 = {:.2e} m'.format(R_0))
print('Minimum radius R_l = {:.2e} m'.format(R_l))
print('Maximum radius R_g = {:.2e} m'.format(R_g))
print('Initial depth z_0 = {:.0f} m'.format(z_0))
zfmax = 20*z_0
Nzf = 190
azf = np.linspace(z_0, zfmax, num = Nzf + 1, endpoint = True)

# ##### initialisation of fluxes
FAm_ref = []
FAm = []
FAm_ref_cold = [] # corresponds to uniform degradation rate with water column but for 15 degrees
FAm_cold = []     # corresponds to uniform degradation rate with water column but for 15 degrees
FAm_ref_Q10 = []
FAm_Q10 = []

### load if files exist ###
if (   os.path.isfile('F_FigS10_Q10.npy')        and
       os.path.isfile('F_ref_FigS10_Q10.npy')    and 
       os.path.isfile('F_ref_FigS10_Q10Q10.npy') and
       os.path.isfile('F_FigS10_Q10Q10.npy')     and
       os.path.isfile('F_FigS10_Q10cold.npy')    and
       os.path.isfile('F_ref_FigS10_Q10cold.npy')):

    FAm = np.load('F_FigS10_Q10.npy',allow_pickle = True)  
    print('Coupled flux loaded from file')
    print(' ')
    FAm_cold = np.load('F_FigS10_Q10cold.npy',allow_pickle = True)  
    print('Cold coupled flux loaded from file')
    print(' ')
    FAm_Q10 = np.load('F_FigS10_Q10Q10.npy',allow_pickle = True)  
    print('Coupled flux with Q10 variation loaded from file')
    print(' ')
    FAm_ref = np.load('F_ref_FigS10_Q10.npy',allow_pickle = True)  
    print('Reference flux loaded from file')
    print(' ')
    FAm_ref_cold = np.load('F_ref_FigS10_Q10cold.npy',allow_pickle = True)  
    print('Cold reference flux loaded from file')
    print(' ')
    FAm_ref_Q10 = np.load('F_ref_FigS10_Q10Q10.npy',allow_pickle = True)  
    print('Reference flux with Q10 variation loaded from file')
    print(' ')
else:


    print('##########################################################################')
    print('Computation of vertical flux of POC in ref and feedback cases with depth z')
    print('with addition of modulation of degradation by temperature via a Q10 factor')
    print('both as a uniform decrease over water column (experimental degradation rate ')
    print('measured at 25 degrees brought down to 15 degrees)                        ')
    print('and as a modulation function of a complex temperature profile with depth (Zakem2018)')
    print('##########################################################################')
    for zf in azf:
        print('zf = {:.0f} m'.format(zf))
        FAm.append(FA_z_m_flowbis(zf, z_0, B, omega, gamma, gamma2, E, chi, rho_dry, delta, C, R_l, R_g, beta,R_0))
        # if the water column is at 15C, so 10C below lab T of 25C, degradation is just overall reduced by a factor Q_10
        FAm_cold.append(FA_z_m_flowbis(zf, z_0, B, omega, gamma/Q_10, gamma2/Q_10, E, chi, rho_dry, delta, C, R_l, R_g, beta,R_0))
        FAm_Q10.append(FA_z_m_flowbis_Q10(zf, z_0, B, omega, gamma, gamma2, E, chi, rho_dry, delta, C, R_l, R_g, beta,R_0, T_exp, Q_10))
        FAm_ref.append(FA_z_m_flow_refbis(zf, z_0, B, omega, gamma, E, chi, rho_dry, delta, C, R_l, R_g, beta,R_0))
        # if the water column is at 15C, so 10C below lab T of 25C, degradation is just overall reduced by a factor Q_10
        FAm_ref_cold.append(FA_z_m_flow_refbis(zf, z_0, B, omega, gamma/Q_10, E, chi, rho_dry, delta, C, R_l, R_g, beta,R_0))
        FAm_ref_Q10.append(FA_z_m_flow_refbis_Q10(zf, z_0, B, omega, gamma, E, chi, rho_dry, delta, C, R_l, R_g, beta,R_0, T_exp, Q_10))


    FAm = np.asarray(FAm)
    FAm_cold = np.asarray(FAm_cold)
    FAm_Q10 = np.asarray(FAm_Q10)
    FAm_ref = np.asarray(FAm_ref)
    FAm_ref_cold = np.asarray(FAm_ref_cold)
    FAm_ref_Q10 = np.asarray(FAm_ref_Q10)


    ### save to file ####
    np.save('F_FigS10_Q10',FAm)
    np.save('F_FigS10_Q10cold',FAm_cold)
    np.save('F_FigS10_Q10Q10',FAm_Q10)
    np.save('F_ref_FigS10_Q10',FAm_ref)
    np.save('F_ref_FigS10_Q10cold',FAm_ref_cold)
    np.save('F_ref_FigS10_Q10Q10',FAm_ref_Q10)

### Printing transfer efficiencies ###
print('Computation of transfer efficiencies')
T100 = FAm[azf==200][0]/FAm[0]
T100_cold = FAm_cold[azf==200][0]/FAm_cold[0]
T100_Q10 = FAm_Q10[azf==200][0]/FAm_Q10[0]
T100_ref = FAm_ref[azf==200][0]/FAm_ref[0]
T100_ref_cold = FAm_ref_cold[azf==200][0]/FAm_ref_cold[0]
T100_ref_Q10 = FAm_ref_Q10[azf==200][0]/FAm_ref_Q10[0]


print("T_100 coupled: %f" % T100)
print("T_100 cold coupled: %f" % T100_cold)
print("T_100 coupled Q10: %f" % T100_Q10)
print("T_100 uncoupled: %f" %T100_ref)
print("T_100 cold uncoupled: %f" %T100_ref_cold)
print("T_100 uncoupled Q10: %f" %T100_ref_Q10)


##### Plotting #####

fig = plt.figure(figsize = (mm2inch(0.6*mm_w), mm2inch(0.5*mm_h))) 
gs = gridspec.GridSpec(1,1)
ax = fig.add_subplot(gs[0,0])

ax.plot(FAm/FAm[0], azf, color = [0.4,0.4,1.0], linestyle = '-', marker = 'None', label = 'Coupled with uniform T = 25 °C')
ax.plot(FAm_cold/FAm_cold[0], azf, color = [0.4,0.4,1.0], linestyle = '--', marker = 'None', label = 'Coupled with uniform T = 15 °C')
ax.plot(FAm_Q10/FAm_Q10[0], azf, color = [0.2,0.2,0.8], linestyle = '-.', marker = 'None', label = 'Coupled with T(z)')
ax.plot(FAm_ref/FAm[0],  azf, color = [1.0,0.4,0.4], linestyle = '-', marker = 'None', label='Uncoupled with uniform T = 25 °C')
ax.plot(FAm_ref_cold/FAm_cold[0], azf, color = [1.0,0.4,0.4], linestyle = '--', marker = 'None', label='Uncoupled with uniform T = 15 °C')
ax.plot(FAm_ref_Q10/FAm_ref_Q10[0],  azf, color = [0.8,0.2,0.2], linestyle = '-.', marker = 'None', label='Uncoupled with T(z)')

ax.set_xlabel(r'Vertical flux $F(Z)/F(Z_0)$')
ax.set_ylabel(r'Depth $Z \;(\mathrm{m})$')

ax.invert_yaxis()
ax.xaxis.tick_top()
ax.xaxis.set_label_position('top') 

ax.set_xlim([0,1.0])

plt.legend(loc = 0, fontsize = 5.0, edgecolor = 'none')

gs.tight_layout(fig, rect = [0.0, 0.0, 1.0, 1.0],  h_pad = 0.0, w_pad = 0.0, pad = pad)

# plt.show()
# plt.close()
# exit()
utils.save("FigureS10", ext = ext, close = True, verbose = True)




