"""Parameters for the model of marine particle microbial degradation
and coupling of degradation with sinking speed, part of the paper

"Sinking enhances the degradation of organic particle by marine bacteria"
Uria Alcolombri, François J. Peaudecerf, Vicente Fernandez, Lars Behrendt, Kang Soo Lee, Roman Stocker
Nature Geosciences (2021)

See Extended Data for more details on chosen parameter values.

Author: Francois Peaudecerf
Creation: 25.02.2019

History of modification
08.10.2019: modification of chosen parameter values
26.11.2019: modification of value of gamma_1 following new fit of experimental data
05.07.2021: editing for publication on Github
"""
# from __future__ import division

###### Numerical parameters #########

mu    = 1e-3     # kg/m/s, dynamic viscosity of water
Drho  = 0.54     # kg/m^3, difference of volumic mass
g     = 10       # m/s^2, gravity
R_0   = 4.4e-4   # m, initial radius in experiments
gamma = 1.06e-10 # m/s, radius shrinking rate from experiments with no flow
R_l   = 125e-6   # m, minimum cut-off radius for size distribution
R_g   = 750e-6   # m, maximum cut-off radius for size distribution
z_0   = 100      # m, depth of particle distribution
C     = 200e3    # m-3, total particle abundance
beta  = 4.0      # power law parameter for particle distribution
omega = 0.26     # exponent in the power law for settling velocity with radius (Alldredge 1988)
B     = 4.18e-3  # m^(1-omega) s-1, pre-factor for settling velocity with radius (Alldredge 1988)
nu    = 1.20e-6  # m2/s, kinematic viscosity modified for sea water at 15 degrees C in v1
D     = 1e-9     # m2/s, diffusivity for Peclet estimation
delta = 0.412    # exponent in the Sherwood expression
gamma1= 4.35e-10 # coefficient in the fit of degradation rate with Sh
gamma2= 0.619*gamma1*(nu/D)**(1.0/3.0)*(2.0/nu)**delta # aggregate coefficient for gamma
E     = 4.55e-5  # kg/m^{chi}, pre-factor in the power law for dry mass as function of size (Alldredge 1988)
chi   = 1.125    # exponent in the power law for dry mass as function of size (Alldredge 1988)
rho_dry= 15      # kg/m^3, dry mass per unit volume of alginate particles

# Alldredge 1988: Alldredge, A. L. & Gotschalk, C. In situ settling behavior of marine snow. Limnol. Ocean. 33, 339–35 (1988)
