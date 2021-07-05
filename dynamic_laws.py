"""Support functions for computing the vertical carbon flux
associated with sinking particles based on different assumptions
of coupling between sinking speed and microbial degradation, of role 
of temperature and of particle distributions.

Associated with the paper
"Sinking enhances the degradation of organic particle by marine bacteria"
Uria Alcolombri, François J. Peaudecerf, Vicente Fernandez, Lars Behrendt, Kang Soo Lee, Roman Stocker
Nature Geosciences (2021)

For more details, see Extended Data of manuscript presenting the model.

Author: Francois Peaudecerf
Creation: 27.02.2019

History of modification
19.09.2019: replace integrate.odeint with integrate.solve_ivp for stability of time integration
05.07.2021: editing for publication on Github
"""
from __future__ import division
import numpy as np
from scipy import integrate
from scipy import optimize
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


####### General functions #######

#### Used by Figure4, FigureS7, FigureS8, FigureS9, FigureS10, FigureS11 ####
def P_R0(R_0,R_l,R_g, beta):
    '''Size distribution of particles between the two limit radii R_l and R_g
    R_l has to be smaller than R_g
    INPUT:
    R_0:   array of initial radii [m]
    R_l:    minimum radius cut off [m]
    R_g:   maximum radius cut-off [m]
    beta:  power law exponent [no units]
    OUTPUT
    P(R_0), probability distribution in 1/m'''
    if R_l>R_g:
        print("Error: max radius larger than min radius.")
    return ( (beta - 1)*(
                         R_l**(beta-1)*R_g**(beta-1)/(R_g**(beta-1)-R_l**(beta-1))
                         )*R_0**(-beta)
        )

#### Used by Figure4, FigureS7, FigureS8, FigureS9, FigureS10, FigureS11 ####
def U_A(R_0, B, omega):
    '''Returns the speed of a particle as a function of its radius
    R_0 following the empirical fit of Alldredge as a power law:
    Alldredge, A. L. & Gotschalk, C. In situ settling behavior of marine snow. Limnol. Ocean. 33, 339–35 (1988)
    Inputs:
    R_0 : radius [m]
    B   : empirical parameter [m^(1-omega) s-1]
    omega: empirical parameter [no units]
    OUTPUT:
    U   : sinking speed [m s^-1]'''
    return B*np.power(R_0,omega) 

#### Used by Figure4, FigureS7, FigureS8, FigureS9, FigureS10, FigureS11 ####
def find_nearest(array, value):
    array = np.asarray(array)
    idx = np.nanargmin((np.abs(array - value)))
    return idx

#### Used in FigureS11 ####
def muTS(T,S):
    '''Dynamic viscosity of sea-water, based on 
    Mostafa H. Sharqawy, John H. Lienhard V, and Syed M. Zubair, 
    "Thermophysical properties of seawater: A review of existing correlations and data," 
    Desalination and Water Treatment, Vol. 16, pp.354-380, 2010.
    INPUTS:
    T : temperature, in Celsius
    S : salinity, in mass fraction kg/kg
    OUTPUT:
    mu: dynamic viscosity in [kg/m-s]'''
    a = [
        1.5700386464E-01,
        6.4992620050E+01,
       -9.1296496657E+01,
        4.2844324477E-05,
        1.5409136040E+00,
        1.9981117208E-02,
       -9.5203865864E-05,
        7.9739318223E+00,
       -7.5614568881E-02,
        4.7237011074E-04
    ]

    mu_w = a[3] + 1./(a[0]*(T+a[1])**2+a[2])


    A  = a[4] + a[5] * T + a[6] * T**2
    B  = a[7] + a[8] * T + a[9]* T**2
    mu = mu_w*(1 + A*S + B*S**2)
    return mu

#### Used in FigureS11 ####
def rhoTS(T,S):
    '''Dynamic viscosity, based on 
    Mostafa H. Sharqawy, John H. Lienhard V, and Syed M. Zubair, 
    "Thermophysical properties of seawater: A review of existing correlations and data," 
    Desalination and Water Treatment, Vol. 16, pp.354-380, 2010.
    INPUTS:
    T : temperature, in Celsius
    S : salinity, in mass fraction kg/kg
    OUTPUT:
    rho: density in [kg/m^3]'''

    a = [
         9.9992293295E+02,
         2.0341179217E-02,
        -6.1624591598E-03,
         2.2614664708E-05,
        -4.6570659168E-08
    ]

    b = [
         8.0200240891E+02,
        -2.0005183488E+00,
         1.6771024982E-02,
        -3.0600536746E-05,
        -1.6132224742E-05
    ]

    rho_w = a[0] + a[1]*T + a[2]*T**2 + a[3]*T**3 + a[4]*T**4
    D_rho = b[0]*S + b[1]*S*T + b[2]*S*T**2 + b[3]*S*T**3 + b[4]*S**2*T**2
    rho_sw_sharq   = rho_w + D_rho
    return rho_sw_sharq

#### Used in FigureS11 ####
def nuTS(T,S):
    '''Kinematic viscosity, based on 
    Mostafa H. Sharqawy, John H. Lienhard V, and Syed M. Zubair, 
    "Thermophysical properties of seawater: A review of existing correlations and data," 
    Desalination and Water Treatment, Vol. 16, pp.354-380, 2010.
    INPUTS:
    T : temperature, in Celsius
    S : salinity, in mass fraction kg/kg
    OUTPUT:
    mu: Kinematic viscosity in [m2/s]'''
    mu_sw = muTS(T,S)
    rho_sw = rhoTS(T,S)
    nu_sw = mu_sw/rho_sw
    return nu_sw

#### Used in FigureS10, FigureS11 ####
def T_profile(z):
    '''Temperature profile with depth, from 
    Zakem et al. 
    "Ecological control of nitrite in the upper ocean"
    Nature Communications, Vol. 9, 1206, 2018.
    INPUT:
    depth z, in [m]
    OUTPUT:
    temperature T, in celsius'''
    T = 12.0*np.exp(-z/150.0) + 12.0*np.exp(-z/500.0) + 2.0
    return T


####### Integration functions for the reference case without flow coupling #######

#### Used by Figure4, FigureS7, FigureS8, FigureS9, FigureS10, FigureS11 ####
def ODE_Rz_mref(t,y, gamma, B, delta, omega, E, chi, rho_dry):
    '''Function returning the instantaneous derivative of radius R(t) and depth z(t) for a particle of given size,
       in the framework of Alldredge sinking speed and Alldredge emprical law for dry mass with size
       without feedback of flow on degradation (reference case)
    INPUTS:
    y                : solution R, z for which we solve
    B                : coefficient in Alldredge fit of sinking speed
    omega            : exponent in the power law for Alldredge speed as function of radius
    gamma            : radius shrinking rate without flow
    delta            : exponent in Sherwood law with Reynolds for Re>0.1
    E                : pre-factor in the power law for dry mass as function of size (Alldredge L&O 1988)
    chi              : exponent in the power law for dry mass as function of size (Alldredge L&O 1988)
    rho_dry          : dry mass per unit volume of alginate particles
    OUTPUT:
    dydt             : time derivative for values R and z'''
    R, z = y
    dydt = [  -4*np.pi*rho_dry*np.power(R,3-chi)*(gamma)/(E*chi),B*np.power(R,omega)]
    return dydt

#### Used in FigureS11 ####
def ODE_Rz_mref_T(t,y, gamma, B_prime, delta, omega, E, chi, rho_dry):
    '''Function returning the instantaneous derivative of radius R(t) and depth z(t) for a particle of given size,
        in the framework of Alldredge sinking speed and Alldredge emprical law for dry mass with size
        without feedback of flow on degradation (reference case)
    INPUTS:
    y                : solution R, z for which we solve
    B_prime          : coefficient in Alldredge fit of sinking speed multiplied by dynamic viscosity in Alldredge conditions
    omega            : exponent in the power law for Alldredge speed as function of radius
    gamma            : radius shrinking rate without flow
    delta            : exponent in Sherwood law with Reynolds for Re>0.1
    E                : pre-factor in the power law for dry mass as function of size (Alldredge L&O 1988)
    chi              : exponent in the power law for dry mass as function of size (Alldredge L&O 1988)
    rho_dry          : dry mass per unit volume of alginate particles
    OUTPUT:
    dydt             : time derivative for values R and z'''
    R, z = y
    T = T_profile(z)
    muTS1 = muTS(T,33.6/1000)
    dydt = [  -4*np.pi*rho_dry*np.power(R,3-chi)*(gamma)/(E*chi),B_prime*np.power(R,omega)/muTS1]
    return dydt

#### Used in FigureS10 ####
def ODE_Rz_mref_Q10(t,y, gamma, B, delta, omega, E, chi, rho_dry, T_exp, Q_10):
    '''Function returning the instantaneous derivative of radius R(t) and depth z(t) for a particle of given size,
        in the framework of Alldredge sinking speed and Alldredge emprical law for dry mass with size
        without feedback of flow on degradation (reference case) and with Q10 modulation of degradation
    INPUTS:
    y                : solution R, z for which we solve
    B                : coefficient in Alldredge fit of sinking speed
    omega            : exponent in the power law for Alldredge speed as function of radius
    gamma            : radius shrinking rate without flow
    delta            : exponent in Sherwood law with Reynolds for Re>0.1
    E                : pre-factor in the power law for dry mass as function of size (Alldredge L&O 1988)
    chi              : exponent in the power law for dry mass as function of size (Alldredge L&O 1988)
    rho_dry          : dry mass per unit volume of alginate particles
    T_exp            : temperature of experiment, in degrees Celsius
    Q_10             : Q10 factor for microbial activity in degradation
    OUTPUT:
    dydt             : time derivative for values R and z'''
    R, z = y
    T = T_profile(z)
    dydt = [  -4*np.pi*rho_dry*np.power(R,3-chi)*(gamma*np.power(Q_10, (T-T_exp)/10.0))/(E*chi),B*np.power(R,omega)]
    return dydt

##

#### Used by Figure4, FigureS7, FigureS8, FigureS9, FigureS10, FigureS11 ####
def solRz_m_flow_ref(y0, tspan, gamma, B, delta, omega, E, chi, rho_dry):
    '''function returning the time evolution of radius and depth for a particle of
    given initial radius and position, on specific time points,
    in the framework of Alldredge sinking speed and Alldredge emprical law for dry mass with size
     without feedback of flow on degradation (reference case)

    INPUTS:
    y0 = [R_0,z_0]   : initial radius and position, both in meters
    tspan            : start and end of the integration, tuple
    gamma            : radius shrinking rate without flow
    B                : coefficient in Alldredge fit
    delta            : exponent in Sherwood law with Reynolds for Re>0.1
    omega            : exponent in the power law for Alldredge speed as function of radius
    E                : pre-factor in the power law for dry mass as function of size (Alldredge L&O 1988)
    chi              : exponent in the power law for dry mass as function of size (Alldredge L&O 1988)
    rho_dry          : dry mass per unit volume of alginate particles
    OUTPUT:
    sol              : sol[:,0] gives the value of R(t) on time points in array t
                       sol[:,1] gives the value of z(t) on time points in array t
    '''

    return integrate.solve_ivp(fun=lambda t, y :ODE_Rz_mref(t,y,gamma, B, delta, omega, E, chi, rho_dry), 
           t_span=tspan, y0=y0, dense_output=True)

#### Used in FigureS11 ####
def solRz_m_flow_ref_T(y0, tspan, gamma, B_prime, delta, omega, E, chi, rho_dry):
    '''function returning the time evolution of radius and depth for a particle of
    given initial radius and position, on specific time points,
    in the framework of Alldredge sinking speed and Alldredge emprical law for dry mass with size
     without feedback of flow on degradation (reference case) but with temperature variation impacting viscosity

    INPUTS:
    y0 = [R_0,z_0]   : initial radius and position, both in meters
    tspan            : start and end of the integration, tuple
    gamma            : radius shrinking rate without flow
    B_prime          : coefficient in Alldredge fit of sinking speed multiplied by dynamic viscosity in Alldredge conditions
    delta            : exponent in Sherwood law with Reynolds for Re>0.1
    omega            : exponent in the power law for Alldredge speed as function of radius
    E                : pre-factor in the power law for dry mass as function of size (Alldredge L&O 1988)
    chi              : exponent in the power law for dry mass as function of size (Alldredge L&O 1988)
    rho_dry          : dry mass per unit volume of alginate particles
    OUTPUT:
    sol              : sol[:,0] gives the value of R(t) on time points in array t
                       sol[:,1] gives the value of z(t) on time points in array t
    '''

    return integrate.solve_ivp(fun=lambda t, y :ODE_Rz_mref_T(t,y,gamma, B_prime, delta, omega, E, chi, rho_dry), 
           t_span=tspan, y0=y0, dense_output=True)

#### Used in FigureS10 ####
def solRz_m_flow_ref_Q10(y0, tspan, gamma, B, delta, omega, E, chi, rho_dry, T_exp, Q_10):
    '''function returning the time evolution of radius and depth for a particle of
    given initial radius and position, on specific time points,
    in the framework of Alldredge sinking speed and Alldredge emprical law for dry mass with size
    without feedback of flow on degradation (reference case) and with Q10 modulation of degradation

    INPUTS:
    y0 = [R_0,z_0]   : initial radius and position, both in meters
    tspan            : start and end of the integration, tuple
    gamma            : radius shrinking rate without flow
    B                : coefficient in Alldredge fit
    delta            : exponent in Sherwood law with Reynolds for Re>0.1
    omega            : exponent in the power law for Alldredge speed as function of radius
    E                : pre-factor in the power law for dry mass as function of size (Alldredge L&O 1988)
    chi              : exponent in the power law for dry mass as function of size (Alldredge L&O 1988)
    rho_dry          : dry mass per unit volume of alginate particles
    T_exp            : temperature of experiment, in degrees Celsius
    Q_10             : Q10 factor for microbial activity in degradation
    OUTPUT:
    sol              : sol[:,0] gives the value of R(t) on time points in array t
                       sol[:,1] gives the value of z(t) on time points in array t
    '''

    return integrate.solve_ivp(fun=lambda t, y :ODE_Rz_mref_Q10(t,y,gamma, B, delta, omega, E, chi, rho_dry, T_exp, Q_10), 
           t_span=tspan, y0=y0, dense_output=True)

##

#### Used by Figure4, FigureS7, FigureS8, FigureS9, FigureS10, FigureS11 ####
def solUV_z_m_flow_refbis(z,z_0, R_0, gamma, B, delta, omega, E, chi, rho_dry,R0_ref):
    '''Function returning speed and volume as function of z for a given initial size, for use in the
    integration of the vertical carbon flux.
    This is the case of Alldredge sinking speed, Alldredge mass law and NO feedback of flow on degradation
    INPUTS:
    z                : depth at which speed and volumes are evaluated, in meters
    z_0, R_0         : initial raposition and radius, both in meters
    gamma            : radius shrinking rate without flow
    B                : coefficient in Alldredge fit
    delta            : exponent in Sherwood law with Reynolds for Re>0.1
    omega            : exponent in the power law for Alldredge speed as function of radius
    E                : pre-factor in the power law for dry mass as function of size (Alldredge 1988)
    chi              : exponent in the power law for dry mass as function of size (Alldredge 1988)
    rho_dry          : dry mass per unit volume of alginate particles
    R0_ref           : reference size of particle corresponding to experiments for making time points
    OUTPUT:
    Uzf, Vzf         : Uzf gives the value of U(z) for a particle of initial radius R_0 at z_0 
                       Vzf gives the value of V(z) for a particle of initial radius R_0 at z_0 
                       '''

    # in the flow feedback case, degradation is faster than without flow, so
    t_max = 5*R0_ref/gamma # is an upper bound for the total degradation
    plot = 0 # switch for optional plots for debugging
    Nt = 10*501 # time points chosen, could be optimised
    t = np.linspace(0, t_max, Nt)
    tspan = (0, t_max)
    y0 = [R_0, z_0]
    solRz_m = solRz_m_flow_ref(y0, tspan, gamma, B, delta, omega, E, chi, rho_dry)

    ############################################
    if plot==1: # show intermediate plot to check the code
        print('time solution of dynamics for R0=',R_0)
        # we have modified R_0 above
        print(R_0)
        print(z_0)
        
        solRz_m_dense = solRz_m.sol(t)

        VRmin= 10**9*4*np.pi*0.000125**3/3
        fig = plt.figure(figsize = (3, 3)) 
        gs8 = gridspec.GridSpec(1,2)
        ax14 = fig.add_subplot(gs8[0,0])
        ax15 = fig.add_subplot(gs8[0,1])

        ax14.semilogy(t/(3600*24),10**9*4*np.pi*solRz_m_dense[0,:]**3/3, color = 'r',marker = 'o')
        ax14.semilogy([0,10000],[VRmin,VRmin],'--')
        ax14.set_xlim([0, max(t/(3600*24))])

        ax15.loglog(solRz_m_dense[1,:],10**9*4*np.pi*solRz_m_dense[0,:]**3/3, color = 'r', linestyle = '-', marker = 'd',markerfacecolor='k')
        ax15.loglog(solRz_m.y[1,:],10**9*4*np.pi*solRz_m.y[0,:]**3/3, color = 'b', linestyle = '--', marker = 'o',markerfacecolor='k')
        ax15.loglog([0,2000], [VRmin,VRmin], linestyle = '--', marker = 'o', color = 'r')
        ax15.loglog([870,870],[0,1],linestyle='--')
        ax15.set_xlim([100, 1000])
        ax15.set_ylim([0.1*VRmin, 10**9*4*np.pi*solRz_m_dense[0,0]**3/3])


        plt.show()
        plt.close()
    ############################################

    i = find_nearest(solRz_m.sol(t)[1,:],z)
    # z should be increasing function, so we can interpolate in the following way
    if solRz_m.sol(t)[1,i]>z:
        #interpolate between i-1 and i
        lamb = (z - solRz_m.sol(t)[1,i-1])/(solRz_m.sol(t)[1,i] - solRz_m.sol(t)[1,i-1])
        R = solRz_m.sol(t)[0,i-1]*(1-lamb) + lamb*solRz_m.sol(t)[0,i]
    else:
        # maybe it's the end of the array
        if i==solRz_m.sol(t)[1,:].shape[0]-1:
            R = solRz_m.sol(t)[0,i]
        # maybe we're stationary
        else:
            if (solRz_m.sol(t)[1,i+1] - solRz_m.sol(t)[1,i])==0:
                lamb = 0
            else:
                # interpolate between i and i+1   
                lamb = (z - solRz_m.sol(t)[1,i])/(solRz_m.sol(t)[1,i+1] - solRz_m.sol(t)[1,i])
            R = solRz_m.sol(t)[0,i]*(1-lamb) + lamb*solRz_m.sol(t)[0,i+1]

    Uzf = B*np.power(R,omega)
    Vzf = 4*np.pi*R**3/3.0

    return Uzf, Vzf

#### Used in FigureS11 ####
def solUV_z_m_flow_refbis_T(z,z_0, R_0, gamma, B_prime, delta, omega, E, chi, rho_dry,R0_ref):
    '''Function returning speed and volume as function of z for a given initial size, for use in the
    integration of the vertical carbon flux.
    This is the case of Alldredge sinking speed, Alldredge mass law and NO feedback of flow on degradation
    This function includes effects of water viscosity variation with temperature

    INPUTS:
    z                : depth at which speed and volumes are evaluated, in meters
    z_0, R_0         : initial raposition and radius, both in meters
    gamma            : radius shrinking rate without flow
    B_prime          : coefficient in Alldredge fit of sinking speed multiplied by dynamic viscosity in Alldredge conditions
    delta            : exponent in Sherwood law with Reynolds for Re>0.1
    omega            : exponent in the power law for Alldredge speed as function of radius
    E                : pre-factor in the power law for dry mass as function of size (Alldredge 1988)
    chi              : exponent in the power law for dry mass as function of size (Alldredge 1988)
    rho_dry          : dry mass per unit volume of alginate particles
    R0_ref           : reference size of particle corresponding to experiments for choosing time points
    OUTPUT:
    Uzf, Vzf         : Uzf gives the value of U(z) for a particle of initial radius R_0 at z_0 
                       Vzf gives the value of V(z) for a particle of initial radius R_0 at z_0 
                       '''

    t_max = 5*R0_ref/gamma # is an upper bound for the total degradation
    plot = 0 # switch for intermediate plots
    Nt = 10*501 # time points chosen, could be optimised
    t = np.linspace(0, t_max, Nt)
    tspan = (0, t_max)
    y0 = [R_0, z_0]
    solRz_m = solRz_m_flow_ref_T(y0, tspan, gamma, B_prime, delta, omega, E, chi, rho_dry)

    ############################################
    if plot==1: # show intermediate plot to check the code
        print('time solution of dynamics for R0=',R_0)
        # we have modified R_0 above
        print(R_0)
        print(z_0)
        
        solRz_m_dense = solRz_m.sol(t)

        VRmin= 10**9*4*np.pi*0.000125**3/3
        fig = plt.figure(figsize = (3, 3)) 
        gs8 = gridspec.GridSpec(1,2)
        ax14 = fig.add_subplot(gs8[0,0])
        ax15 = fig.add_subplot(gs8[0,1])

        ax14.semilogy(t/(3600*24),10**9*4*np.pi*solRz_m_dense[0,:]**3/3, color = 'r',marker = 'o')
        ax14.semilogy([0,10000],[VRmin,VRmin],'--')
        ax14.set_xlim([0, max(t/(3600*24))])

        ax15.loglog(solRz_m_dense[1,:],10**9*4*np.pi*solRz_m_dense[0,:]**3/3, color = 'r', linestyle = '-', marker = 'd',markerfacecolor='k')
        ax15.loglog(solRz_m.y[1,:],10**9*4*np.pi*solRz_m.y[0,:]**3/3, color = 'b', linestyle = '--', marker = 'o',markerfacecolor='k')
        ax15.loglog([0,2000], [VRmin,VRmin], linestyle = '--', marker = 'o', color = 'r')
        ax15.loglog([870,870],[0,1],linestyle='--')
        ax15.set_xlim([100, 1000])
        ax15.set_ylim([0.1*VRmin, 10**9*4*np.pi*solRz_m_dense[0,0]**3/3])


        plt.show()
        plt.close()
    ############################################

    i = find_nearest(solRz_m.sol(t)[1,:],z)
    # z should be increasing function, so we can interpolate in the following way
    if solRz_m.sol(t)[1,i]>z:
        #interpolate between i-1 and i
        lamb = (z - solRz_m.sol(t)[1,i-1])/(solRz_m.sol(t)[1,i] - solRz_m.sol(t)[1,i-1])
        R = solRz_m.sol(t)[0,i-1]*(1-lamb) + lamb*solRz_m.sol(t)[0,i]
    else:
        # maybe it's the end of the array
        if i==solRz_m.sol(t)[1,:].shape[0]-1:
            R = solRz_m.sol(t)[0,i]
        # maybe we're stationary
        else:
            if (solRz_m.sol(t)[1,i+1] - solRz_m.sol(t)[1,i])==0:
                lamb = 0
            else:
                # interpolate between i and i+1   
                lamb = (z - solRz_m.sol(t)[1,i])/(solRz_m.sol(t)[1,i+1] - solRz_m.sol(t)[1,i])
            R = solRz_m.sol(t)[0,i]*(1-lamb) + lamb*solRz_m.sol(t)[0,i+1]

    T = T_profile(z)
    muTS1 = muTS(T,33.6/1000)
    Uzf = B_prime*np.power(R,omega)/muTS1
    Vzf = 4*np.pi*R**3/3.0

    return Uzf, Vzf

#### Used in FigureS10 ####
def solUV_z_m_flow_refbis_Q10(z,z_0, R_0, gamma, B, delta, omega, E, chi, rho_dry,R0_ref, T_exp, Q_10):
    '''Function returning speed and volume as function of z for a given initial size, for use in the
    integration of the vertical carbon flux.
    This is the case of Alldredge sinking speed, Alldredge mass law and NO feedback of flow on degradation
    Includes modulation of degradation by temperature.

    INPUTS:
    z                : depth at which speed and volumes are evaluated, in meters
    z_0, R_0         : initial raposition and radius, both in meters
    gamma            : radius shrinking rate without flow
    B                : coefficient in Alldredge fit
    delta            : exponent in Sherwood law with Reynolds for Re>0.1
    omega            : exponent in the power law for Alldredge speed as function of radius
    E                : pre-factor in the power law for dry mass as function of size (Alldredge 1988)
    chi              : exponent in the power law for dry mass as function of size (Alldredge 1988)
    rho_dry          : dry mass per unit volume of alginate particles
    R0_ref           : reference size of particle corresponding to experiments for making time points
    T_exp            : temperature of experiment, in degrees Celsius
    Q_10             : Q10 factor for microbial activity in degradation
    OUTPUT:
    Uzf, Vzf         : Uzf gives the value of U(z) for a particle of initial radius R_0 at z_0 
                       Vzf gives the value of V(z) for a particle of initial radius R_0 at z_0 
                       '''

    t_max = 5*R0_ref/gamma # is an upper bound for the total degradation
    plot = 0 # switch for intermediate plots
    Nt = 10*501 # time points chosen, could be optimised
    t = np.linspace(0, t_max, Nt)
    tspan = (0, t_max)
    y0 = [R_0, z_0]
    solRz_m = solRz_m_flow_ref_Q10(y0, tspan, gamma, B, delta, omega, E, chi, rho_dry, T_exp, Q_10)

    ############################################
    if plot==1: # show intermediate plot to check the code
        print('time solution of dynamics for R0=',R_0)
        # we have modified R_0 above
        print(R_0)
        print(z_0)
        
        solRz_m_dense = solRz_m.sol(t)

        VRmin= 10**9*4*np.pi*0.000125**3/3
        fig = plt.figure(figsize = (3, 3)) 
        gs8 = gridspec.GridSpec(1,2)
        ax14 = fig.add_subplot(gs8[0,0])
        ax15 = fig.add_subplot(gs8[0,1])

        ax14.semilogy(t/(3600*24),10**9*4*np.pi*solRz_m_dense[0,:]**3/3, color = 'r',marker = 'o')
        ax14.semilogy([0,10000],[VRmin,VRmin],'--')
        ax14.set_xlim([0, max(t/(3600*24))])

        ax15.loglog(solRz_m_dense[1,:],10**9*4*np.pi*solRz_m_dense[0,:]**3/3, color = 'r', linestyle = '-', marker = 'd',markerfacecolor='k')
        ax15.loglog(solRz_m.y[1,:],10**9*4*np.pi*solRz_m.y[0,:]**3/3, color = 'b', linestyle = '--', marker = 'o',markerfacecolor='k')
        ax15.loglog([0,2000], [VRmin,VRmin], linestyle = '--', marker = 'o', color = 'r')
        ax15.loglog([870,870],[0,1],linestyle='--')
        ax15.set_xlim([100, 1000])
        ax15.set_ylim([0.1*VRmin, 10**9*4*np.pi*solRz_m_dense[0,0]**3/3])


        plt.show()
        plt.close()
    ############################################

    i = find_nearest(solRz_m.sol(t)[1,:],z)
    # z should be increasing function, so we can interpolate in the following way
    if solRz_m.sol(t)[1,i]>z:
        #interpolate between i-1 and i
        lamb = (z - solRz_m.sol(t)[1,i-1])/(solRz_m.sol(t)[1,i] - solRz_m.sol(t)[1,i-1])
        R = solRz_m.sol(t)[0,i-1]*(1-lamb) + lamb*solRz_m.sol(t)[0,i]
    else:
        # maybe it's the end of the array
        if i==solRz_m.sol(t)[1,:].shape[0]-1:
            R = solRz_m.sol(t)[0,i]
        # maybe we're stationary
        else:
            if (solRz_m.sol(t)[1,i+1] - solRz_m.sol(t)[1,i])==0:
                lamb = 0
            else:
                # interpolate between i and i+1   
                lamb = (z - solRz_m.sol(t)[1,i])/(solRz_m.sol(t)[1,i+1] - solRz_m.sol(t)[1,i])
            R = solRz_m.sol(t)[0,i]*(1-lamb) + lamb*solRz_m.sol(t)[0,i+1]

    Uzf = B*np.power(R,omega)
    Vzf = 4*np.pi*R**3/3.0

    return Uzf, Vzf

##

#### Used by Figure4, FigureS7, FigureS8, FigureS9, FigureS10, FigureS11 ####
def phiA_m_flow_refbis(R_0, z, z_0, B, omega, gamma,  E, chi, rho_dry, delta, C, R_l, R_g, beta,R0_ref):
    '''Function for the integrand of the flux in the Alldredge sinking speed and mass framework with NO feedback of flow (reference case)
    INPUTS:
    R_0              : initial radius of particle in meters
    z                : depth at which speed and volumes are evaluated, in meters
    z_0              : initial position in meters
    B                : coefficient in Alldredge fit
    omega            : exponent in the power law for Alldredge speed as function of radius
    gamma            : radius shrinking rate without flow
    delta            : exponent in Sherwood law with Reynolds for Re>0.1
    E                : pre-factor in the power law for dry mass as function of size (Alldredge 1988)
    chi              : exponent in the power law for dry mass as function of size (Alldredge 1988)
    rho_dry          : dry mass per unit volume of alginate particles
    C                : total abundance of particles, in m-3
    R_l              : lower size cut-off in meter
    R_g              : greater size cut-off in meter
    beta             : exponent of the size distribution
    OUTPUT:
    phiA_m_flow        : integrand value for calculating vertical flux of dry mass
                       phiA_flow(z, R_0) = c*U*m
                       '''
    c_zR0 = C*P_R0(R_0,R_l,R_g,beta)

    U_zR0, V_zR0 = solUV_z_m_flow_refbis(z,z_0, R_0, gamma, B, delta, omega, E, chi, rho_dry,R0_ref) 

    if np.isnan(U_zR0) or np.isnan(V_zR0):
        U_zR0 = 0
        V_zR0 = 0
    m_zR0 = E*np.power(3*V_zR0/(4*np.pi),chi/3.0)

    # by conservation of numbers, the initial speed goes in the integrand
    U_R0 = U_A(R_0, B, omega)

    return c_zR0*U_R0*m_zR0

#### Used in FigureS11 ####
def phiA_m_flow_refbis_T(R_0, z, z_0, B_prime, omega, gamma,  E, chi, rho_dry, delta, C, R_l, R_g, beta,R0_ref):
    '''Function for the integrand of the flux in the Alldredge sinking speed and mass framework with NO feedback of flow (reference case)
    This function includes effects of water viscosity variation with temperature

    INPUTS:
    R_0              : initial radius of particle in meters
    z                : depth at which speed and volumes are evaluated, in meters
    z_0              : initial position in meters
    B_prime          : coefficient in Alldredge fit of sinking speed multiplied by dynamic viscosity in Alldredge conditions
    omega            : exponent in the power law for Alldredge speed as function of radius
    gamma            : radius shrinking rate without flow
    delta            : exponent in Sherwood law with Reynolds for Re>0.1
    E                : pre-factor in the power law for dry mass as function of size (Alldredge 1988)
    chi              : exponent in the power law for dry mass as function of size (Alldredge 1988)
    rho_dry          : dry mass per unit volume of alginate particles
    C                : total abundance of particles, in m-3
    R_l              : lower size cut-off in meter
    R_g              : greater size cut-off in meter
    beta             : exponent of the size distribution
    OUTPUT:
    phiA_m_flow        : integrand value for calculating vertical flux of dry mass
                       phiA_flow(z, R_0) = c*U*m
                       '''
    c_zR0 = C*P_R0(R_0,R_l,R_g,beta)

    U_zR0, V_zR0 = solUV_z_m_flow_refbis_T(z,z_0, R_0, gamma, B_prime, delta, omega, E, chi, rho_dry, R0_ref) 

    if np.isnan(U_zR0) or np.isnan(V_zR0):
        U_zR0 = 0
        V_zR0 = 0
    m_zR0 = E*np.power(3*V_zR0/(4*np.pi),chi/3.0)

    T = T_profile(z_0)
    muTS1 = muTS(T,33.6/1000)

    # by conservation of numbers, the initial speed goes in the integrand.
    U_R0 = U_A(R_0, B_prime/muTS1, omega)

    return c_zR0*U_R0*m_zR0

#### Used in FigureS10 ####
def phiA_m_flow_refbis_Q10(R_0, z, z_0, B, omega, gamma,  E, chi, rho_dry, delta, C, R_l, R_g, beta, R0_ref, T_exp, Q_10):
    '''Function for the integrand of the flux in the Alldredge sinking speed and mass framework with NO feedback of flow (reference case)
    Includes modulation of degradation rate by temperature.

    INPUTS:
    R_0              : initial radius of particle in meters
    z                : depth at which speed and volumes are evaluated, in meters
    z_0              : initial position in meters
    B                : coefficient in Alldredge fit
    omega            : exponent in the power law for Alldredge speed as function of radius
    gamma            : radius shrinking rate without flow
    delta            : exponent in Sherwood law with Reynolds for Re>0.1
    E                : pre-factor in the power law for dry mass as function of size (Alldredge 1988)
    chi              : exponent in the power law for dry mass as function of size (Alldredge 1988)
    rho_dry          : dry mass per unit volume of alginate particles
    C                : total abundance of particles, in m-3
    R_l              : lower size cut-off in meter
    R_g              : greater size cut-off in meter
    beta             : exponent of the size distribution
    T_exp            : temperature of experiment, in degrees Celsius
    Q_10             : Q10 factor for microbial activity in degradation
    OUTPUT:
    phiA_m_flow        : integrand value for calculating vertical flux of dry mass
                       phiA_flow(z, R_0) = c*U*m
                       '''
    c_zR0 = C*P_R0(R_0,R_l,R_g,beta)

    U_zR0, V_zR0 = solUV_z_m_flow_refbis_Q10(z,z_0, R_0, gamma, B, delta, omega, E, chi, rho_dry,R0_ref, T_exp, Q_10) 

    if np.isnan(U_zR0) or np.isnan(V_zR0):
        U_zR0 = 0
        V_zR0 = 0
    m_zR0 = E*np.power(3*V_zR0/(4*np.pi),chi/3.0)

    # by conservation of numbers, the initial speed goes in the integrand.
    U_R0 = U_A(R_0, B, omega)

    return c_zR0*U_R0*m_zR0

##

#### used by Figure4, FigureS7, FigureS8, FigureS9, FigureS10, FigureS11 ####
def FA_z_m_flow_refbis(z, z_0, B, omega, gamma, E, chi, rho_dry, delta, C, R_l, R_g, beta,R0_ref):
    '''Function returning flux of dry mass at a given depth z for all particles radii R_0
    in the Alldredge sinking speed fit case, mass fit case, with NO feedback of flow on degradation
    INPUTS:
    z                : depth at which speed and volumes are evaluated, in meters
    z_0              : initial position in meters
    B                : coefficient in Alldredge fit
    omega            : exponent in the power law for Alldredge speed as function of radius
    gamma            : radius shrinking rate without flow
    E                : pre-factor in the power law for dry mass as function of size (Alldredge 1988)
    chi              : exponent in the power law for dry mass as function of size (Alldredge 1988)
    rho_dry          : dry mass per unit volume of alginate particles
    delta            : exponent in Sherwood law with Reynolds for Re>0.1
    C                : total abundance of particles, in m-3
    R_l              : lower size cut-off in meter
    R_g              : greater size cut-off in meter
    beta             : exponent of the size distribution
    OUTPUT:
    F(z), in kg per m2 per s'''

    result = integrate.quad(phiA_m_flow_refbis, R_l, R_g, args=(z, z_0, B, omega, gamma, E, chi, rho_dry,delta, C, R_l, R_g, beta,R0_ref))
    return result[0]

#### Used in FigureS11 ####
def FA_z_m_flow_refbis_T(z, z_0, B_prime, omega, gamma, E, chi, rho_dry, delta, C, R_l, R_g, beta,R0_ref):
    '''Function returning flux of dry mass at a given depth z for all particles radii R_0
    in the Alldredge sinking speed fit case, mass fit case, with NO feedback of flow on degradation
    This function includes effects of water viscosity variation with temperature

    INPUTS:
    z                : depth at which speed and volumes are evaluated, in meters
    z_0              : initial position in meters
    B_prime          : coefficient in Alldredge fit of sinking speed multiplied by dynamic viscosity in Alldredge conditions
    omega            : exponent in the power law for Alldredge speed as function of radius
    gamma            : radius shrinking rate without flow
    E                : pre-factor in the power law for dry mass as function of size (Alldredge 1988)
    chi              : exponent in the power law for dry mass as function of size (Alldredge 1988)
    rho_dry          : dry mass per unit volume of alginate particles
    delta            : exponent in Sherwood law with Reynolds for Re>0.1
    C                : total abundance of particles, in m-3
    R_l              : lower size cut-off in meter
    R_g              : greater size cut-off in meter
    beta             : exponent of the size distribution
    OUTPUT:
    F(z), in kg per m2 per s'''

    result = integrate.quad(phiA_m_flow_refbis_T, R_l, R_g, args=(z, z_0, B_prime, omega, gamma, E, chi, rho_dry,delta, C, R_l, R_g, beta,R0_ref))
    return result[0]

#### Used in FigureS10 ####
def FA_z_m_flow_refbis_Q10(z, z_0, B, omega, gamma, E, chi, rho_dry, delta, C, R_l, R_g, beta,R0_ref, T_exp, Q_10):
    '''Function returning flux of dry mass at a given depth z for all particles radii R_0
    in the Alldredge sinking speed fit case, mass fit case, with NO feedback of flow on degradation
    Includes modulation of degradation rate by temperature.

    INPUTS:
    z                : depth at which speed and volumes are evaluated, in meters
    z_0              : initial position in meters
    B                : coefficient in Alldredge fit
    omega            : exponent in the power law for Alldredge speed as function of radius
    gamma            : radius shrinking rate without flow
    E                : pre-factor in the power law for dry mass as function of size (Alldredge 1988)
    chi              : exponent in the power law for dry mass as function of size (Alldredge 1988)
    rho_dry          : dry mass per unit volume of alginate particles
    delta            : exponent in Sherwood law with Reynolds for Re>0.1
    C                : total abundance of particles, in m-3
    R_l              : lower size cut-off in meter
    R_g              : greater size cut-off in meter
    beta             : exponent of the size distribution
    T_exp            : temperature of experiment, in degrees Celsius
    Q_10             : Q10 factor for microbial activity in degradation
    OUTPUT:
    F(z), in kg per m2 per s'''

    result = integrate.quad(phiA_m_flow_refbis_Q10, R_l, R_g, args=(z, z_0, B, omega, gamma, E, chi, rho_dry,delta, C, R_l, R_g, beta,R0_ref, T_exp, Q_10))
    return result[0]


####### Integration functions for the coupled case #######

#### used by Figure4, FigureS7, FigureS8, FigureS9, FigureS10, FigureS11 ####
def ODE_Rz_m(t, y, gamma, gamma2, B, delta, omega, E, chi, rho_dry):
    '''Function returning the instantaneous derivative of radius R(t) and depth z(t) for a particle of given size,
        in the framework of Alldredge sinking speed and Alldredge emprical law for dry mass with size,
        with feedback of flow speed on degradation rate
    INPUTS:
    y                : solution R, z for which we solve
    B                : coefficient in Alldredge fit of sinking speed
    omega            : exponent in the power law for Alldredge speed as function of radius
    gamma            : radius shrinking rate without flow
    gamma2           : shrinking rate coefficient in presence of flow (see notes)
    delta            : exponent in Sherwood law with Reynolds for Re>0.1
    E                : pre-factor in the power law for dry mass as function of size (Alldredge 1988)
    chi              : exponent in the power law for dry mass as function of size (Alldredge 1988)
    rho_dry          : dry mass per unit volume of alginate particles
    OUTPUT:
    dydt             : time derivative for values R and z'''
    R, z = y
    dydt = [  -4*np.pi*rho_dry*np.power(R,3-chi)*(gamma + gamma2*np.power(B,delta)*np.power(R,delta*(1.0 + omega)) )/(E*chi),B*np.power(R,omega)]
    return dydt

#### Used in FigureS11 ####
def ODE_Rz_m_T(t, y, gamma, gamma1, B_prime, D, delta, omega, E, chi, rho_dry):
    '''Function returning the instantaneous derivative of radius R(t) and depth z(t) for a particle of given size,
        in the framework of Alldredge sinking speed and Alldredge emprical law for dry mass with size,
        with feedback of flow speed on degradation rate and varying temperature.
        The varying temperature and associated viscosity change impacts both the sinking law and the Sherwood expression.
    INPUTS:
    y                : solution R, z for which we solve
    B_prime          : coefficient in Alldredge fit of sinking speed multiplied by dynamic viscosity in Alldredge conditions
    omega            : exponent in the power law for Alldredge speed as function of radius
    gamma            : radius shrinking rate without flow
    gamma1           : raw shrinking rate coefficient in presence of flow (see notes)
    D                : diffusivity of studied compound
    delta            : exponent in Sherwood law with Reynolds for Re>0.1
    E                : pre-factor in the power law for dry mass as function of size (Alldredge 1988)
    chi              : exponent in the power law for dry mass as function of size (Alldredge 1988)
    rho_dry          : dry mass per unit volume of alginate particles
    OUTPUT:
    dydt             : time derivative for values R and z'''
    R, z = y
    T = T_profile(z)
    muTS1 = muTS(T, 33.6/1000)
    nuTS1 = nuTS(T, 33.6/1000)
    B = B_prime/muTS1 #we compute the B factor at this temperature
    gamma2 = 0.619*gamma1*(nuTS1/D)**(1.0/3.0)*(2.0/nuTS1)**delta # aggregate coefficient for gamma
    # computed at the given temperature

    dydt = [  -4*np.pi*rho_dry*np.power(R,3-chi)*(gamma + gamma2*np.power(B,delta)*np.power(R,delta*(1.0 + omega)) )/(E*chi),
              B*np.power(R,omega)]
    return dydt

#### Used in FigureS10 ####
def ODE_Rz_m_Q10(t, y, gamma, gamma2, B, delta, omega, E, chi, rho_dry, T_exp, Q_10):
    '''Function returning the instantaneous derivative of radius R(t) and depth z(t) for a particle of given size,
        in the framework of Alldredge sinking speed and Alldredge emprical law for dry mass with size,
        with feedback of flow speed on degradation rate and modulation of degradation rate by temperature
    INPUTS:
    y                : solution R, z for which we solve
    B                : coefficient in Alldredge fit of sinking speed
    omega            : exponent in the power law for Alldredge speed as function of radius
    gamma            : radius shrinking rate without flow
    gamma2           : shrinking rate coefficient in presence of flow (see notes)
    delta            : exponent in Sherwood law with Reynolds for Re>0.1
    E                : pre-factor in the power law for dry mass as function of size (Alldredge 1988)
    chi              : exponent in the power law for dry mass as function of size (Alldredge 1988)
    rho_dry          : dry mass per unit volume of alginate particles
    T_exp            : temperature of experiment, in degrees Celsius
    Q_10             : Q10 factor for microbial activity in degradation
    OUTPUT:
    dydt             : time derivative for values R and z'''
    R, z = y
    T = T_profile(z)
    dydt = [  -4*np.pi*rho_dry*np.power(R,3-chi)*np.power(Q_10,(T-T_exp)/10.0)*(gamma + gamma2*np.power(B,delta)*np.power(R,delta*(1.0 + omega)) )/(E*chi),B*np.power(R,omega)]
    return dydt

##

#### Used by Figure4, FigureS7, FigureS8, FigureS9, FigureS10, FigureS11 ####
def solRz_m_flow(y0, tspan, gamma, gamma2, B, delta, omega, E, chi, rho_dry):
    '''Function returning the time evolution of radius and depth for a particle of
    given initial radius and position, on specific time points,
    in the framework of Alldredge sinking speed and Alldredge emprical law for dry mass with size
    with feedback of flow speed on degradation rate
    INPUTS:
    y0 = [R_0,z_0]   : initial radius and position, both in meters
    tspan            : start and end of the integration, tuple
    gamma            : radius shrinking rate without flow
    gamma2           : shrinking rate coefficient in presence of flow (see notes)
    B                : coefficient in Alldredge fit
    delta            : exponent in Sherwood law with Reynolds for Re>0.1
    omega            : exponent in the power law for Alldredge speed as function of radius
    E                : pre-factor in the power law for dry mass as function of size (Alldredge 1988)
    chi              : exponent in the power law for dry mass as function of size (Alldredge 1988)
    rho_dry          : dry mass per unit volume of alginate particles
    OUTPUT:
    sol              : sol[:,0] gives the value of R(t) on time points in array t
                       sol[:,1] gives the value of z(t) on time points in array t
    '''

    return integrate.solve_ivp(fun=lambda t, y :ODE_Rz_m(t,y,gamma, gamma2, B, delta, omega, E, chi, rho_dry), 
           t_span=tspan, y0=y0, dense_output=True)

#### Used in FigureS11 ####
def solRz_m_flow_T(y0, tspan, gamma, gamma1, B_prime, D, delta, omega, E, chi, rho_dry):
    '''Function returning the time evolution of radius and depth for a particle of
    given initial radius and position, on specific time points,
    in the framework of Alldredge sinking speed and Alldredge emprical law for dry mass with size
    with feedback of flow speed on degradation rate and variation of viscosity with temperature
    INPUTS:
    y0 = [R_0,z_0]   : initial radius and position, both in meters
    tspan            : start and end of the integration, tuple
    gamma            : radius shrinking rate without flow
    gamma1           : raw shrinking rate coefficient in presence of flow (see notes)
    B_prime          : coefficient in Alldredge fit of sinking speed multiplied by dynamic viscosity in Alldredge conditions
    D                : diffusivity of studied compound
    delta            : exponent in Sherwood law with Reynolds for Re>0.1
    omega            : exponent in the power law for Alldredge speed as function of radius
    E                : pre-factor in the power law for dry mass as function of size (Alldredge 1988)
    chi              : exponent in the power law for dry mass as function of size (Alldredge 1988)
    rho_dry          : dry mass per unit volume of alginate particles
    OUTPUT:
    sol              : sol[:,0] gives the value of R(t) on time points in array t
                       sol[:,1] gives the value of z(t) on time points in array t
    '''

    return integrate.solve_ivp(fun=lambda t, y :ODE_Rz_m_T(t,y,gamma, gamma1, B_prime, D, delta, omega, E, chi, rho_dry), 
           t_span=tspan, y0=y0, dense_output=True)

#### Used in FigureS10 ####
def solRz_m_flow_Q10(y0, tspan, gamma, gamma2, B, delta, omega, E, chi, rho_dry, T_exp, Q_10):
    '''Function returning the time evolution of radius and depth for a particle of
    given initial radius and position, on specific time points,
    in the framework of Alldredge sinking speed and Alldredge emprical law for dry mass with size
    with feedback of flow speed on degradation rate and modulation of degradation rate by temperature
    INPUTS:
    y0 = [R_0,z_0]   : initial radius and position, both in meters
    tspan            : start and end of the integration, tuple
    gamma            : radius shrinking rate without flow
    gamma2           : shrinking rate coefficient in presence of flow (see notes)
    B                : coefficient in Alldredge fit
    delta            : exponent in Sherwood law with Reynolds for Re>0.1
    omega            : exponent in the power law for Alldredge speed as function of radius
    E                : pre-factor in the power law for dry mass as function of size (Alldredge 1988)
    chi              : exponent in the power law for dry mass as function of size (Alldredge 1988)
    rho_dry          : dry mass per unit volume of alginate particles made by Uria
    T_exp            : temperature of experiment, in degrees Celsius
    Q_10             : Q10 factor for microbial activity in degradation
    OUTPUT:
    sol              : sol[:,0] gives the value of R(t) on time points in array t
                       sol[:,1] gives the value of z(t) on time points in array t
    '''

    return integrate.solve_ivp(fun=lambda t, y :ODE_Rz_m_Q10(t,y,gamma, gamma2, B, delta, omega, E, chi, rho_dry, T_exp, Q_10), 
           t_span=tspan, y0=y0, dense_output=True)

##

#### Used by Figure4, FigureS7, FigureS8, FigureS9, FigureS10, FigureS11 ####
def solUV_z_m_flowbis(z,z_0, R_0, gamma, gamma2, B, delta, omega, E, chi, rho_dry, R0_ref):
    '''Function returning speed and volume as function of z for a given initial size, for use in the
    integration of the vertical carbon flux.
    This is the case of Alldredge sinking speed, Alldredge mass law and feedback of flow on degradation
    INPUTS:
    z                : depth at which speed and volumes are evaluated, in meters
    z_0, R_0         : initial position and radius, both in meters
    gamma            : radius shrinking rate without flow
    gamma2           : shrinking rate coefficient in presence of flow (see notes)
    B                : coefficient in Alldredge fit
    delta            : exponent in Sherwood law with Reynolds for Re>0.1
    omega            : exponent in the power law for Alldredge speed as function of radius
    E                : pre-factor in the power law for dry mass as function of size (Alldredge 1988)
    chi              : exponent in the power law for dry mass as function of size (Alldredge 1988)
    rho_dry          : dry mass per unit volume of alginate particles
    R0_ref           : reference size of particle corresponding to experiments for making time points
    OUTPUT:
    Uzf, Vzf         : Uzf gives the value of U(z) for a particle of initial radius R_0 at z_0 
                       Vzf gives the value of V(z) for a particle of initial radius R_0 at z_0 
                       '''

    # in the flow feedback case, degradation is faster than without flow, so
    t_max = 5*R0_ref/gamma # is an upper bound for the total degradation
    plot = 0 # switch for optional plots for debugging
    Nt = 10*501 # time points chosen, could be optimised
    t = np.linspace(0, t_max, Nt)
    tspan = (0, t_max)
    y0 = [R_0, z_0]
    solRz_m = solRz_m_flow(y0, tspan, gamma, gamma2, B, delta, omega, E, chi, rho_dry)

    ############################################
    if plot==1: # show intermediate plot to check the code
        print('time solution of dynamics for R0=',R_0)
        # we have modified R_0 above
        print(R_0)
        print(z_0)
        
        solRz_m_dense = solRz_m.sol(t)

        VRmin= 10**9*4*np.pi*0.000125**3/3
        fig = plt.figure(figsize = (3, 3)) 
        gs8 = gridspec.GridSpec(1,2)
        ax14 = fig.add_subplot(gs8[0,0])
        ax15 = fig.add_subplot(gs8[0,1])

        ax14.semilogy(t/(3600*24),10**9*4*np.pi*solRz_m_dense[0,:]**3/3, color = 'r',marker = 'o')
        ax14.semilogy([0,10000],[VRmin,VRmin],'--')
        ax14.set_xlim([0, max(t/(3600*24))])

        ax15.loglog(solRz_m_dense[1,:],10**9*4*np.pi*solRz_m_dense[0,:]**3/3, color = 'r', linestyle = '-', marker = 'd',markerfacecolor='k')
        ax15.loglog(solRz_m.y[1,:],10**9*4*np.pi*solRz_m.y[0,:]**3/3, color = 'b', linestyle = '--', marker = 'o',markerfacecolor='k')
        ax15.loglog([0,2000], [VRmin,VRmin], linestyle = '--', marker = 'o', color = 'r')
        ax15.loglog([870,870],[0,1],linestyle='--')
        ax15.set_xlim([100, 1000])
        ax15.set_ylim([0.1*VRmin, 10**9*4*np.pi*solRz_m_dense[0,0]**3/3])


        plt.show()
        plt.close()
    ############################################

    i = find_nearest(solRz_m.sol(t)[1,:],z)
    # z should be increasing function, so we can interpolate in the following way
    if solRz_m.sol(t)[1,i]>z:
        #interpolate between i-1 and i
        lamb = (z - solRz_m.sol(t)[1,i-1])/(solRz_m.sol(t)[1,i] - solRz_m.sol(t)[1,i-1])
        R = solRz_m.sol(t)[0,i-1]*(1-lamb) + lamb*solRz_m.sol(t)[0,i]
    else:
        # maybe it's the end of the array
        if i==solRz_m.sol(t)[1,:].shape[0]-1:
            R = solRz_m.sol(t)[0,i]
        # maybe we're stationary
        else:
            if (solRz_m.sol(t)[1,i+1] - solRz_m.sol(t)[1,i])==0:
                lamb = 0
            else:
                # interpolate between i and i+1   
                lamb = (z - solRz_m.sol(t)[1,i])/(solRz_m.sol(t)[1,i+1] - solRz_m.sol(t)[1,i])
            R = solRz_m.sol(t)[0,i]*(1-lamb) + lamb*solRz_m.sol(t)[0,i+1]

    Uzf = B*np.power(R,omega)
    Vzf = 4*np.pi*R**3/3.0

    return Uzf, Vzf

#### Used in FigureS11 ####
def solUV_z_m_flowbis_T(z,z_0, R_0, gamma, gamma1, B_prime, D, delta, omega, E, chi, rho_dry,R0_ref):
    '''Function returning speed and volume as function of z for a given initial size, for use in the
    integration of the vertical carbon flux.
    This is the case of Alldredge sinking speed, Alldredge mass law and feedback of flow on degradation
    with on top variation of viscosity with temperature.
    INPUTS:
    z                : depth at which speed and volumes are evaluated, in meters
    z_0, R_0         : initial position and radius, both in meters
    gamma            : radius shrinking rate without flow
    gamma1           : raw shrinking rate coefficient in presence of flow (see notes)
    D                : diffusivity of studied compound
    B_prime          : coefficient in Alldredge fit of sinking speed multiplied by dynamic viscosity in Alldredge conditions
    delta            : exponent in Sherwood law with Reynolds for Re>0.1
    omega            : exponent in the power law for Alldredge speed as function of radius
    E                : pre-factor in the power law for dry mass as function of size (Alldredge 1988)
    chi              : exponent in the power law for dry mass as function of size (Alldredge 1988)
    rho_dry          : dry mass per unit volume of alginate particles
    R0_ref           : reference size of particle corresponding to experiments for making time points
    OUTPUT:
    Uzf, Vzf         : Uzf gives the value of U(z) for a particle of initial radius R_0 at z_0 
                       Vzf gives the value of V(z) for a particle of initial radius R_0 at z_0 
                       '''

    # in the flow feedback case, degradation is faster than without flow, so
    t_max = 5*R0_ref/gamma # is an upper bound for the total degradation
    plot = 0 # switch for optional plot for debugging
    Nt = 10*501 # time points chosen, could be optimised
    t = np.linspace(0, t_max, Nt)
    tspan = (0, t_max)
    y0 = [R_0, z_0]
    solRz_m = solRz_m_flow_T(y0, tspan, gamma, gamma1, B_prime, D, delta, omega, E, chi, rho_dry)

    ############################################
    if plot==1: # show intermediate plot to check the code
        print('time solution of dynamics for R0=',R_0)
        # we have modified R_0 above
        print(R_0)
        print(z_0)
        
        solRz_m_dense = solRz_m.sol(t)

        VRmin= 10**9*4*np.pi*0.000125**3/3
        fig = plt.figure(figsize = (3, 3)) 
        gs8 = gridspec.GridSpec(1,2)
        ax14 = fig.add_subplot(gs8[0,0])
        ax15 = fig.add_subplot(gs8[0,1])

        ax14.semilogy(t/(3600*24),10**9*4*np.pi*solRz_m_dense[0,:]**3/3, color = 'r',marker = 'o')
        ax14.semilogy([0,10000],[VRmin,VRmin],'--')
        ax14.set_xlim([0, max(t/(3600*24))])

        ax15.loglog(solRz_m_dense[1,:],10**9*4*np.pi*solRz_m_dense[0,:]**3/3, color = 'r', linestyle = '-', marker = 'd',markerfacecolor='k')
        ax15.loglog(solRz_m.y[1,:],10**9*4*np.pi*solRz_m.y[0,:]**3/3, color = 'b', linestyle = '--', marker = 'o',markerfacecolor='k')
        ax15.loglog([0,2000], [VRmin,VRmin], linestyle = '--', marker = 'o', color = 'r')
        ax15.loglog([870,870],[0,1],linestyle='--')
        ax15.set_xlim([100, 1000])
        ax15.set_ylim([0.1*VRmin, 10**9*4*np.pi*solRz_m_dense[0,0]**3/3])


        plt.show()
        plt.close()
    ############################################

    i = find_nearest(solRz_m.sol(t)[1,:],z)
    # z should be increasing function, so we can interpolate in the following way
    if solRz_m.sol(t)[1,i]>z:
        #interpolate between i-1 and i
        lamb = (z - solRz_m.sol(t)[1,i-1])/(solRz_m.sol(t)[1,i] - solRz_m.sol(t)[1,i-1])
        R = solRz_m.sol(t)[0,i-1]*(1-lamb) + lamb*solRz_m.sol(t)[0,i]
    else:
        # maybe it's the end of the array
        if i==solRz_m.sol(t)[1,:].shape[0]-1:
            R = solRz_m.sol(t)[0,i]
        # maybe we're stationary
        else:
            if (solRz_m.sol(t)[1,i+1] - solRz_m.sol(t)[1,i])==0:
                lamb = 0
            else:
                # interpolate between i and i+1   
                lamb = (z - solRz_m.sol(t)[1,i])/(solRz_m.sol(t)[1,i+1] - solRz_m.sol(t)[1,i])
            R = solRz_m.sol(t)[0,i]*(1-lamb) + lamb*solRz_m.sol(t)[0,i+1]

    T = T_profile(z)
    muTS1 = muTS(T,33.6/1000)
    Uzf = B_prime*np.power(R,omega)/muTS1
    Vzf = 4*np.pi*R**3/3.0

    return Uzf, Vzf

#### Used in FigureS10 ####
def solUV_z_m_flowbis_Q10(z,z_0, R_0, gamma, gamma2, B, delta, omega, E, chi, rho_dry,R0_ref, T_exp, Q_10):
    '''Function returning speed and volume as function of z for a given initial size, for use in the
    integration of the vertical carbon flux.
    This is the case of Alldredge sinking speed, Alldredge mass law and feedback of flow on degradation
    and modulation of degradation by temperature
    INPUTS:
    z                : depth at which speed and volumes are evaluated, in meters
    z_0, R_0         : initial position and radius, both in meters
    gamma            : radius shrinking rate without flow
    gamma2           : shrinking rate coefficient in presence of flow (see notes)
    B                : coefficient in Alldredge fit
    delta            : exponent in Sherwood law with Reynolds for Re>0.1
    omega            : exponent in the power law for Alldredge speed as function of radius
    E                : pre-factor in the power law for dry mass as function of size (Alldredge 1988)
    chi              : exponent in the power law for dry mass as function of size (Alldredge 1988)
    rho_dry          : dry mass per unit volume of alginate particles
    R0_ref           : reference size of particle corresponding to experiments for making time points
    T_exp            : temperature of experiment, in degrees Celsius
    Q_10             : Q10 factor for microbial activity in degradation
    OUTPUT:
    Uzf, Vzf         : Uzf gives the value of U(z) for a particle of initial radius R_0 at z_0 
                       Vzf gives the value of V(z) for a particle of initial radius R_0 at z_0 
                       '''

    # in the flow feedback case, degradation is faster than without flow, so
    t_max = 5*R0_ref/gamma # is an upper bound for the total degradation
    plot = 0 # switch for optional plot for debugging
    Nt = 10*501 # time points chosen, could be optimised
    t = np.linspace(0, t_max, Nt)
    tspan = (0, t_max)
    y0 = [R_0, z_0]
    solRz_m = solRz_m_flow_Q10(y0, tspan, gamma, gamma2, B, delta, omega, E, chi, rho_dry, T_exp, Q_10)

    ############################################
    if plot==1: # show intermediate plot to check the code
        print('time solution of dynamics for R0=',R_0)
        # we have modified R_0 above
        print(R_0)
        print(z_0)
        
        solRz_m_dense = solRz_m.sol(t)

        VRmin= 10**9*4*np.pi*0.000125**3/3
        fig = plt.figure(figsize = (3, 3)) 
        gs8 = gridspec.GridSpec(1,2)
        ax14 = fig.add_subplot(gs8[0,0])
        ax15 = fig.add_subplot(gs8[0,1])

        ax14.semilogy(t/(3600*24),10**9*4*np.pi*solRz_m_dense[0,:]**3/3, color = 'r',marker = 'o')
        ax14.semilogy([0,10000],[VRmin,VRmin],'--')
        ax14.set_xlim([0, max(t/(3600*24))])

        ax15.loglog(solRz_m_dense[1,:],10**9*4*np.pi*solRz_m_dense[0,:]**3/3, color = 'r', linestyle = '-', marker = 'd',markerfacecolor='k')
        ax15.loglog(solRz_m.y[1,:],10**9*4*np.pi*solRz_m.y[0,:]**3/3, color = 'b', linestyle = '--', marker = 'o',markerfacecolor='k')
        ax15.loglog([0,2000], [VRmin,VRmin], linestyle = '--', marker = 'o', color = 'r')
        ax15.loglog([870,870],[0,1],linestyle='--')
        ax15.set_xlim([100, 1000])
        ax15.set_ylim([0.1*VRmin, 10**9*4*np.pi*solRz_m_dense[0,0]**3/3])

        plt.show()
        plt.close()
    ############################################

    i = find_nearest(solRz_m.sol(t)[1,:],z)
    # z should be increasing function, so we can interpolate in the following way
    if solRz_m.sol(t)[1,i]>z:
        #interpolate between i-1 and i
        lamb = (z - solRz_m.sol(t)[1,i-1])/(solRz_m.sol(t)[1,i] - solRz_m.sol(t)[1,i-1])
        R = solRz_m.sol(t)[0,i-1]*(1-lamb) + lamb*solRz_m.sol(t)[0,i]
    else:
        # maybe it's the end of the array
        if i==solRz_m.sol(t)[1,:].shape[0]-1:
            R = solRz_m.sol(t)[0,i]
        # maybe we're stationary
        else:
            if (solRz_m.sol(t)[1,i+1] - solRz_m.sol(t)[1,i])==0:
                lamb = 0
            else:
                # interpolate between i and i+1   
                lamb = (z - solRz_m.sol(t)[1,i])/(solRz_m.sol(t)[1,i+1] - solRz_m.sol(t)[1,i])
            R = solRz_m.sol(t)[0,i]*(1-lamb) + lamb*solRz_m.sol(t)[0,i+1]

    Uzf = B*np.power(R,omega)
    Vzf = 4*np.pi*R**3/3.0

    return Uzf, Vzf

##

#### Used by Figure4, FigureS7, FigureS8, FigureS9, FigureS10, FigureS11 ####
def phiA_m_flowbis(R_0, z, z_0, B, omega, gamma, gamma2, E, chi, rho_dry, delta, C, R_l, R_g, beta,R0_ref):
    '''Function for the integrand of the flux in the Alldredge sinking speed and mass/feedback of flow case
    INPUTS:
    R_0              : initial radius of particle in meters
    z                : depth at which speed and volumes are evaluated, in meters
    z_0              : initial position in meters
    B                : coefficient in Alldredge fit
    omega            : exponent in the power law for Alldredge speed as function of radius
    gamma            : radius shrinking rate without flow
    gamma2           : shrinking rate coefficient in presence of flow (see notes)
    delta            : exponent in Sherwood law with Reynolds for Re>0.1
    E                : pre-factor in the power law for dry mass as function of size (Alldredge 1988)
    chi              : exponent in the power law for dry mass as function of size (Alldredge 1988)
    rho_dry          : dry mass per unit volume of alginate particles
    C                : total abundance of particles, in m-3
    R_l              : lower size cut-off in meter
    R_g              : greater size cut-off in meter
    beta             : exponent of the size distribution
    R0_ref           : reference size for time vector build

    OUTPUT:
    phiA_m_flow        : integrand value for calculating vertical flux of dry mass
                       phiA_flow(z, R_0) = c*U*m
                       '''
    c_zR0 = C*P_R0(R_0,R_l,R_g,beta)

    U_zR0, V_zR0 = solUV_z_m_flowbis(z,z_0, R_0, gamma, gamma2, B, delta, omega, E, chi, rho_dry, R0_ref) 

    if np.isnan(U_zR0) or np.isnan(V_zR0):
        U_zR0 = 0
        V_zR0 = 0
    m_zR0 = E*np.power(3*V_zR0/(4*np.pi),chi/3.0)
    # by conservation of numbers, the initial speed goes in the integrand.
    U_R0 = U_A(R_0, B, omega)

    return c_zR0*U_R0*m_zR0

#### Used in FigureS11 ####
def phiA_m_flowbis_T(R_0, z, z_0, B_prime, omega, gamma, gamma1, E, chi, rho_dry, D, delta, C, R_l, R_g, beta,R0_ref):
    '''Function for the integrand of the flux in the Alldredge sinking speed and mass/feedback of flow case
    with variation of viscosity with temperature
    INPUTS:
    R_0              : initial radius of particle in meters
    z                : depth at which speed and volumes are evaluated, in meters
    z_0              : initial position in meters
    B_prime          : coefficient in Alldredge fit of sinking speed multiplied by dynamic viscosity in Alldredge conditions
    omega            : exponent in the power law for Alldredge speed as function of radius
    gamma            : radius shrinking rate without flow
    gamma1           : raw shrinking rate coefficient in presence of flow (see notes)
    D                : diffusivity of studied compound
    delta            : exponent in Sherwood law with Reynolds for Re>0.1
    E                : pre-factor in the power law for dry mass as function of size (Alldredge 1988)
    chi              : exponent in the power law for dry mass as function of size (Alldredge 1988)
    rho_dry          : dry mass per unit volume of alginate particles
    C                : total abundance of particles, in m-3
    R_l              : lower size cut-off in meter
    R_g              : greater size cut-off in meter
    beta             : exponent of the size distribution
    R0_ref           : reference size for time vector build

    OUTPUT:
    phiA_m_flow        : integrand value for calculating vertical flux of dry mass
                       phiA_flow(z, R_0) = c*U*m
                       '''
    c_zR0 = C*P_R0(R_0,R_l,R_g,beta)

    U_zR0, V_zR0 = solUV_z_m_flowbis_T(z, z_0, R_0, gamma, gamma1, B_prime, D, delta, omega, E, chi, rho_dry,R0_ref) 

    if np.isnan(U_zR0) or np.isnan(V_zR0):
        U_zR0 = 0
        V_zR0 = 0
    m_zR0 = E*np.power(3*V_zR0/(4*np.pi),chi/3.0)
    # by conservation of numbers, the initial speed goes in the integrand.
    T = T_profile(z_0)
    muTS1 = muTS(T,33.6/1000)

    U_R0 = U_A(R_0, B_prime/muTS1, omega)

    return c_zR0*U_R0*m_zR0

#### Used in FigureS10 ####
def phiA_m_flowbis_Q10(R_0, z, z_0, B, omega, gamma, gamma2, E, chi, rho_dry, delta, C, R_l, R_g, beta,R0_ref, T_exp, Q_10):
    '''Function for the integrand of the flux in the Alldredge sinking speed and mass/feedback of flow case
    and modulation of degradation by temperature
    INPUTS:
    R_0              : initial radius of particle in meters
    z                : depth at which speed and volumes are evaluated, in meters
    z_0              : initial position in meters
    B                : coefficient in Alldredge fit
    omega            : exponent in the power law for Alldredge speed as function of radius
    gamma            : radius shrinking rate without flow
    gamma2           : shrinking rate coefficient in presence of flow (see notes)
    delta            : exponent in Sherwood law with Reynolds for Re>0.1
    E                : pre-factor in the power law for dry mass as function of size (Alldredge 1988)
    chi              : exponent in the power law for dry mass as function of size (Alldredge 1988)
    rho_dry          : dry mass per unit volume of alginate particles
    C                : total abundance of particles, in m-3
    R_l              : lower size cut-off in meter
    R_g              : greater size cut-off in meter
    beta             : exponent of the size distribution
    R0_ref           : reference size for time vector build
    T_exp            : temperature of experiment, in degrees Celsius
    Q_10             : Q10 factor for microbial activity in degradation

    OUTPUT:
    phiA_m_flow        : integrand value for calculating vertical flux of dry mass
                       phiA_flow(z, R_0) = c*U*m
                       '''
    c_zR0 = C*P_R0(R_0,R_l,R_g,beta)

    U_zR0, V_zR0 = solUV_z_m_flowbis_Q10(z,z_0, R_0, gamma, gamma2, B, delta, omega, E, chi, rho_dry,R0_ref, T_exp, Q_10) 

    if np.isnan(U_zR0) or np.isnan(V_zR0):
        U_zR0 = 0
        V_zR0 = 0
    m_zR0 = E*np.power(3*V_zR0/(4*np.pi),chi/3.0)
    # by conservation of numbers, the initial speed goes in the integrand.
    U_R0 = U_A(R_0, B, omega)

    return c_zR0*U_R0*m_zR0

##

#### Used by Figure4, FigureS7, FigureS8, FigureS9, FigureS10, FigureS11 ####
def FA_z_m_flowbis(z, z_0, B, omega, gamma, gamma2, E, chi, rho_dry, delta, C, R_l, R_g, beta,R0_ref):
    '''Function returning flux of dry mass at a given depth z for all particles radii R_0
    in the Alldredge sinking speed fit case, mass fit case, with feedback of flow on degradation
    INPUTS:
    z                : depth at which speed and volumes are evaluated, in meters
    z_0              : initial position in meters
    B                : coefficient in Alldredge fit
    omega            : exponent in the power law for Alldredge speed as function of radius
    gamma            : radius shrinking rate without flow
    gamma2           : shrinking rate coefficient in presence of flow (see notes)
    E                : pre-factor in the power law for dry mass as function of size (Alldredge 1988)
    chi              : exponent in the power law for dry mass as function of size (Alldredge 1988)
    rho_dry          : dry mass per unit volume of alginate particles
    delta            : exponent in Sherwood law with Reynolds for Re>0.1
    C                : total abundance of particles, in m-3
    R_l              : lower size cut-off in meter
    R_g              : greater size cut-off in meter
    beta             : exponent of the size distribution
    R0_ref           : reference size for time vector build
    OUTPUT:
    F(z), in kg per m2 per s'''

    result = integrate.quad(phiA_m_flowbis, R_l, R_g, args=(z, z_0, B, omega, gamma, gamma2, E, chi, rho_dry,delta, C, R_l, R_g, beta, R0_ref))
    return result[0]

#### Used in FigureS11 ####
def FA_z_m_flowbis_T(z, z_0, B_prime, omega, gamma, gamma1, E, chi, rho_dry, D, delta, C, R_l, R_g, beta,R0_ref):
    '''Function returning flux of dry mass at a given depth z for all particles radii R_0
    in the Alldredge sinking speed fit case, mass fit case, with feedback of flow on degradation
    with variation of viscosity with temperature
    INPUTS:
    z                : depth at which speed and volumes are evaluated, in meters
    z_0              : initial position in meters
    B_prime          : coefficient in Alldredge fit of sinking speed multiplied by dynamic viscosity in Alldredge conditions
    omega            : exponent in the power law for Alldredge speed as function of radius
    gamma            : radius shrinking rate without flow
    gamma1           : raw shrinking rate coefficient in presence of flow (see notes)
    E                : pre-factor in the power law for dry mass as function of size (Alldredge 1988)
    chi              : exponent in the power law for dry mass as function of size (Alldredge 1988)
    rho_dry          : dry mass per unit volume of alginate particles
    D                : diffusivity of studied compound
    delta            : exponent in Sherwood law with Reynolds for Re>0.1
    C                : total abundance of particles, in m-3
    R_l              : lower size cut-off in meter
    R_g              : greater size cut-off in meter
    beta             : exponent of the size distribution
    R0_ref           : reference size for time vector build
    OUTPUT:
    F(z), in kg per m2 per s'''

    result = integrate.quad(phiA_m_flowbis_T, R_l, R_g, args=(z, z_0, B_prime, omega, gamma, gamma1, E, chi, rho_dry, D, delta, C, R_l, R_g, beta,R0_ref))
    return result[0]

#### Used in FigureS10 ####
def FA_z_m_flowbis_Q10(z, z_0, B, omega, gamma, gamma2, E, chi, rho_dry, delta, C, R_l, R_g, beta,R0_ref, T_exp, Q_10):
    '''Function returning flux of dry mass at a given depth z for all particles radii R_0
    in the Alldredge sinking speed fit case, mass fit case, with feedback of flow on degradation
    and modulation of degradation rate by temperature
    INPUTS:
    z                : depth at which speed and volumes are evaluated, in meters
    z_0              : initial position in meters
    B                : coefficient in Alldredge fit
    omega            : exponent in the power law for Alldredge speed as function of radius
    gamma            : radius shrinking rate without flow
    gamma2           : shrinking rate coefficient in presence of flow (see notes)
    E                : pre-factor in the power law for dry mass as function of size (Alldredge 1988)
    chi              : exponent in the power law for dry mass as function of size (Alldredge 1988)
    rho_dry          : dry mass per unit volume of alginate particles
    delta            : exponent in Sherwood law with Reynolds for Re>0.1
    C                : total abundance of particles, in m-3
    R_l              : lower size cut-off in meter
    R_g              : greater size cut-off in meter
    beta             : exponent of the size distribution
    R0_ref           : reference size for time vector build
    T_exp            : temperature of experiment, in degrees Celsius
    Q_10             : Q10 factor for microbial activity in degradation
    OUTPUT:
    F(z), in kg per m2 per s'''

    result = integrate.quad(phiA_m_flowbis_Q10, R_l, R_g, args=(z, z_0, B, omega, gamma, gamma2, E, chi, rho_dry,delta, C, R_l, R_g, beta,R0_ref, T_exp, Q_10))
    return result[0]


####### Functions for integration with fixed lower boundary #######

#### Used by FigureS7 ####
def R_0_minbis(z, R_min, z_0, B, omega, gamma, gamma2, E, chi, rho_dry,delta, R_g, R_low, R0_ref):
    '''Function returning the initial particle radius R0 at z_0 giving a radius equal to R_lim at depth z
    in the Alldredge sinking speed fit case, mass fit case, with feedback of flow on degradation.
    This initial radius is then used as a minimal bound of integration for the vertical flux.
    INPUTS:
    z                : depth considered
    R_min            : target minimal radius at depth z for flux integration
    z_0              : initial position in meters
    B                : coefficient in Alldredge fit
    omega            : exponent in the power law for Alldredge speed as function of radius
    gamma            : radius shrinking rate without flow
    gamma2           : shrinking rate coefficient in presence of flow (see notes)
    E                : pre-factor in the power law for dry mass as function of size (Alldredge 1988)
    chi              : exponent in the power law for dry mass as function of size (Alldredge 1988)
    rho_dry          : dry mass per unit volume of alginate particles
    delta            : exponent in Sherwood law with Reynolds for Re>0.1
    C                : total abundance of particles, in m-3
    R_g              : greater size cut-off in meter for initial distribution
    R_low            : lower bracket for finding solution
    R0_ref           : reference size for time vector build
    OUTPUT:
    R_0_min, in m'''
    
    sol = optimize.root_scalar(Rdif_to_Rminbis, args = (z, R_min, z_0, B, omega, gamma, gamma2, E, chi, rho_dry,delta,R0_ref),
                               bracket=[R_low,R_g], method='bisect',xtol=2e-12, rtol=8.881784197001252e-16, maxiter=100)
    if sol.converged ==0:
        print('root finding unconverged in search of lower bound R0')
    if sol.converged ==1:
        print('accuracy reached')
    print(sol.flag)
    return sol.root

#### Used by FigureS7 ####
def Rdif_to_Rminbis(R_0, z, R_min, z_0, B, omega, gamma, gamma2, E, chi, rho_dry,delta, R0_ref):
    '''function returning the difference between the radius of a particle initially of size R_0 at depth z
    and the minimal radius of integration R_min
    INPUTS:
    R_0              : initial radius considered in meters
    z                : depth considered in meters
    R_min            : target minimal radius at depth z for flux integration
    z_0              : initial position in meters
    B                : coefficient in Alldredge fit
    omega            : exponent in the power law for Alldredge speed as function of radius
    gamma            : radius shrinking rate without flow
    gamma2           : shrinking rate coefficient in presence of flow (see notes)
    E                : pre-factor in the power law for dry mass as function of size (Alldredge 1988)
    chi              : exponent in the power law for dry mass as function of size (Alldredge 1988)
    rho_dry          : dry mass per unit volume of alginate particles
    delta            : exponent in Sherwood law with Reynolds for Re>0.1
    C                : total abundance of particles, in m-3
    R_l              : lower size cut-off in meter
    R_g              : greater size cut-off in meter
    beta             : exponent of the size distribution
    OUTPUT:
    difference between R(R_0,z) and R_min in m'''
    # we reuse an existing function integrating the dynamics
    U_zR0, V_zR0 = solUV_z_m_flowbis(z,z_0, R_0, gamma, gamma2, B, delta, omega, E, chi, rho_dry,R0_ref)
    R_z =  np.power(3.0*V_zR0/(4*np.pi),1.0/3.0)

    return R_z - R_min

#### Used in FigureS7 ####
def FA_z_m_flow_fixed_cutoffbis(z, z_0, B, omega, gamma, gamma2, E, chi, rho_dry, delta, C, R_l, R_g, beta, R_0_min,R0_ref):
    '''Function returning flux of dry mass at a given depth z for all particles radii R_0
    in the Alldredge sinking speed fit case, mass fit case, with feedback of flow on degradation,
    with a lower size of integration passed as an argument.
    INPUTS:
    z                : depth at which speed and volumes are evaluated, in meters
    z_0              : initial position in meters
    B                : coefficient in Alldredge fit
    omega            : exponent in the power law for Alldredge speed as function of radius
    gamma            : radius shrinking rate without flow
    gamma2           : shrinking rate coefficient in presence of flow (see notes)
    E                : pre-factor in the power law for dry mass as function of size (Alldredge 1988)
    chi              : exponent in the power law for dry mass as function of size (Alldredge 1988)
    rho_dry          : dry mass per unit volume of alginate particles
    delta            : exponent in Sherwood law with Reynolds for Re>0.1
    C                : total abundance of particles, in m-3
    R_l              : lower size cut-off in meter
    R_g              : greater size cut-off in meter
    beta             : exponent of the size distribution
    R_0_min          : minimum radius of integration R_0 for considered depth z
    R0_ref           : reference size for time vector build

    OUTPUT:
    F(z), in kg per m2 per s'''

    result = integrate.quad(phiA_m_flowbis, R_0_min, R_g, args=(z, z_0, B, omega, gamma, gamma2, E, chi, rho_dry,delta, C, R_l, R_g, beta, R0_ref))
    return result[0]

#### Used in FigureS7 ####
def R_0_min_ref(z, R_min, z_0, B, omega, gamma, E, chi, rho_dry,delta, R_g, R_low,R0_ref):
    '''Function returning the initial particle radius R0 at z_0 giving a radius equal to R_lim at depth z
    in the Alldredge sinking speed fit case, mass fit case, with NO feedback of flow on degradation.
    This initial radius is then used as a minimal bound of integration for the vertical flux.
    INPUTS:
    z                : depth considered
    R_min            : target minimal radius at depth z for flux integration
    z_0              : initial position in meters
    B                : coefficient in Alldredge fit
    omega            : exponent in the power law for Alldredge speed as function of radius
    gamma            : radius shrinking rate without flow
    E                : pre-factor in the power law for dry mass as function of size (Alldredge 1988)
    chi              : exponent in the power law for dry mass as function of size (Alldredge 1988)
    rho_dry          : dry mass per unit volume of alginate particles
    delta            : exponent in Sherwood law with Reynolds for Re>0.1
    C                : total abundance of particles, in m-3
    R_g              : greater size cut-off in meter for initial distribution
    R_low            : lower bracket for finding solution
    R0_ref           : reference size for time vector build
    OUTPUT:
    R_0_min, in m'''
    
    sol = optimize.root_scalar(Rdif_to_Rmin_ref, args = (z, R_min, z_0, B, omega, gamma, E, chi, rho_dry,delta,R0_ref),
                               bracket=[R_low,R_g], method='bisect',xtol=2e-12, rtol=8.881784197001252e-16, maxiter=100)
    if sol.converged ==0:
        print('root finding unconverged in search of lower bound R0')
    if sol.converged ==1:
        print('accuracy reached')
    print(sol.flag)
    return sol.root

#### Used in FigureS7 ####
def Rdif_to_Rmin_ref(R_0, z, R_min, z_0, B, omega, gamma, E, chi, rho_dry,delta,R0_ref):
    '''function returning the difference between the radius of a particle initially of size R_0 at depth z
    and the minimal radius of integration R_min, mass case no flow feedback
    INPUTS:
    R_0              : initial radius considered in meters
    z                : depth considered in meters
    R_min            : target minimal radius at depth z for flux integration
    z_0              : initial position in meters
    B                : coefficient in Alldredge fit
    omega            : exponent in the power law for Alldredge speed as function of radius
    gamma            : radius shrinking rate without flow
    E                : pre-factor in the power law for dry mass as function of size (Alldredge 1988)
    chi              : exponent in the power law for dry mass as function of size (Alldredge 1988)
    rho_dry          : dry mass per unit volume of alginate particles
    delta            : exponent in Sherwood law with Reynolds for Re>0.1
    C                : total abundance of particles, in m-3
    R_l              : lower size cut-off in meter
    R_g              : greater size cut-off in meter
    beta             : exponent of the size distribution
    OUTPUT:
    difference between R(R_0,z) and R_min in m'''
    # we reuse an existing function integrating the dynamics
    U_zR0, V_zR0 = solUV_z_m_flow_refbis(z,z_0, R_0, gamma, B, delta, omega, E, chi, rho_dry,R0_ref)
    R_z =  np.power(3.0*V_zR0/(4*np.pi),1.0/3.0)

    return R_z - R_min

#### Used in FigureS7 ####
def FA_z_m_flow_ref_fixed_cutoffbis(z, z_0, B, omega, gamma, E, chi, rho_dry, delta, C, R_l, R_g, beta, R_0_min,R0_ref):
    '''Function returning flux of dry mass at a given depth z for all particles radii R_0
    in the Alldredge sinking speed fit case, mass fit case, with NO feedback of flow on degradation
    and a cutoff size which is passed as argument
    INPUTS:
    z                : depth at which speed and volumes are evaluated, in meters
    z_0              : initial position in meters
    B                : coefficient in Alldredge fit
    omega            : exponent in the power law for Alldredge speed as function of radius
    gamma            : radius shrinking rate without flow
    E                : pre-factor in the power law for dry mass as function of size (Alldredge 1988)
    chi              : exponent in the power law for dry mass as function of size (Alldredge 1988)
    rho_dry          : dry mass per unit volume of alginate particles
    delta            : exponent in Sherwood law with Reynolds for Re>0.1
    C                : total abundance of particles, in m-3
    R_l              : lower size cut-off in meter
    R_g              : greater size cut-off in meter
    beta             : exponent of the size distribution
    R_0_min          : minimum radius of integration R_0 for considered depth z
    R0_ref           : reference size for time vector build

    OUTPUT:
    F(z), in kg per m2 per s'''

    result = integrate.quad(phiA_m_flow_refbis, R_0_min, R_g, args=(z, z_0, B, omega, gamma, E, chi, rho_dry,delta, C, R_l, R_g, beta,R0_ref))
    return result[0]


