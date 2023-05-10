#-*- coding: utf-8 -*-
'''
Dynamic model for a 3 Degrees Of Freedom longitudinal aircraft  
'''
import math, numpy as np, scipy.optimize
import matplotlib.pyplot as plt

import utils as ut

''' naming of state and input components '''
s_x, s_z, s_va, s_a, s_th, s_q, s_size = range(0, 7)
i_dphr, i_dth, i_size =  range(0, 3)

def get_mach(va, T, k=1.4, Rs=287.05): return va/math.sqrt(k*Rs*T)

def va_of_mach(m, h, k=1.4, Rs=287.05):
    p, rho, T = ut.isa(h)
    return m*math.sqrt(k*Rs*T)

def propulsion_model(X, U, P):
    p, rho, T = ut.isa(X[s_z])
    rho0 = 1.225
    mach = get_mach(X[s_va], T)
    return P.F0*math.pow(rho/rho0, 0.6)*(0.568+0.25*math.pow(1.2-mach, 3))*U[i_dth]

def get_aero_coefs(va, alpha, q, dphr, P):
    St_over_S = P.St/P.S
    CL0 = (St_over_S*0.25*P.CLat - P.CLa)*P.a0
    CLa = P.CLa + St_over_S*P.CLat*(1-0.25)
    CLq = P.lt * St_over_S * P.CLat * P.CLq
    CLdphr = St_over_S*P.CLat
    # lift coefficient
    CL = CL0 + CLa*alpha + CLq*q/va + CLdphr * dphr
    if P.use_stall: #post stall modelling
        Nstall, Cstall = 4.2, 2e-6
        # Model inspired from the following paper:
        # Development of Aerodynamic Modeling and Calibration
        # Methods for General Aviation Aircraft Performance
        # Analysis - a Survey and Comparison of Models
        # Sanggyu Min, Evan Harrison, Hernando Jimenez, Dimitri Mavris
        # AIAA AVIATION Forum
        # 22-26 June 2015, Dallas, TX
        # 15th AIAA Aviation Technology, Integration, and Operations Conference
        CL -= Cstall*np.rad2deg(1)**Nstall*(np.abs(alpha - P.a0))**Nstall
    # induced drag coefficient
    if P.use_improved_induced_drag:
        CLw = P.CLa*(alpha-P.a0)
        alphat = alpha - 0.25*(alpha-P.a0) + dphr + P.Cmq*P.lt/va*q 
        CLt = P.CLat*alphat
        CDi = CLw**2 / (math.pi*P._lambda) + St_over_S*CLt**2 / (math.pi*P._lambdat) + St_over_S*CLw*CLt / (math.pi*P._lambda)
    else:
        CDi = P.ki*CL**2 # induced drag coefficient
    # drag coefficient
    CD = P.CD0 + CDi
    # pitching moment coefficient
    Cm = P.Cm0 - P.ms*P.CLa*(alpha-P.a0) + P.Cmq*P.lt/va*q + P.Cmd*dphr
    return CL, CD, Cm

def get_aero_forces_and_moments(X, U, P):
    p, rho, T = ut.isa(X[s_z])
    pdyn = 0.5*rho*X[s_va]**2
    CL, CD, Cm = get_aero_coefs(X[s_va], X[s_a], X[s_q], U[i_dphr], P)
    L, D, M = pdyn*P.S*np.array([CL, CD, P.cbar*Cm])
    return L, D, M

def dyn(X, t, U, P):
    '''  Dynamic model '''
    Xdot = np.zeros(s_size)
    gamma_a = X[s_th] - X[s_a]  # air path angle
    cg, sg = math.cos(gamma_a), math.sin(gamma_a)
    ca, sa = math.cos(X[s_a]), math.sin(X[s_a])
    L, D, M = get_aero_forces_and_moments(X, U, P)
    F = propulsion_model(X, U, P)
    Xdot[s_x] = X[s_va] * cg
    Xdot[s_z] = X[s_va] * sg
    Xdot[s_va] = (F*ca-D)/P.m-P.g*sg 
    Xdot[s_a] = X[s_q] - (L+F*sa)/P.m/X[s_va]  + P.g/X[s_va]*cg 
    Xdot[s_th] = X[s_q]
    Xdot[s_q] = M/P.Iyy
    return Xdot

def trim(P, args=None):
    va = args.get('va', 100.)
    gamma = args.get('gamma', 0.)
    z = args.get('z', 5000.)
    def err_func(arg):
        (dphr, dth, alpha) = arg
        theta = gamma + alpha
        U = np.array([dphr, dth])
        X = np.array([0., z, va, alpha, theta, 0])
        L, D, M = get_aero_forces_and_moments(X, U, P)
        F = propulsion_model(X, U, P)
        cg, sg = math.cos(gamma), math.sin(gamma)
        ca, sa = math.cos(alpha), math.sin(alpha)
        return [(F*ca-D)/P.m-P.g*sg, -(L+F*sa)/P.m + P.g*cg, M]
    p0 = [np.deg2rad(0.), 0.5, np.deg2rad(1.)]
    sol = scipy.optimize.root(err_func, p0, method='hybr')
    dphr, dth, alpha = sol.x # elevator, throttle, angle of attack
    X, U = [0, z, va, alpha, gamma+alpha, 0], [dphr, dth]
    return X, U


class Param:
    def __init__(self, km=0.5, ms=0.3, use_improved_induced_drag=True, use_stall=True):
        self.km = km                  # mass coefficient
        self.ms = ms                  # static margin
        self.use_improved_induced_drag = use_improved_induced_drag
        self.use_stall = use_stall
        # constants
        self.a0 = np.deg2rad(-2.)     # zero lift angle
        self.CD0 = 0.025              # zero drag coefficient
        self.Cm0 = -0.59              # zero moment coefficient
        self.CLq = 1.3
        self.g = 9.81                 # gravity

    def set_mass_and_static_margin(self, km, sm):
        self.km = km
        self.ms = sm
        self.compute_auxiliary()

    def compute_auxiliary(self):
        self.m = (1-self.km)*self.m_OWE + self.km*self.m_MTOW
        self.Iyy = 0.5*(1./12.*self.m*self.l_fus**2)

        self.lt = 0.5 * self.l_fus           # CG to tail distance

        self.CLa = math.pi*self._lambda/(1.+math.sqrt(1+(0.5*self._lambda)**2))
        self.ki = 1./(math.pi*self._lambda)

        self.CLat = math.pi*self._lambdat/(1.+math.sqrt(1+(0.5*self._lambdat)**2))
        
        self.Vt = self.lt*self.St/self.cbar/self.S # tail volume
        self.Cmd = -self.Vt*self.CLat
        self.Cmq = self.Cmd * self.CLq 

    def get_name(self): return self.name.replace (" ", "_")


class Param_A320(Param):
    def __init__(self, **args):
        Param.__init__(self, **args)
        self.name = 'Airbus A-320'
        self.m_OWE  = 39733.
        self.m_MTOW = 73500.

        self.l_fus    =  37.57          # length of fuselage
        self.cbar     =   4.19          # wing reference chord
        self.St       =  31.            # tail lifting surface
        self.S        = 122.44          # wing surface
        self._lambdat =   5.            # tail aspect ratio
        self._lambda  =   9.39          # wing aspect ratio
        
        self.F0       = 2.*111205       # engine max thrust
        self.eng_name = 'CFM 56-5A1' 

        self.compute_auxiliary()


class Param_737_800(Param):
    def __init__(self, ):
        Param.__init__(self)
        self.name = 'Boeing 737-800'
        self.m_OWE  = 41413.
        self.m_MTOW = 70534.
 
        self.l_fus    =  38.02          # length of fuselage
        self.cbar     =   4.17          # wing reference chord
        self.St       =  32.8           # tail lifting surface
        self.S        = 124.6           # wing surface
        self._lambdat =   6.28          # tail aspect ratio
        self._lambda  =   9.45          # wing aspect ratio
        
        self.F0       = 2.*106757       # engine max thrust
        self.eng_name = 'CFM 56-7B24' 

        self.compute_auxiliary()


class Param_A319(Param):
    def __init__(self):
        Param.__init__(self)
        self.name = 'Airbus A-319'
        self.m_OWE  = 39358.
        self.m_MTOW = 64000.

        self.l_fus    =  33.84          # length of fuselage
        self.cbar     =   4.19          # wing reference chord
        self.St       =  31.0           # tail lifting surface
        self.S        = 122.44          # wing surface
        self._lambdat =   5.            # tail aspect ratio
        self._lambda  =   9.39          # wing aspect ratio

        self.F0       = 2.*97860        # engine max thrust
        self.eng_name = 'CFM 56-5B5' 

        self.compute_auxiliary()


class Param_A321(Param):
    def __init__(self):
        Param.__init__(self)
        self.name = 'Airbus A-321'
        self.m_OWE  = 47000.
        self.m_MTOW = 89000.

        self.l_fus    =  44.51          # length of fuselage
        self.cbar     =   4.34          # wing reference chord
        self.St       =  31.0           # tail lifting surface
        self.S        = 126.00          # wing surface
        self._lambdat =   5.            # tail aspect ratio
        self._lambda  =   9.13          # wing aspect ratio

        self.F0       = 2.*133446       # engine max thrust
        self.eng_name = 'CFM 56-5B1' 

        self.compute_auxiliary()


class Param_737_700(Param):
    def __init__(self):
        Param.__init__(self)
        self.name = 'Boeing 737-700'
        self.m_OWE  = 37648.
        self.m_MTOW = 60326.

        self.l_fus    =  32.18          # length of fuselage
        self.cbar     =   4.17          # wing reference chord
        self.St       =  32.8           # tail lifting surface
        self.S        = 124.60          # wing surface
        self._lambdat =   6.28          # tail aspect ratio
        self._lambda  =   9.44          # wing aspect ratio

        self.F0       = 2.*91633        # engine max thrust
        self.eng_name = 'CFM 56-7B20' 

        self.compute_auxiliary()


class Param_737_300(Param):
    def __init__(self):
        Param.__init__(self)
        self.name = 'Boeing 737-300'
        self.m_OWE  = 31480.
        self.m_MTOW = 56473.

        self.l_fus    =  32.18          # length of fuselage
        self.cbar     =   3.73          # wing reference chord
        self.St       =  31.31          # tail lifting surface
        self.S        =  91.04          # wing surface
        self._lambdat =   5.15          # tail aspect ratio
        self._lambda  =   9.16          # wing aspect ratio

        self.F0       = 2.*88694        # engine max thrust
        self.eng_name = 'CFM 56-3B1' 

        self.compute_auxiliary()

all_ac_types = [Param_A320, Param_737_800, Param_A319, Param_A321, Param_737_700, Param_737_300]

def plot(time, X, U=None, figure=None, window_title="Trajectory"):
    figure = ut.prepare_fig(figure, window_title, (20.48, 10.24))
    plots = [("$x$",       "m",     X[:,s_x], None),
             ("$z$",       "m",     X[:,s_z], 2.),
             ("$v_a$",     "m/s",   X[:,s_va], 1.),
             ("$\\alpha$", "deg",   np.rad2deg(X[:,s_a]), 2.),
             ("$\\theta$", "deg",   np.rad2deg(X[:,s_th]), 2.),
             ("$q$",       "deg/s", np.rad2deg(X[:,s_q]), 2.)]
    for i, (title, ylab, data, min_yspan) in enumerate(plots):
        ax = plt.subplot(3, 2, i+1)
        plt.plot(time, data)
        ut.decorate(ax, title=title, ylab=ylab, min_yspan=min_yspan)
    return figure
