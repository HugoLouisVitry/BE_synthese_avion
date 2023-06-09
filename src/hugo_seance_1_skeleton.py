#!/usr/bin/env python3
#-*- coding: utf-8 -*-
import sys, math, numpy as np, matplotlib.pyplot as plt

import dynamic as dyn ,utils as ut

km = 0.5
def plot_thrust(P, filename=None):
    figure = ut.prepare_fig(None, f'Poussée {P.name}')
    U = [0, 1., 0, 0]
    hs, machs = np.linspace(3000, 11000, 5), np.linspace(0.5, 0.8, 30)
    for h in hs:
        thrusts = [dyn.propulsion_model([0, h, dyn.va_of_mach(mach, h), 0, 0, 0], U, P) for mach in machs] 
        plt.plot(machs, thrusts)
    ut.decorate(plt.gca(), f'Poussée maximum {P.eng_name}', 'Mach', '$N$', 
                [f'{_h} m' for _h in hs])
    ut.savefig(filename)
    return figure


def CL(P, alpha, dphr): return dyn.get_aero_coefs(1, alpha, 0, dphr, P)[0]

def plot_CL(P, filename=None):
    alphas = np.deg2rad(np.linspace(-10, 20, 30))
    dphrs = np.deg2rad(np.linspace(20, -30, 3))
    figure = ut.prepare_fig(None, f'Coefficient de portance {P.name}')
    for dphr in dphrs:
        plt.plot(np.rad2deg(alphas), CL(P, alphas, dphr))
    ut.decorate(plt.gca(), u'Coefficient de Portance {}'.format(P.name), r'$\alpha$ en degres', '$C_L$',
                ['$\delta _{{PHR}} =  ${:.1f}'.format(np.rad2deg(dphr)) for dphr in dphrs])
    ut.savefig(filename)

def Cm(P, alpha): return P.Cm0 -P.ms*P.CLa*(alpha-P.a0)

def plot_Cm(P, filename=None):
    alphas = np.deg2rad(np.linspace(-10, 20, 30))
    mss = [-0.1, 0., 0.2, 1.]
    figure = ut.prepare_fig(None, f'Coefficient de moment {P.name}')
    for ms in mss:
        P.set_mass_and_static_margin(km, ms)
        plt.plot(np.rad2deg(alphas), Cm(P, alphas))
    ut.decorate(plt.gca(), u'Coefficient de moment {}'.format(P.name), r'$\alpha$ en degres', '$C_m$',
                ['$ms =  ${: .1f}'.format(ms) for ms in mss])
    ut.savefig(filename)
    

def seance_1(ac=dyn.Param_A319(),coefm=0.5):
    km = coefm
    #plot_thrust(ac, f'./plots/{ac.get_name()}_thrust.png')
    #plot_CL(ac, f'./plots/{ac.get_name()}_CL.png')
    #plot_Cm(ac, f'./plots/{ac.get_name()}_Cm.png')
    plot_dphrEq(ac,f'./plots/{ac.get_name()}_dphr.png')
    plot_CL_Eq(ac, f'./plots/{ac.get_name()}_CL_e.png')
    pass

def dphr(P,alpha):return-(P.Cm0 - P.ms*P.CLa*(alpha-P.a0))/P.Cmd

def plot_dphrEq(P, filename=None):
    alphas = np.deg2rad(np.linspace(-10, 20, 30))
    mss = [-0.1, 0., 0.2, 1.]
    figure = ut.prepare_fig(None, f'dphrEq {P.name}')

    for ms in mss:
        P.set_mass_and_static_margin(km, ms)
        plt.plot(np.rad2deg(alphas),np.rad2deg(dphr(P,alphas)))
    ut.decorate(plt.gca(), u'$\delta$phrEq {}'.format(P.name), r'$\alpha$_e en degres', '$\delta$phrEq',
                ['$ms =  ${: .1f}'.format(ms) for ms in mss])
    ut.savefig(filename)

def plot_CL_Eq(P, filename=None):
    alphas = np.deg2rad(np.linspace(-10, 20, 30))
    mss = [0.2,1]
    figure = ut.prepare_fig(None, f'CL_Eq {P.name}')
    for ms in mss:
        P.set_mass_and_static_margin(km, ms)
        dphrs = dphr(P,alphas)
        plt.plot(np.rad2deg(alphas),CL(P, alphas, dphrs))
    ut.decorate(plt.gca(), u'CL_e {}'.format(P.name), r'$\alpha$_e en degres', 'CL_e',
                ['$ms =  ${: .1f}'.format(ms) for ms in mss])
    ut.savefig(filename)
    
    return alphas
    
    
    
    
if __name__ == "__main__":
    if 'all' in sys.argv:
        for t in dyn.all_ac_types:
            seance_1(t())
    else:
        P = dyn.Param_A319()#use_improved_induced_drag = False, use_stall = False)
        seance_1(P,0.1)
        plt.show()
    plt.show()
        
        
