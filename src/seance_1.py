#!/usr/bin/env python3
#-*- coding: utf-8 -*-
import sys, math, numpy as np, matplotlib.pyplot as plt

import dynamic as dyn ,utils as ut

def plot_thrust(P, filename=None):
    figure = ut.prepare_fig(None, f'Poussée {P.name}')
    U = [0, 1.]
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

def Cm(P, alpha):
    Cma = -P.ms*P.CLa
    return P.Cm0 + Cma*(alpha-P.a0)

def plot_Cm(P, filename=None):
    alphas = np.deg2rad(np.linspace(-10, 20, 30))
    mss = [-0.1, 0., 0.2, 1.]
    figure = ut.prepare_fig(None, f'Coefficient de moment {P.name}')
    for ms in mss:
        P.set_mass_and_static_margin(0.5, ms)
        plt.plot(np.rad2deg(alphas), Cm(P, alphas))
    ut.decorate(plt.gca(), u'Coefficient de moment {}'.format(P.name), r'$\alpha$ en degres', '$C_m$',
                ['$ms =  ${: .1f}'.format(ms) for ms in mss])
    ut.savefig(filename)


def dphr_e(P, alpha):
    return 1./P.Cmd*(P.ms*P.CLa*(alpha-P.a0) - P.Cm0)
    
def plot_dphr_e(P, filename=None):
    alphas = np.deg2rad(np.linspace(-10, 20, 30))
    mss = [-0.1, 0., 0.2, 1.]
    figure = ut.prepare_fig(None, f'Équilibre {P.name}')
    for ms in mss:
        P.set_mass_and_static_margin(0.5, ms)
        dmes = np.array([dphr_e(P, alpha) for alpha in alphas])
        plt.plot(np.rad2deg(alphas), np.rad2deg(dmes))
    ut.decorate(plt.gca(), r'$\delta_{{PHR_e}}$ {}'.format(P.name), r'$\alpha$ en degres', r'$\delta_{PHR_e}$ en degres',
                ['$ms =  ${: .1f}'.format(ms) for ms in mss])
    ut.savefig(filename)

def plot_CLe(P, filename=None):
    alphas = np.deg2rad(np.linspace(-10, 20, 30))
    figure = ut.prepare_fig(None, f'Coefficient de portance équilibrée {P.name}')
    sms = [0.2, 1]
    for sm in sms:
        P.set_mass_and_static_margin(0.5, sm)
        dphres = [dphr_e(P, alpha) for alpha in alphas]
        CLes = [CL(P, alpha, dphr) for alpha, dphr in zip(alphas, dphres)]
        plt.plot(np.rad2deg(alphas), CLes)
    ut.decorate(plt.gca(), u'Coefficient de portance équilibrée {}'.format(P.name), r'$\alpha$ en degres', r'$CL_e$',
                ['$ms =  ${: .1f}'.format(sm) for sm in sms])
    ut.savefig(filename)

def plot_polar(P, filename=None, figure=None):
    alphas = np.deg2rad(np.linspace(-10, 20, 30))
    figure = ut.prepare_fig(figure, f'Polaire équilibrée {P.name}')
    sms = [0.2, 1]
    for sm in sms:
        P.set_mass_and_static_margin(0.5, sm)
        dphres = [dphr_e(P, alpha) for alpha in alphas]
        CLes = [CL(P, alpha, dphre) for alpha, dphre in zip(alphas, dphres)]
        CDes = [dyn.get_aero_coefs(1, alpha, 0, dphre, P)[1] for alpha, dphre in zip(alphas, dphres)]
        l_over_d  = np.array(CLes)/np.array(CDes)
        l_over_d_max, imax = np.amax(l_over_d), np.argmax(l_over_d)
        plt.plot(CDes, CLes, label=f'sm: {sm}, fmax {l_over_d_max:.1f}')
        print(f'finesse max: {P.name} - > {l_over_d_max:.2f} ({imax})')
        plt.plot([0., CDes[imax]],[0., CLes[imax]], '--b')
        
    ut.decorate(plt.gca(), f'Polaire équilibrée {P.name} ', r'$CD_e$', r'$CL_e$')#,(finesse max: {l_over_d_max:.1f})
                #['$ms =  ${: .1f}'.format(sm) for sm in sms])
    plt.legend()

    plt.plot()
    ut.savefig(filename)
    return figure

def seance_1(ac=dyn.Param_A319()):
    plot_thrust(ac, f'./plots/{ac.get_name()}_charlie_thrust.png')
    plot_CL(ac, f'./plots/{ac.get_name()}_charlie_CL.png')
    plot_Cm(ac, f'./plots/{ac.get_name()}_charlie_Cm.png')
    plot_dphr_e(ac, f'./plots/{ac.get_name()}_charlie_dPHR.png')
    plot_CLe(ac, f'./plots/{ac.get_name()}_charlie_CLe.png')
    plot_polar(ac, f'./plots/{ac.get_name()}_charlie_polar.png')

if __name__ == "__main__":
    if 'all' in sys.argv:
        for t in dyn.all_ac_types:
            seance_1(t())
    else:
        P = dyn.Param_A319()#use_improved_induced_drag = False, use_stall = False)
        seance_1(P)
        plt.show()
