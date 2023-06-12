#!/usr/bin/env python3
#-*- coding: utf-8 -*-
import math, numpy as np, scipy.integrate
import matplotlib.pyplot as plt
import pdb

import dynamic as dyn
import utils as ut


def get_trim(aircraft, z, Ma, sm, km):
    '''
    Calcul du trim pour un point de vol
    '''
    aircraft.set_mass_and_static_margin(km, sm)
    va = dyn.va_of_mach(Ma, z)
    Xe, Ue = dyn.trim(aircraft, {'va':va, 'z':z, 'gamma':0})
    return Xe, Ue

def get_all_trims(aircraft, zs, Mas, sms, kms):
    '''
    Calcul de trims pour une serie de points de vol
    '''
    trims = np.zeros((len(zs), len(Mas), len(sms), len(kms), 3))
    for i, z in enumerate(zs):
        for j, Ma in enumerate(Mas):
            for k, sm in enumerate(sms):
                for l,km in enumerate(kms):
                    Xe, Ue = get_trim(aircraft, z, Ma, sm, km)
                    trims[i, j, k, l] = (Xe[dyn. s_a], Ue[dyn.i_dphr], Ue[dyn.i_dth])
    return trims


def plot_all_trims(aircraft, hs, Mas, sms, kms, trims, filename=None):
    '''
    Affichage des trims
    '''
    margins = (0.03, 0.05, 0.98, 0.95, 0.2, 0.38)
    fig = ut.prepare_fig(window_title=f'Trims {aircraft.name}', figsize=(20.48, 10.24), margins=margins)
    m=0 # index of current subplot
    for k, sm in enumerate(sms):
        for l,km in enumerate(kms):
            for i, h in enumerate(hs):
                for j, Ma in enumerate(Mas):
                    alpha, dphr, dth = trims[i, j, k, l]
                    fmt = 'alt {:5.0f} Ma {:.1f} sm {:.1f} km {:.1f} -> alpha {:5.2f} deg phr {:-5.1f} deg throttle {:.1f} %'
                    print(fmt.format(h, Ma, sm, km, np.rad2deg(alpha), np.rad2deg(dphr), 100*dth))
            ax = plt.subplot(4, 3, 3*m+1)
            plt.plot(hs, np.rad2deg(trims[:, 0, k, l, 0]))
            plt.plot(hs, np.rad2deg(trims[:, 1, k, l, 0]))
            ut.decorate(ax, r'$\alpha \quad sm {} \quad km {}$'.format(sm, km), r'altitude', '$deg$', legend=['Mach {}'.format(Ma) for Ma in Mas])
            ax = plt.subplot(4, 3, 3*m+2)
            plt.plot(hs, np.rad2deg(trims[:, 0, k, l, 1]))
            plt.plot(hs, np.rad2deg(trims[:, 1, k, l, 1]))
            ut.decorate(ax, r'$phr \quad sm {} \quad km {}$'.format(sm, km), r'altitude', '$deg$', legend=['Mach {}'.format(Ma) for Ma in Mas])
            ax = plt.subplot(4, 3, 3*m+3)
            plt.plot(hs, trims[:, 0, k, l, 2]*100)
            plt.plot(hs, trims[:, 1, k, l, 2]*100)
            ut.decorate(ax, r'$throttle \quad sm {} \quad km {}$'.format(sm, km), r'altitude', '$\%$', legend=['Mach {}'.format(Ma) for Ma in Mas])
            m = m+1
    ut.savefig(filename)


def plot_traj_trim(aircraft, h, Ma, sm, km):
    '''
    Affichage d'une trajectoire temporelle avec un point de trim comme condition initiale
    '''
    aircraft.set_mass_and_static_margin(km, sm)
    va = dyn.va_of_mach(Ma, h)
    Xe, Ue = dyn.trim(aircraft, {'va':va, 'h':h, 'gamma':0})
    time = np.arange(0., 100, 0.5)
    X = scipy.integrate.odeint(dyn.dyn, Xe, time, args=(Ue, aircraft))
    dyn.plot(time, X)


def get_CL_Fmax_trim(aircraft, h, Ma):
    p, rho, T = ut.isa(h)
    va = dyn.va_of_mach(Ma, h)
    pdyn = 0.5*rho*va**2
    return P.m*P.g/(pdyn*P.S), propulsion_model([0, h, va, 0, 0, 0], [0, 1, 0, 0], aircraft)


def get_linearized_model(aircraft, h, Ma, sm, km):    
    '''
    Calcul numérique du modèle tangeant linéarisé pour un point de trim
    '''
    aircraft.set_mass_and_static_margin(km, sm)
    va = dyn.va_of_mach(Ma, h)
    Xe, Ue = dyn.trim(aircraft, {'va':va, 'h':h, 'gamma':0})
    if np.isnan(Ue[dyn.i_dphr]) : return None, None, None, None
    A, B = ut.num_jacobian(Xe, Ue, aircraft, dyn.dyn)
    poles, vect_p = np.linalg.eig(A[dyn.s_va:, dyn.s_va:])
    return A, B, poles, vect_p

def plot_poles(aircraft, hs, Mas, sms, kms, filename=None):
    margins = (0.03, 0.05, 0.98, 0.95, 0.2, 0.38)
    fig = ut.prepare_fig(window_title='Poles {} (km:{})'.format(aircraft.name, kms), figsize=(20.48, 10.24), margins=margins)
    plt.grid(True, color='k', linestyle='-', which='both', axis='both', linewidth=0.2)
    for i, h in enumerate(hs[::-1]):
        for j, Ma in enumerate(Mas):
            ax = plt.subplot(len(hs), len(Mas), i*len(Mas)+j+1)
            for k, sm in enumerate(sms):
                for l,km in enumerate(kms):
                    A, B, poles, vect_p = get_linearized_model(aircraft, h, Ma, sm, km)
                    print('{}'.format(poles))
                    if A is not None:
                        plt.plot(poles.real, poles.imag, '*', markersize=20, alpha=1.)
            ut.decorate(ax, r'$h:{}m \quad  Ma:{} Km:{}$'.format(h, Ma, km), legend=['ms: {}'.format(sm) for sm in sms])
    ut.savefig(filename)


def seance_2(aircraft=dyn.Param_A319()):

    zs, Mas = np.linspace(3000, 11000, 9), [0.5, 0.8]
    sms, kms = [0.2, 1.], [0.1, 0.9]

    trims = get_all_trims(aircraft, zs, Mas, sms, kms)

    plot_all_trims(aircraft, zs, Mas, sms, kms, trims, f'./plots/{aircraft.get_name()}_trim.png')

    plot_traj_trim(aircraft, 5000, 0.5, 0.2, 0.5)

    plot_poles(aircraft, zs, Mas, sms, [kms[0]], f'./plots/{aircraft.get_name()}_charlie_poles_1.png')
    plot_poles(aircraft, zs, Mas, sms, [kms[1]], f'./plots/{aircraft.get_name()}_charlie_poles_2.png')



if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True, linewidth=200)
    P = dyn.Param_A319
    ()
    #; P.use_stall = False
    seance_2()
    plt.show()
