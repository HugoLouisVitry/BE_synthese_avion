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
                    trims[i, j, k, l] = (Xe[dyn.s_a], Ue[dyn.i_dphr], Ue[dyn.i_dth])
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


def seance_2(aircraft=dyn.Param_A319()):

    zs, Mas = np.linspace(3000, 11000, 9), [0.4, 0.8]
    sms, kms = [0.2, 1.], [0.1, 0.9]

    trims = get_all_trims(aircraft, zs, Mas, sms, kms)

    #plot_all_trims(aircraft, zs, Mas, sms, kms, trims, f'./plots/{aircraft.get_name()}_trim.png')
    #all_F(aircraft,f'./plots/{aircraft.get_name()}_throttle.png')

def Fn(aircraft,Xe,Ue):
    
    D = dyn.get_aero_forces_and_moments(Xe, Ue, aircraft)[1]
    alpha = Xe[3]
    F= D/math.cos(alpha)
    return F

def all_F(aircraft=dyn.Param_A319(),filename=None):
    zs, Mas = [3000,10000], np.linspace(0.4,0.9,50)
    sms, kms = [0.2, 1.], [0.1, 1]
    
    margins = (0.03, 0.05, 0.98, 0.95, 0.2, 0.38)
    fig = ut.prepare_fig(window_title=f'Throttle {aircraft.name}', figsize=(20.48, 10.24), margins=margins)
    m=0 # index of current subplot
    
    Fs = np.zeros((len(zs), len(Mas), len(sms), len(kms), 1))
    for k, sm in enumerate(sms):
            for l,km in enumerate(kms):
                for i, z in enumerate(zs):
                    for j, Ma in enumerate(Mas):
                        Xe,Ue = get_trim(aircraft, z, Ma, sm, km)
                        Fs[i,j,k,l] = Fn(aircraft,Xe,Ue)
            
                ax = plt.subplot(2, 2, m+1)
                plt.plot(Mas,Fs[0,:,k,l,0]/1000)
                plt.plot(Mas,Fs[1,:,k,l,0]/1000)    
                ut.decorate(ax, r'$F \quad sm {} \quad km {}$'.format(sm, km), r'Mach', 'Poussée en kN', legend=['z {}'.format(z) for z in zs])
                m = m+1
    ut.savefig(filename)
def get_CL(aircraft, z, Ma, sm, km):
    Xe,Ue = get_trim(aircraft,z, Ma, sm, km)
    va=Xe[2]
    alpha=Xe[3]
    q=0
    dphr=Ue[0]
    CL = dyn.get_aero_coefs(va, alpha, q, dphr, aircraft)[0]
    return CL

def dth(aircraft,F,z,Mach):
    p, rho, T = ut.isa(z)
    rho0 = 1.225
    dth = F/aircraft.F0 *(rho0/z*rho)**0.6 *(1/(0.568+0.25*(1.2-Mach)**3))
    return dth 

def plot_dth(aircraft, z, Ma, sm, km):
    Xe,Ue = get_trim(aircraft, z, Ma, sm, km)
    F = Fn(aircraft,Xe,Ue)
    Dth = dth(aircraft,F,z,Ma)
        
    
    
if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True, linewidth=200)
    P = dyn.Param_A319()
    #; P.use_stall = False
    seance_2()
    CL = get_CL(dyn.Param_A319(), 5000, 0.5,0.2 ,0.1 )
    print(" CL = ",CL)

    plt.show()
