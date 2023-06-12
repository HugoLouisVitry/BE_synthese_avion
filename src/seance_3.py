#!/usr/bin/env python3
#-*- coding: utf-8 -*-
'''
squelette de code pour la troisième séance du projet de synthèse
'''
import math, numpy as np, scipy.integrate
import matplotlib.pyplot as plt
import pdb

import dynamic as dyn
import utils as ut
np.set_printoptions(precision=3, suppress=True, linewidth=200)

def get_trim(aircraft, z, Ma, sm, km):
    '''
    Calcul du trim pour un point de vol
    '''
    aircraft.set_mass_and_static_margin(km, sm)
    va = dyn.va_of_mach(Ma, z)
    Xe, Ue = dyn.trim(aircraft, {'va':va, 'z':z, 'gamma':0})
    return Xe, Ue

def NonLinearTraj(aircraft, z, Ma, sm, km, Wh, time):
    '''
    Calcul d'une trajectoire avec un point de trim comme condition initiale
    '''
    aircraft.set_mass_and_static_margin(km, sm)
    va = dyn.va_of_mach(Ma, z)
    Xe, Ue = dyn.trim(aircraft, {'va':va, 'z':z, 'gamma':0})
    Xe[dyn.s_a] = Xe[dyn.s_a] + math.atan(Wh/Xe[dyn.s_va]) #perturbation
    X = scipy.integrate.odeint(dyn.dyn, Xe, time, args=(Ue, aircraft))
    return X

'''  Dynamic model '''
def LinearDyn(X, t, U, A, B): return np.dot(A,X) + np.dot(B,U)


def LinearTraj(aircraft, z, Ma, sm, km, Wh, time):
    '''
    Calcul d'une trajectoire avec un point de trim comme condition initiale
    '''
    aircraft.set_mass_and_static_margin(km, sm)
    va = dyn.va_of_mach(Ma, z)
    Xe, Ue = dyn.trim(aircraft, {'va':va, 'z':z, 'gamma':0})
    A, B = ut.num_jacobian(Xe, Ue, aircraft, dyn.dyn)
    dX0 = np.zeros(dyn.s_size)
    dU = np.zeros(dyn.i_size)
    dX0[dyn.s_a] = math.atan(Wh/Xe[dyn.s_va]) #perturbation
    dX = scipy.integrate.odeint(LinearDyn, dX0, time, args=(dU, A, B))
    Xlin = np.array([dXi+Xe for dXi in dX])
    return Xlin

def get_linearized_model(aircraft, h, Ma, sm, km):
    '''
    Calcul numérique du modèle tangeant linéarisé pour un point de trim
    '''
    aircraft.set_mass_and_static_margin(km, sm)
    Xe, Ue = dyn.trim(aircraft, {'va':dyn.va_of_mach(Ma, h), 'h':h, 'gamma':0})
    A, B = ut.num_jacobian(Xe, Ue, aircraft, dyn.dyn)
    print('poles dim6 {}'.format(np.linalg.eig(A)))
    Al, Bl = A[dyn.s_va:, dyn.s_va:], B[dyn.s_va:, :dyn.i_size] # limit size of the representation
    poles, vect_p = np.linalg.eig(Al)
    return Al, Bl, poles, vect_p

# division selon puissances croissantes
def div_inc_pow(num, den, order):
    rnum =  np.zeros(len(den))
    for i in range(0,len(num)): rnum[i] = num[-i-1]
    rden = den[::-1]
    res = np.zeros(order)
    for i in range(0, order):
        quot, rem = np.polydiv(rnum, rden)
        res[i], rnum = quot, np.zeros(len(den))
        for i in range(0,len(rem)):
            rnum[i] = rem[i]
    return res[::-1]



def compute_controlability_matrix(A, B):
    nra, nca = A.shape
    nrb, ncb = B.shape
    Qc = np.zeros((nra, nra*ncb), dtype=complex)
    tmp = np.array(B)
    for i in range(0, nra):
        Qc[:,i*ncb:(i+1)*ncb] = tmp
        tmp = np.dot(A, tmp)
    return Qc, np.linalg.matrix_rank(Qc)

def faddeev(A):
    r,c = A.shape
    F = [np.eye(r)]
    d = [-1]
    for i in range(0, r):
        AFi = np.dot(A, F[-1])
        d.append(1./(i+1)*np.trace(AFi))
        F.append(AFi-d[-1]*np.eye(r))
    return -np.array(d), F

def compute_transfert_matrix(A, B, C):
    nr, nc = A.shape
    d, F =  faddeev(A)
    M = np.zeros((nr, nc, nc-1))
    N = np.zeros((nr, nc, nc-1))
    for i in range(0,nr):
        for j in range(0,nc):
            M[i, j] = [F[k][i,j] for k in range(0, nr-1)]
            #
            # a finir
            #
    return d.real, M.real, N.real


def seance_3(aircraft = dyn.Param_A319()):

    def run_trajecotry(time, h, Ma, ms, km, Wh=2, filename=None):
        Xe, Ue =  get_trim(aircraft, h, Ma, ms, km)
        fmt = r'Trim: h={:.1f}, Ma={:.1f}, ms={:.1f}, km={:.1f} -> $\alpha$={:.1f}, $\delta_{{PHR}}$={:.1f}, $\delta_{{th}}$={:.1f}'
        print(fmt.format(h, Ma, ms, km, np.rad2deg(Xe[dyn.s_a]), np.rad2deg(Ue[dyn.i_dphr]), Ue[dyn.i_dth]))
        X = NonLinearTraj(aircraft, h, Ma, ms, km, Wh, time)
        Xlin = LinearTraj(aircraft, h, Ma, ms, km, Wh, time)
        figure = dyn.plot(time, X)
        dyn.plot(time, Xlin, figure=figure)
        if filename is not None: plt.savefig(filename, dpi=160)    

    h, Ma, ms, km = 3000, 0.5, 0.2, 0.1
    run_trajecotry(np.arange(0., 240, 0.1), h, Ma, ms, km, Wh=2, filename='./plots/{}-charlie_traj_phugoide'.format(aircraft.get_name()))
    run_trajecotry(np.arange(0., 10, 0.1), h, Ma, ms, km, Wh=2, filename='./plots/{}-charlie_traj_osc_incid'.format(aircraft.get_name()))

    A, B, poles, vect_p = get_linearized_model(aircraft, h, Ma, ms, km)
    print('A={}'.format(A))
    print('B={}'.format(B))
    print('poles={}'.format(poles))
    print('vect_p={}'.format(vect_p))
    Am = np.diag(poles)
    print('Am=\n{}'.format(Am))
    Bm = np.dot(np.linalg.inv(vect_p), B)
    print('Bm=\n{}'.format(Bm))

    if 1:
        import control.matlab
        Qc = control.matlab.ctrb(A,B)
        print (np.linalg.matrix_rank(Qc))
        Qcm = control.matlab.ctrb(Am,Bm)
        print (np.linalg.matrix_rank(Qcm))

        C = [0, 0, 1, 0]
        Cm = np.dot(C, vect_p)

        # compute transfert function
        sys_ss = control.matlab.ss(A, B, C, [0, 0])
        tf = control.matlab.ss2tf(sys_ss)
        print(tf)
        # get elevator to theta function
        num = tf.num[0][0]
        den = tf.den[0][0]
        print('tf ', num, den)
        # get poles
        poles =  np.roots(den)
        slow_poles = poles[2:]
        # compute polynomial with the pair of slow poles
        red_den = np.poly(slow_poles)
        print('reduced den', red_den)

        # compute reduced transfert function
        dpc = div_inc_pow(num, den, 2)
        bdash0 = dpc[1]*red_den[-1]
        bdash1 = dpc[0]*red_den[-1]+red_den[-2]*bdash0
        red_num = [bdash1, bdash0]

        red_sys = control.matlab.tf2ss(red_num, red_den)

        # compare original and reduced model
        # T = np.linspace(0, 240, 500)
        # Y1, T1 = control.matlab.step(sys_ss, T)
        # Y2, T2 = control.matlab.step(red_sys, T)
        # plt.figure()
        # plt.plot(T1, Y1/10)
        # plt.plot(T2, Y2/10)
        # ut.decorate(plt.gca(), title='', xlab='time in s', ylab='$\\theta$ in degres', legend=[u'originale (4ème ordre)', u'réduite (2ème ordre)'])
        # plt.savefig('../plots/{}_traj_reduced_model.png'.format(aircraft.get_name()), dpi=160) 


if __name__ == "__main__":
    seance_3(dyn.Param_A319())
    plt.show()
