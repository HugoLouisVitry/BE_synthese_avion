#-*- coding: utf-8 -*-
'''

Utilities

'''
import math, numpy as np
import matplotlib.pyplot as plt
import pdb

"""
Unit convertions
"""
# http://en.wikipedia.org/wiki/Nautical_mile
def m_of_NM(nm): return nm*1852.
def NM_of_m(m): return m/1852.
# http://en.wikipedia.org/wiki/Knot_(speed)
def mps_of_kt(kt): return kt*0.514444
def kt_of_mps(mps): return mps/0.514444


"""
International Standard Atmosphere Model
see: http://en.wikipedia.org/wiki/International_Standard_Atmosphere
"""
_name, _h0, _z0, _a, _T0, _p0 = np.arange(6)
#      name         h(m)    z(km)    a(K/m)    T0(K)     p0(Pa)
isa_param = \
[['Troposphere',       0,   0.0,    -6.5e-3,   288.15,   101325],
 ['Tropopause',    11000,  11.019,   0.0e-3,   216.65,    22632],
 ['Stratosphere',  20000,  20.063,   1.0e-3,   216.65,     5474.9],
 ['Stratosphere',  32000,  32.162,   2.8e-3,   228.65,      868.02],
 ['Stratopause',   47000,  47.350,   0.0e-3,   270.65,      110.91],
 ['Mesosphere',    51000,  51.413,  -2.8e-3,   270.65,       66.939],
 ['Mesosphere',    71000,  71.802,  -2.0e-3,   214.65,        3.9564],
 ['Mesopause',     84852,  86.000,   0.,       186.87,        0.3734]]

def isa(h):
    layer = 0
    while isa_param[layer][_h0] < h: layer+=1
    if layer == 0: layer = 1 # in case h<= 0
    name, h0, z0, a, T0, p0 = isa_param[layer-1]
    dh = h - h0
    T = T0 + a*dh
    g, R = 9.81, 287.05
    if a != 0.:
        p = p0*math.pow(T/T0, -g/a/R)
    else:
        p = p0*math.exp(-g/R/T0*dh)
    rho = p/R/T
    return p, rho, T


"""
Compute numerical jacobian 
"""
def num_jacobian(X, U, P, dyn):
    s_size = len(X)
    i_size = len(U)
    epsilonX = (0.1*np.ones(s_size)).tolist()
    dX = np.diag(epsilonX)
    A = np.zeros((s_size, s_size))
    for i in range(0, s_size):
        dx = dX[i,:]
        delta_f = dyn(X+dx/2, 0, U, P) - dyn(X-dx/2, 0, U, P)
        delta_f = delta_f / dx[i]
        A[:,i] = delta_f

    epsilonU = (0.1*np.ones(i_size)).tolist()
    dU = np.diag(epsilonU)
    B = np.zeros((s_size,i_size))
    for i in range(0, i_size):
        du = dU[i,:]
        delta_f = dyn(X, 0, U+du/2, P) - dyn(X, 0, U-du/2, P)
        delta_f = delta_f / du[i]
        B[:,i] = delta_f
        
    return A,B


"""
Plotting
"""
def decorate(ax, title=None, xlab=None, ylab=None, legend=None, xlim=None, ylim=None, min_yspan=None):
    ax.xaxis.grid(color='k', linestyle='-', linewidth=0.2)
    ax.yaxis.grid(color='k', linestyle='-', linewidth=0.2)
    if xlab: ax.xaxis.set_label_text(xlab)
    if ylab: ax.yaxis.set_label_text(ylab)
    if title: ax.set_title(title, {'fontsize': 20 })
    if legend != None: ax.legend(legend, loc='best')
    if xlim != None: ax.set_xlim(xlim[0], xlim[1])
    if ylim != None: ax.set_ylim(ylim[0], ylim[1])
    if min_yspan != None: ensure_yspan(ax, min_yspan)

def ensure_yspan(ax, yspan):
    ymin, ymax = ax.get_ylim()
    if ymax-ymin < yspan:
        ym =  (ymin+ymax)/2
        ax.set_ylim(ym-yspan/2, ym+yspan/2)

def prepare_fig(fig=None, window_title=None, figsize=(20.48, 10.24), margins=None):
    if fig == None:
        fig = plt.figure(figsize=figsize)
    else:
        plt.figure(fig.number)
    if margins:
        left, bottom, right, top, wspace, hspace = margins
        fig.subplots_adjust(left=left, right=right, bottom=bottom, top=top,
                            hspace=hspace, wspace=wspace)
    if window_title:
         fig.canvas.set_window_title(window_title)
    return fig

def savefig(filename):
    if filename is not None: plt.savefig(filename, dpi=160)
