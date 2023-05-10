#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np, matplotlib.pyplot as plt
import utils, dynamic

if __name__ == "__main__":
    
    v_Kt = np.array([100, 150, 200])
    v_ms = utils.mps_of_kt(v_Kt)
    print('{} Kt -> {} m/s'.format(v_Kt, v_ms))

    h = 5000
    p, rho, T = utils.isa(h)
    print('atmosphère normalisée à {} m: {:.1f} Pa {:.3} kg/m3 {:.2f} K'.format(h, p, rho, T))

    hs = np.linspace(0, 10000, 20)
    isa = np.array([utils.isa(h) for h in hs])
    plt.plot(hs, isa[:,0]/100)
    plt.xlabel('altitude en m'); plt.ylabel('pression en Hectopascal');
    plt.savefig('./images/isa_pressure.png', dpi=120)

    p = dynamic.Param_A320()
    print('longueur du fuselage de l\'{}: {}m'.format(p.name, p.l_fus))


    plt.show()
