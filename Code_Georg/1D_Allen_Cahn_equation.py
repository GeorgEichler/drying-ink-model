'''
Numerical analysis of nondimensionalised Allen-Cahn equation using periodic bounddary conditions
partial phi/partial t = partial^2 phi/partial x^2  + phi - phi^3 + mu
'''

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def fun(t, y, mu, h):
    ye = np.roll(y, 1)  #shift y to right
    yw = np.roll(y, -1) #shift y to left

    d2yd2x = (yw - 2*y + ye)/h**2
    dydt = d2yd2x + y - y**3 + mu

    return dydt