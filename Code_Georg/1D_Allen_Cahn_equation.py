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

L = 1   #length of domain [0,L]
N = 64  #number of grid points
h = L/N #grid spacing

x_grid = np.linspace(start = h, stop = L - h, num = N)
phi0 = 0.2*np.ones(N)

mu = 0 #moisture parameter
t_span = [0, 1]
t_eval = np.linspace(0, 1, 101)

solution = solve_ivp(fun, t_span = t_span, y0 = phi0, method = "BDF", t_eval = t_eval, args = (mu,h))

for t in range(0, 101, 20):
    plt.plot(x_grid, solution.y[:, t], label = f't = {t}')

plt.xlabel('x')
plt.ylabel(r'$\phi$')
plt.title("Allen-Cahn evolution")
plt.legend()
plt.show()