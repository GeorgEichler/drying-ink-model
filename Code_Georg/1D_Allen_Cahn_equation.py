'''
Numerical analysis of nondimensionalised Allen-Cahn equation using periodic bounddary conditions
partial phi/partial t = partial^2 phi/partial x^2  + phi - phi^3 + mu
'''

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import scipy.sparse as sp



#discretisation of second derivative for non-periodic boundary conditions
def sparse_matrix(N):
    main_diagonal = np.concatenate([[-1],-2*np.ones(N-2),[-1]])
    off_diagonal = np.ones(N - 1)
    #format csr for fast matrix vector multiplication
    A = sp.diags([main_diagonal, off_diagonal, off_diagonal], [0, -1, 1], format = "csr")

    return A


def periodic_boundary_fun(t, y, mu, h):
    ye = np.roll(y, 1)  #shift y to right
    yw = np.roll(y, -1) #shift y to left

    d2yd2x = (yw - 2*y + ye)/h**2
    dydt = d2yd2x + y - y**3 + mu

    return dydt

def non_periodic_boundary_fun(t, y, mu, h, boundary_condition):
    N = len(y)

    A = sparse_matrix(N) 
    b = np.concatenate([[-1*boundary_condition[0]],np.zeros(N-2) ,[boundary_condition[1]]])

    dydt = 1/h**2*A.dot(y) + 1/h*b + y - np.power(y, 3) + mu
    

    return dydt

L = 1   #length of domain [0,L]
N = 64  #number of grid points
h = L/N #grid spacing
boundary_condition = [0, 0]

x_grid = np.linspace(start = h, stop = L - h, num = N)

#phi0 = -0.1*np.ones(N)
#phi0 = np.concatenate([-0.1*np.ones(N//2),0.1*np.ones(N//2)])
#phi0 = np.power(x_grid, 1) - 1/2
phi0 = np.random.uniform(-0.1, 0.1, N)
#phi0 = np.cos(2*np.pi*x_grid)
#phi0 = np.tanh((x_grid - 0.5) * 10)

mu = 0 #moisture parameter
t_span = [0, 1]
t_eval = np.linspace(0, 1, 101)

#solution = solve_ivp(periodic_boundary_fun, t_span = t_span, y0 = phi0, method = "BDF", t_eval = t_eval, args = (mu,h))

solution = solve_ivp(non_periodic_boundary_fun, t_span = t_span, y0 = phi0, method = "BDF",
                     t_eval = t_eval, args = (mu,h, boundary_condition))

for i in range(0, 101, 20):
    plt.plot(x_grid, solution.y[:, i], label = f't = {i/(len(t_eval)-1)*t_span[1]:.2f}')

plt.xlabel('x')
plt.ylabel(r'$\phi$')
plt.title("Allen-Cahn evolution")
plt.legend()
plt.show()