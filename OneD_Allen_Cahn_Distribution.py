import ufl
from dolfin import *
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

plt.rcParams.update({
            "axes.titlesize": 18,
            "axes.labelsize": 16,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 14,
            "figure.dpi": 100 #change resolution, standard is 100
        })


def random(nx,V):
    random_values = np.random.uniform(-0.1, 0.1, (nx+1))
    phi_init = Function(V)
    phi_init.vector()[:] = random_values.flatten()
    interpolate(phi_init, V)
    return phi_init

def D1(phi):
    return 0.5*(1 + ufl.tanh(10*phi))

def D(phi,n):
    n_low_threshold = 0.2
    n_high_threshold = 1
    return 0.5*(1 + ufl.tanh(10*phi))*(1 + ufl.tanh(10 * (n - n_low_threshold) * (n_high_threshold - n)))

# Parameters
T = 10              # Time intervall length
num_steps = 50      # number of time steps
dt = T / num_steps  # time step size
mu = 0.5            # moisture parameter
L = 50              # domain length [0,L]
eps = -1          # how much the ink likes to be in the solvent
alpha = 0.2*0.03    # product of evaporation coefficient (M) and surface tension (sigma)

# Mesh and function space (linear Lagrange elements and 1d interval)
nx = 1000
mesh = IntervalMesh(nx, 0, L)

def solve_1D(T = 10, num_steps = 50, mu = 0, L = 50, eps = -1, alpha = 0.2*0.03, nx = 1000):

    dt = T/num_steps
    mesh = IntervalMesh(nx, 0, L)
    V = FunctionSpace(mesh, "Lagrange", 1)

    # initial conditions
    #phi_k = interpolate(Expression("0.2*tanh(x[0]-5)", degree=2), V)
    #phi_k = interpolate(Expression("x[0] < L/2 ? -0.1 : 0.1", L = L, degree=1),V)
    #phi_k = interpolate(Expression("2*exp(-pow(x[0]-L/2, 2)/10)-1", L=L, degree=2), V)
    phi_k = random(nx,V)
    phi_0 = phi_k.copy(deepcopy=True)

    #n_k = interpolate(Constant(1/L), V)
    #n_k = interpolate(Expression("x[0] < L/2 ? 0.2 : 0", L=L, degree=1), V)
    n_k = interpolate(Constant(0), V)
    #n_k = interpolate(Expression("1*exp(-pow(x[0]-L/2, 2)/10)", L=L, degree=2), V)
    n_0 = n_k.copy(deepcopy=True)


    # weak formulation of problem
    phi = Function(V)
    v = TestFunction(V)
    F_phi = (phi - phi_k)/dt*v*dx + inner(grad(phi), grad(v))*dx - (phi  - phi**3 - eps*n_k + mu)*v*dx
    J_phi = derivative(F_phi, phi)

    n = Function(V)
    F_n = (n - n_k)/dt*v*dx + (1/alpha)*D(phi,n)*inner(grad(n) + eps*n*grad(phi), grad(v))*dx
    J_n = derivative(F_n, n)

    # set up the solvers
    problem_phi = NonlinearVariationalProblem(F_phi, phi, J=J_phi)
    solver_phi = NonlinearVariationalSolver(problem_phi)

    problem_n = NonlinearVariationalProblem(F_n, n, J=J_n)
    solver_n = NonlinearVariationalSolver(problem_n)

    #Iterate for solution
    iterates_phi = [phi_0]
    iterates_n = [n_0]

    for i in range(num_steps):
        solver_phi.solve()
        phi_k.assign(phi)
        iterates_phi.append(phi_k.copy(deepcopy=True))

        solver_n.solve()
        n_k.assign(n)
        iterates_n.append(n_k.copy(deepcopy=True))

    return phi_0, n_0, iterates_phi, iterates_n


# Preparations for the plots
x = mesh.coordinates().flatten()
t = np.linspace(0, T, num_steps+1)
X, T_grid = np.meshgrid(x, t)

"""
epsilons = [-0.1,-0.5,-1]
fig, axes = plt.subplots(1, 2, figsize = (16,8))

for eps in epsilons:
    phi_0, n_0, iterates_phi, iterates_n = solve_1D(T = 10, num_steps=50, eps=eps,mu = -0.5)
    axes[0].plot(x, iterates_phi[-1].compute_vertex_values(), label = f"$\epsilon=${eps}")
    axes[1].plot(x, iterates_n[-1].compute_vertex_values(), label = f"$\epsilon=${eps}")

axes[0].plot(x, phi_0.compute_vertex_values(), linestyle = "dotted", label = "initial")
axes[0].set_xlabel("x")
axes[0].set_ylabel("$\phi$")
axes[0].legend()

axes[1].plot(x, n_0.compute_vertex_values(), linestyle = "dotted", label = "initial")
axes[1].set_xlabel("x")
axes[1].set_ylabel("n")
axes[1].legend()

plt.tight_layout()
plt.show()

"""
phi_0, n_0, iterates_phi, iterates_n = solve_1D(eps=0)


plotted_timesteps = [num_steps //2, num_steps]

plt.subplots(1, 2, figsize = (15,8))

plt.subplot(1, 2, 1)
plt.plot(x, phi_0.compute_vertex_values(), label="t=0")
for j in plotted_timesteps:
    plt.plot(x, iterates_phi[j].compute_vertex_values(), label=f"t={j*dt}")

plt.xlim(0,L)
#plt.ylim(-1.1, 1.1)
plt.xlabel("x")
plt.ylabel("$\phi$")
plt.title("Solution of 1D Allen-Cahn\n equation (order parameter)")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(x, n_0.compute_vertex_values(), label="t=0")
for j in plotted_timesteps:
    plt.plot(x, iterates_n[j].compute_vertex_values(), label=f"t={j*dt}")
plt.xlim(0, L)
plt.xlabel("x")
plt.ylabel("n")
plt.title("Solution of 1D Allen-Cahn\n equation (distribution)")
plt.legend()

phi_values = np.array([phi.compute_vertex_values() for phi in iterates_phi])
n_values = np.array([n.compute_vertex_values() for n in iterates_n])

fig1 = plt.figure(figsize=(12,6))

ax1 = fig1.add_subplot(111, projection = '3d')
ax1.plot_surface(X, T_grid, phi_values, color="blue")
ax1.set_xlabel('x')
ax1.set_ylabel('t')
ax1.set_zlabel('$\phi$')
ax1.set_title('Evolution of $\phi$')
#ax.view_init(elev=30, azim=180)  # Rotate to bring x-axis to front
ax1.view_init(elev=30, azim=-90) #Rotate to bring y-axis to front

fig2 = plt.figure(figsize=(12,6))

ax2 = fig2.add_subplot(111, projection='3d')
ax2.plot_surface(X, T_grid, n_values, color='grey')
ax2.set_xlabel('x')
ax2.set_ylabel('t')
ax2.set_zlabel('n')
ax2.set_title('Evolution of $n$')
ax2.set_zlim(0,1)


plt.tight_layout()


plt.show()