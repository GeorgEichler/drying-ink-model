import numpy as np
import fenics as fe
import matplotlib.pyplot as plt
import os

# Create output directories
output_dir_phi = "output/phi_solutions"
output_dir_n = "output/n_solutions"
os.makedirs(output_dir_phi, exist_ok=True)
os.makedirs(output_dir_n, exist_ok=True)

# Parameter
T = 10                # time interval length
num_steps = 50        # number of time steps
dt = T / num_steps    # time step size
mu = 0.0              # moisture parameter
L = 10                # domain length [0,L]x[0,L]

eps = 1e-1            # ink-solvent interaction parameter
alpha = 0.2 * 0.03    # Product of evaporation coefficient (M) and surface tension (sigma)

# Create mesh and funcion space
nx, ny = 50, 50
mesh = fe.RectangleMesh(fe.Point(0,0), fe.Point(L,L), nx, ny)
V = fe.FunctionSpace(mesh, "Lagrange", 1)

# Initial condotions for phi and n
phi_k = fe.interpolate(fe.Expression("0.2*cos(pi*(x[0]+x[1])/5)", degree=2),V)
phi_0 = phi_k.copy(deepcopy=True)

n_k = fe.interpolate(fe.Expression("0.1*exp(-pow(x[0]-L/2, 2)/0.01) * exp(-pow(x[1]-L/2, 2)/0.01)", L=L, degree=2), V)
n_0 = n_k.copy(deepcopy=True)

# Define mobility function
D = lambda phi: (1 + phi)**3

# Define variational formulation
phi = fe.Function(V)
n = fe.Function(V)
v = fe.TestFunction(V)

# Allen-Cahn equation for phi
F_phi = ((phi - phi_k) / dt ) * v * fe.dx \
        + fe.inner(fe.grad(phi), fe.grad(v)) * fe.dx \
        - (phi - phi**3 - eps * n + mu) * v * fe.dx 
J_phi = fe.derivative(F_phi, phi)

# Ink distribution diffusion equation
F_n = ((n - n_k) / dt) * v * fe.dx \
      + (1/alpha) * D(phi) * fe.inner(fe.grad(n) + eps * n * fe.grad(phi), fe.grad(v)) * fe.dx
J_n = fe.derivative(F_n, n)

# Set up solvers
problem_phi = fe.NonlinearVariationalProblem(F_phi, phi, J=J_phi)
solver_phi = fe.NonlinearVariationalSolver(problem_phi)

problem_n = fe.NonlinearVariationalProblem(F_n, n, J=J_n)
solver_n = fe.NonlinearVariationalSolver(problem_n)

# Create output files for ParaView
phi_file = fe.File(os.path.join(output_dir_phi, "phi_solution.pvd"))
n_file = fe.File(os.path.join(output_dir_n, "n_solution.pvd"))

# Time-stepping loop
phi_solutions = [phi_0]
n_solutions = [n_0]

times_to_plot = [0, num_steps // 2, num_steps - 1]

for i in range(num_steps):
    solver_phi.solve()
    phi_k.assign(phi)
    phi_file << (phi, i * dt)

    solver_n.solve()
    n_k.assign(n)
    n_file << (n, i * dt)

    if i in times_to_plot:
        phi_solutions.append(phi.copy(deepcopy=True))
        n_solutions.append(n.copy(deepcopy=True))

# Visualisation
fig, axes = plt.subplots(2, len(times_to_plot), figsize=(15, 6))
cmap = "coolwarm"

for idx, t_idx in enumerate(times_to_plot):
    plt.sca(axes[0, idx])
    c_phi = fe.plot(phi_solutions[idx], mode="color")
    c_phi.set_cmap(cmap)
    axes[0, idx].set_title(f"phi at t={t_idx*dt}")

    plt.sca(axes[1, idx])
    c_n = fe.plot(n_solutions[idx], mode="color")
    c_n.set_cmap(cmap)
    axes[1, idx].set_title(f"n at t={t_idx*dt}")

plt.tight_layout()
plt.show()