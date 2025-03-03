import fenics as fe
import ufl #needed to use exp, tanh etc. function for fenics code
import config as cfg
import os
import plot_results as plot

# Option if results should be stored
store_values = True

# Create output directories
if store_values:
    output_dir_phi = "output/phi_solutions"
    output_dir_n = "output/n_solutions"
    os.makedirs(output_dir_phi, exist_ok=True)
    os.makedirs(output_dir_n, exist_ok=True)

# name for txt document to store parameter values
txt_name = "Parameter_values.txt"


phi_init_option = "gaussian"
n_init_option = "gaussian"

n_low_threshold = 0.5
n_high_threshold = 1


config = cfg.Config(mu = -0.1)

phi_k, n_k = config.set_ics(phi_init_option, n_init_option)
phi_0 = phi_k.copy(deepcopy = True)
n_0 = n_k.copy(deepcopy = True)

# Define variational formulation
phi = fe.Function(config.V)
n = fe.Function(config.V)
v = fe.TestFunction(config.V)

# Allen-Cahn equation for phi
F_phi = ((phi - phi_k) / config.dt ) * v * fe.dx \
        + fe.inner(fe.grad(phi), fe.grad(v)) * fe.dx \
        - (phi - phi**3 - config.eps * n + config.mu) * v * fe.dx 
J_phi = fe.derivative(F_phi, phi)

# Ink distribution diffusion equation
F_n = ((n - n_k) / config.dt) * v * fe.dx \
      + (1/config.alpha) * config.D(phi,n) * fe.inner(fe.grad(n) + config.eps * n * fe.grad(phi), fe.grad(v)) * fe.dx
J_n = fe.derivative(F_n, n)

# Define the free energy
free_energy = (-(phi**2)/2 + (phi**4)/4 + 1/2*fe.inner(fe.grad(phi), fe.grad(phi)) \
               - config.mu*phi + config.eps*n*phi + n*(ufl.ln(n) - 1))*fe.dx


# Set up solvers
problem_phi = fe.NonlinearVariationalProblem(F_phi, phi, J=J_phi)
solver_phi = fe.NonlinearVariationalSolver(problem_phi)

problem_n = fe.NonlinearVariationalProblem(F_n, n, J=J_n)
solver_n = fe.NonlinearVariationalSolver(problem_n)

#initial values
phi_solutions = [phi_0]
n_solutions = [n_0]
free_energy_vals = []

times_to_plot = [0, config.num_steps // 2, config.num_steps]


# Create output files for ParaView
if store_values:
    """
    phi_xdmf = fe.XDMFFile(os.path.join(output_dir_phi, "phi_solution.xdmf"))
    n_xdmf = fe.XDMFFile(os.path.join(output_dir_n, "n_solution.xdmf"))
    
    phi_xdmf.parameters["flush_output"] = True
    n_xdmf.parameters["flush_output"] = True
    phi_xdmf.parameters["functions_share_mesh"] = True
    n_xdmf.parameters["functions_share_mesh"] = True

    phi_xdmf.write(phi_k, 0)
    n_xdmf.write(n_k, 0)
    """

    phi_file = fe.File(os.path.join(output_dir_phi, "phi_solution.pvd"))
    n_file = fe.File(os.path.join(output_dir_n, "n_solution.pvd"))

    # Parameters or txt file
    params = f"""
    Simulation Parameter:
    ---------------------
    Total time: {config.T}
    Time step number: {config.num_steps}
    Time step size: {config.dt}
    Moisture parameter (mu): {config.mu}
    Domain length: {config.L}
    Ink solvent interaction (eps) {config.eps}
    Evaporation coefficient * surface tension = M * sigma = {config.alpha}
    Mesh Resolution: {config.nx} x {config.ny}"""

    param_file_phi = os.path.join(output_dir_phi, txt_name)
    param_file_n = os.path.join(output_dir_n, txt_name)

    with open(param_file_phi, "w") as f:
        f.write(params)
    with open(param_file_n, "w") as f:
        f.write(params)


for i in range(config.num_steps):
    t = (i+1)*config.dt
    solver_phi.solve()
    phi_k.assign(phi)
    if store_values:
        #phi_xdmf.write(phi_k, t)
        phi_file << (phi, i * config.dt)

    solver_n.solve()
    n_k.assign(n)
    if store_values:
        #n_xdmf.write(n_k, t)
        n_file << (n, i * config.dt)

    if i in times_to_plot:
        phi_solutions.append(phi.copy(deepcopy=True))
        n_solutions.append(n.copy(deepcopy=True))

    free_energy_vals.append(fe.assemble(free_energy))

#if store_values:
#    phi_xdmf.close()
#    n_xdmf.close()

figure_handler = plot.FigureHandler(config)
save_heatmap = True
save_free_energy = True
save_slices = True

figure_handler.plot_heatmaps(phi_solutions, n_solutions, times_to_plot, save_heatmap)
figure_handler.plot_free_energy(free_energy_vals, save_free_energy)
figure_handler.plot_horizontal_slice_n(n_solutions, times_to_plot, 12.5, 100, save_slices)